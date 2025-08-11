"""
规则化立场检测与时间步聚合导出

功能：
- 从 SQLite 数据库读取模拟产生的社交平台帖子数据（表 `post`）
- 通过关键词/规则的 NLP（不使用 LLM）判别帖子是否“支持/反对/中立”，并识别“告知者/澄清/声明”类帖子
- 根据 `created_at` 推断时间步 `timestep`（对唯一时间戳做稠密排名 1..T），每个时间步聚合计算 `support_ratio`
- 自动检测首次出现“告知者”帖子的时间步 `claim_step`（若未出现则为 -1），并在 CSV 中作为一列
- 输出 CSV 仅包含三列：`timestep`, `support_ratio`, `claim_step`

用法示例：
  python -m cim.core.stance_detector \
    --db_path data/processed/twitter_simulation.db \
    --output_csv data/output/stance_by_timestep.csv

可选参数：
  --only_original           仅统计原创帖（默认 True）
  --exclude_neutral         支持率分母是否排除“中立”（默认 False，即分母=该步所有帖子数）

依赖：pandas
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# ===========================
# 关键词规则（可按需扩展）
# ===========================
SUPPORT_KEYWORDS_ZH: List[str] = [
    "支持", "赞成", "同意", "认可", "拥护", "鼓励", "推荐", "力挺", "点赞",
    "继续购买", "继续使用", "正确信息", "合理", "应该", "我支持",
]
SUPPORT_KEYWORDS_EN: List[str] = [
    "support", "agree", "approve", "endorse", "back", "in favor", "pro",
]

OPPOSE_KEYWORDS_ZH: List[str] = [
    "反对", "抵制", "不同意", "反驳", "谴责", "拒绝", "取缔", "禁止", "不应该",
    "误导", "造谣", "虚假", "不实", "谣言",
]
OPPOSE_KEYWORDS_EN: List[str] = [
    "oppose", "against", "boycott", "ban", "stop", "reject", "refuse",
]

# 标记“告知者/澄清/声明/辟谣”信息的关键词
INFORMER_KEYWORDS_ZH: List[str] = [
    "告知", "澄清", "声明", "公告", "通报", "提醒", "说明", "更正", "辟谣", "事实是",
]
INFORMER_KEYWORDS_EN: List[str] = [
    "clarify", "clarification", "correction", "announce", "announcement",
    "official", "statement", "claim",
]


@dataclass
class DetectorConfig:
    only_original: bool = True
    exclude_neutral: bool = False  # True: 支持率分母仅统计支持/反对；False: 分母为该步所有帖子


class RuleBasedStanceDetector:
    """基于关键词规则的立场检测器（不使用 LLM）。"""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        # 预编译正则，忽略大小写
        self._re_support = self._compile_any(SUPPORT_KEYWORDS_ZH + SUPPORT_KEYWORDS_EN)
        self._re_oppose = self._compile_any(OPPOSE_KEYWORDS_ZH + OPPOSE_KEYWORDS_EN)
        self._re_informer = self._compile_any(INFORMER_KEYWORDS_ZH + INFORMER_KEYWORDS_EN)

    @staticmethod
    def _compile_any(keywords: Iterable[str]) -> re.Pattern:
        escaped = [re.escape(k) for k in keywords if k]
        if not escaped:
            return re.compile(r"^$")  # 匹配不到任何文本
        pattern = r"(" + r"|".join(escaped) + r")"
        return re.compile(pattern, flags=re.IGNORECASE)

    def classify(self, text: str) -> Tuple[str, bool]:
        """对单条文本进行规则分类。

        返回：(stance, is_informer)
          - stance ∈ {"support", "oppose", "neutral"}
          - is_informer: 是否为“告知/澄清/声明”等信息发布类帖子
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral", False

        s = text.strip()
        has_support = bool(self._re_support.search(s))
        has_oppose = bool(self._re_oppose.search(s))
        is_informer = bool(self._re_informer.search(s))

        if has_support and not has_oppose:
            stance = "support"
        elif has_oppose and not has_support:
            stance = "oppose"
        elif has_support and has_oppose:
            # 简化处理：同时出现时视为中立/不确定
            stance = "neutral"
        else:
            stance = "neutral"

        return stance, is_informer


# ===========================
# 数据库读取与时间步推断
# ===========================


def read_posts(db_path: str, only_original: bool = True) -> pd.DataFrame:
    """从 SQLite 读取帖子。

    读取字段：post_id, user_id, content, created_at, original_post_id
    若 only_original=True 则过滤 original_post_id IS NULL。
    """
    conn = sqlite3.connect(db_path)
    try:
        base_sql = (
            "SELECT post_id, user_id, content, created_at, original_post_id "
            "FROM post "
        )
        where = "WHERE original_post_id IS NULL" if only_original else ""
        order = "ORDER BY created_at ASC, post_id ASC"
        sql = f"{base_sql} {where} {order}"
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def assign_timesteps(df: pd.DataFrame) -> pd.DataFrame:
    """基于 created_at 推断时间步。

    策略：
    - 将 created_at 解析为时间（to_datetime，errors='coerce'）
    - 对非空时间做去重排序后进行稠密排名（1..T），作为 timestep
    - 若全部解析失败，则按原行顺序稠密分配（每个不同 created_at 视为一个时间点）
    """
    if df.empty:
        return df.assign(timestep=pd.Series(dtype=int))

    ts = pd.to_datetime(df["created_at"], errors="coerce")
    if ts.notna().any():
        # 使用非空时间做稠密排名
        # 注意：同一时间戳将映射到同一时间步
        order = ts.rank(method="dense").astype("Int64")
        # 若有 NaT，则为其填入最近的时间步或新开时间步；这里简单地用前向填充后再填后向
        order = order.ffill().bfill().astype(int)
        df = df.copy()
        df["timestep"] = order
        return df
    else:
        # created_at 全部无法解析，则按行号稠密排名（稳定排序）
        df = df.copy()
        df["timestep"] = pd.RangeIndex(start=1, stop=len(df) + 1)
        return df


def compute_support_by_timestep(
    df: pd.DataFrame,
    detector: RuleBasedStanceDetector,
) -> Tuple[pd.DataFrame, int]:
    """对带有 `timestep` 的帖子表进行逐步聚合，计算支持率并检测首个告知者步骤。

    返回：
      - agg_df: 列包含 ["timestep", "support_ratio", "claim_step"]
      - claim_step: 首个出现告知者的时间步（未出现则为 -1）
    """
    if df.empty:
        return pd.DataFrame(columns=["timestep", "support_ratio", "claim_step"]), -1

    # 分类
    stance_list: List[str] = []
    informer_flags: List[bool] = []
    for text in df["content"].astype(str).tolist():
        stance, is_inf = detector.classify(text)
        stance_list.append(stance)
        informer_flags.append(is_inf)

    df = df.copy()
    df["stance"] = stance_list
    df["is_informer"] = informer_flags

    # 首次告知者时间步
    claim_step: int = -1
    tmp = df.loc[df["is_informer"], "timestep"]
    if not tmp.empty:
        claim_step = int(tmp.min())

    # 逐步聚合
    rows: List[dict] = []
    for t, sub in df.groupby("timestep", sort=True):
        total = len(sub)
        if detector.config.exclude_neutral:
            denom = int((sub["stance"] != "neutral").sum())
        else:
            denom = total
        support_cnt = int((sub["stance"] == "support").sum())
        ratio = (support_cnt / denom) if denom > 0 else 0.0
        rows.append({
            "timestep": int(t),
            "support_ratio": float(ratio),
            "claim_step": int(claim_step),  # 全表一致，逐行复写，保持与示例 CSV 风格一致
        })

    agg_df = pd.DataFrame(rows).sort_values("timestep").reset_index(drop=True)
    return agg_df, claim_step


def export_csv(agg_df: pd.DataFrame, out_path: str) -> str:
    """导出聚合结果到 CSV（仅三列）。"""
    cols = ["timestep", "support_ratio", "claim_step"]
    agg_df[cols].to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based stance detection and timestep CSV export")
    parser.add_argument("--db_path", type=str, required=True, help="SQLite 数据库路径（含 post 表）")
    parser.add_argument("--output_csv", type=str, required=True, help="输出 CSV 路径")
    parser.add_argument("--only_original", action="store_true", default=True, help="仅统计原创帖（默认开启）")
    parser.add_argument("--include_reposts", dest="only_original", action="store_false", help="包含转发/引用")
    parser.add_argument("--exclude_neutral", action="store_true", help="支持率分母排除中立/未判定")

    args = parser.parse_args()

    cfg = DetectorConfig(only_original=args.only_original, exclude_neutral=args.exclude_neutral)
    detector = RuleBasedStanceDetector(cfg)

    # 1) 读取 & 时间步
    posts = read_posts(args.db_path, only_original=cfg.only_original)
    posts = assign_timesteps(posts)

    # 2) 聚合
    agg_df, claim_step = compute_support_by_timestep(posts, detector)

    # 3) 导出
    out_csv = export_csv(agg_df, args.output_csv)

    print(f"✓ 导出 CSV: {out_csv}")
    if claim_step == -1:
        print("- 告知者步骤: 未出现（claim_step = -1）")
    else:
        print(f"- 告知者步骤: {claim_step}")


if __name__ == "__main__":
    main()
