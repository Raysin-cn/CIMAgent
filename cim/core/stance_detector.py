"""
基于 BERT（Zero-shot NLI）的立场检测与时间步聚合导出

功能：
- 从 SQLite 数据库读取模拟产生的社交平台帖子数据（表 `post`）
- 使用 Hugging Face Transformers 的 zero-shot 分类（MNLI）判别帖子“支持/反对/中立”，并可选检测“告知者/澄清/声明”类帖子
- 根据 `created_at` 推断时间步 `timestep`（对唯一时间戳做稠密排名 1..T），每个时间步聚合计算 `support_ratio`
- 自动检测首次出现“告知者”帖子的时间步 `claim_step`（若未出现则为 -1），并在 CSV 中作为一列
- 输出 CSV 仅包含三列：`timestep`, `support_ratio`, `claim_step`

用法示例：
  python -m cim.core.stance_detector \
    --db_path data/processed/twitter_simulation.db \
    --output_csv data/output/stance_by_timestep.csv \
    --model_name typeform/distilbert-base-uncased-mnli \
    --device cuda:0

可选参数：
- --only_original           仅统计原创帖（默认 True）
- --exclude_neutral         支持率分母是否排除“中立”（默认 False，即分母=该步所有帖子数）
- --model_name              Hugging Face 模型名或本地路径（默认 DistilBERT MNLI）
- --batch_size              推理批大小（默认 16）
- --device                  推理设备，如 cpu、cuda:0（默认自动探测）

依赖：pandas、transformers、torch
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sqlite3 as _sqlite3  # for claim_step extraction from trace

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


@dataclass
class DetectorConfig:
    """BERT Zero-shot 立场检测配置。"""

    only_original: bool = True
    exclude_neutral: bool = False  # True: 支持率分母仅统计支持/反对；False: 分母为该步所有帖子
    model_name_or_path: str = "facebook/bart-large-mnli"  # 英文 zero-shot 常用模型
    device: Optional[str] = None  # cpu / cuda:0 / auto
    batch_size: int = 16
    enable_informer: bool = True
    target_text: Optional[str] = None
    confidence_threshold: float = 0.5
    informer_threshold: float = 0.6


class BertZeroShotStanceDetector:
    """基于 MNLI zero-shot 的立场检测器。

    - 立场标签：support / oppose / neutral（单标签）
    - 告知者：informer / not informer（单标签，启用可选）
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

        # 设备选择
        if self.config.device is None:
            hf_device = 0 if torch.cuda.is_available() else -1
        else:
            if self.config.device == "cpu":
                hf_device = -1
            elif self.config.device.startswith("cuda"):
                # 解析 cuda:idx
                try:
                    idx = int(self.config.device.split(":")[1]) if ":" in self.config.device else 0
                except Exception:
                    idx = 0
                hf_device = idx
            else:
                hf_device = -1

        # 模型与分词器
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name_or_path)
        self._pipe = pipeline(
            task="zero-shot-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=hf_device,
        )

        # 标签与模板
        # 立场候选短语与模板（支持提供 target_text 提升准确率）
        if self.config.target_text:
            tgt = self.config.target_text.strip()
            self._stance_labels = [
                f"supports the claim '{tgt}'",
                f"opposes the claim '{tgt}'",
                f"is neutral or unrelated to the claim '{tgt}'",
            ]
        else:
            self._stance_labels = [
                "expresses support",
                "expresses opposition",
                "is neutral or unrelated",
            ]
        self._stance_to_canonical = {
            self._stance_labels[0]: "support",
            self._stance_labels[1]: "oppose",
            self._stance_labels[2]: "neutral",
        }
        self._stance_template = "This text {}."

        # 告知者检测短语与模板
        self._informer_labels = [
            "is an official clarification or announcement",
            "is not an official clarification or announcement",
        ]
        self._informer_template = "This text {}."

    def classify_texts(self, texts: List[str]) -> Tuple[List[str], List[bool]]:
        """对一批文本分类，返回 (stances, informer_flags)。"""
        if not texts:
            return [], []

        # 立场预测（单标签）
        stance_results = self._pipe(
            sequences=texts,
            candidate_labels=self._stance_labels,
            hypothesis_template=self._stance_template,
            multi_label=False,
            batch_size=self.config.batch_size,
            truncation=True,
        )
        if isinstance(stance_results, dict):  # 兼容单条输入返回字典
            stance_results = [stance_results]

        stances: List[str] = []
        for res in stance_results:
            # 取最高分标签并映射到 canonical；低置信度降为 neutral
            if "labels" in res and res["labels"]:
                top_label = res["labels"][0]
                top_score = float(res.get("scores", [0.0])[0])
                mapped = self._stance_to_canonical.get(top_label, "neutral")
                if top_score < self.config.confidence_threshold:
                    mapped = "neutral"
                stances.append(mapped)
            else:
                stances.append("neutral")

        # 告知者预测（可选）
        informer_flags: List[bool] = [False] * len(texts)
        if self.config.enable_informer:
            informer_results = self._pipe(
                sequences=texts,
                candidate_labels=self._informer_labels,
                hypothesis_template=self._informer_template,
                multi_label=False,
                batch_size=self.config.batch_size,
                truncation=True,
            )
            if isinstance(informer_results, dict):
                informer_results = [informer_results]
            for i, res in enumerate(informer_results):
                if "labels" in res and res["labels"]:
                    top_label = res["labels"][0]
                    top_score = float(res.get("scores", [0.0])[0])
                    informer_flags[i] = (top_label == self._informer_labels[0]) and (top_score >= self.config.informer_threshold)

        return stances, informer_flags


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
    - 对非空时间做去重排序后进行稠密排名（1..U），作为初始 timestep
    - 若全部解析失败，则按原行顺序稠密分配（每个不同 created_at 视为一个时间点）
    """
    if df.empty:
        return df.assign(timestep=pd.Series(dtype=int))

    # TODO 时间上理论上是[1,2,...,12]总共12步，每个created_at应该在这个范围内。
    # TODO 每个用户在每个时间步，都需要有一个post内容作为其立场状态。这里需要判断用户是否在某个时间步缺失content，如果缺失content，则需要填充一下该时间步前一个最近发布的content。
    ts = pd.to_datetime(df["created_at"], errors="coerce")
    if ts.notna().any():
        # 使用非空时间做稠密排名，并从0开始
        # 注意：同一时间戳将映射到同一时间步
        order = ts.rank(method="dense").astype("Int64") - 1
        # 若有 NaT，则为其填入最近的时间步或新开时间步；这里简单地用前向填充后再填后向
        order = order.ffill().bfill().astype(int)
        df = df.copy()
        df["timestep"] = order
        return df
    else:
        # created_at 全部无法解析，则按行号稠密排名（稳定排序）
        df = df.copy()
        df["timestep"] = pd.RangeIndex(start=0, stop=len(df))
        return df


def fill_missing_timesteps_with_ffill(
    df: pd.DataFrame,
    total_steps: int,
) -> pd.DataFrame:
    """按用户对齐 1..total_steps，并对缺失时间步前向填充上一条非空 content。

    约束与说明：
    - 仅前向填充，不进行后向填充。即：若某用户在其首条内容出现之前的时间步没有内容，则这些时间步仍然为空，将被丢弃。
    - 会丢弃 content 仍为缺失的行（例如该用户从未发过帖，或首条内容在较晚时间步，之前步无内容）。
    - 若 df 中存在超过 total_steps 的 timestep，将被裁剪至 1..total_steps 范围内。
    """
    if df.empty:
        return df

    if "timestep" not in df.columns:
        raise ValueError("fill_missing_timesteps_with_ffill 需要输入包含 'timestep' 列的 DataFrame")

    # 仅保留 1..total_steps 范围
    df = df.copy()
    df = df.loc[(df["timestep"].astype(int) >= 1) & (df["timestep"].astype(int) <= int(total_steps))]
    if df.empty:
        return df.assign(timestep=pd.Series(dtype=int))

    required_cols = ["user_id", "content", "timestep"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    filled_parts: list[pd.DataFrame] = []
    for user_id, sub in df.groupby("user_id", sort=False):
        if user_id == 0:
            continue
        # 对同一用户在同一时间步的多条记录进行去重：保留该步内“最新”的一条
        # 依据顺序：timestep 升序，created_at 升序，post_id 升序，然后取每个 timestep 的最后一条
        sort_cols = ["timestep"]
        if "created_at" in sub.columns:
            sort_cols.append("created_at")
        if "post_id" in sub.columns:
            sort_cols.append("post_id")
        sub = (
            sub.sort_values(sort_cols)
               .drop_duplicates(subset=["timestep"], keep="last")
               .sort_values("timestep")
               .copy()
        )
        observed_steps = set(int(x) for x in sub["timestep"].tolist())

        # 重建 1..total_steps 的索引
        full_index = pd.Index(range(1, int(total_steps) + 1), name="timestep")
        frame = (
            sub.set_index("timestep")
            .reindex(full_index)
        )
        frame["user_id"] = user_id

        # 仅前向填充 content
        frame["content"] = frame["content"].ffill()

        # 标注该时间步是否为观测到的原始帖
        frame["is_observed"] = frame.index.to_series().apply(lambda t: int(t) in observed_steps)

        # 丢弃 content 仍为空的行（该用户在此步之前从未发帖）
        frame = frame.reset_index()
        frame = frame.loc[frame["content"].notna()]

        filled_parts.append(frame[["user_id", "content", "timestep", "is_observed"]])

    if not filled_parts:
        return df.assign(timestep=pd.Series(dtype=int))

    out = pd.concat(filled_parts, ignore_index=True)
    out["timestep"] = out["timestep"].astype(int)
    return out


def compute_support_by_timestep(
    df: pd.DataFrame,
    detector: BertZeroShotStanceDetector,
) -> Tuple[pd.DataFrame, int]:
    """对带有 `timestep` 的帖子表进行逐步聚合，计算支持率并检测首个告知者步骤。

    返回：
      - agg_df: 列包含 ["timestep", "support_ratio", "claim_step"]
      - claim_step: 首个出现告知者的时间步（未出现则为 -1）
    """
    if df.empty:
        return pd.DataFrame(columns=["timestep", "support_ratio", "claim_step"]), -1

    # 分类（批量）
    texts: List[str] = df["content"].astype(str).tolist()
    stance_list, informer_flags = detector.classify_texts(texts)

    df = df.copy()
    df["stance"] = stance_list
    df["is_informer"] = informer_flags

    # 旧的告知者时间步逻辑废除：由外部确定后再覆盖
    claim_step: int = -1

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


def read_claim_step_from_trace(db_path: str) -> int:
    """从 SQLite 的 `trace` 表读取用户 0 的 `send_to_group` 动作的 created_at，作为 claim_step。

    若不存在，则返回 -1。
    该 created_at 在模拟中等同于时间步编号。
    """
    conn = _sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT created_at
            FROM trace
            WHERE user_id = 0 AND action = 'send_to_group'
            ORDER BY created_at ASC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if row is None:
            return -1
        # created_at 记录的是整数时间步
        try:
            return int(row[0])
        except Exception:
            return -1
    finally:
        conn.close()


def export_csv(agg_df: pd.DataFrame, out_path: str) -> str:
    """导出聚合结果到 CSV（仅三列）。"""
    cols = ["timestep", "support_ratio", "claim_step"]
    agg_df[cols].to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT zero-shot stance detection and timestep CSV export")
    parser.add_argument("--db_path", type=str, required=True, help="SQLite 数据库路径（含 post 表）")
    parser.add_argument("--output_csv", type=str, required=True, help="输出 CSV 路径")
    parser.add_argument("--only_original", action="store_true", default=True, help="仅统计原创帖（默认开启）")
    parser.add_argument("--include_reposts", dest="only_original", action="store_false", help="包含转发/引用")
    parser.add_argument("--exclude_neutral", action="store_true", help="支持率分母排除中立/未判定")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-mnli", help="Hugging Face 模型名或路径（MNLI）")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批大小")
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 cpu、cuda:0；默认自动")
    parser.add_argument("--no_informer", action="store_true", help="关闭告知者检测")
    parser.add_argument("--target", type=str, default=None, help="（可选）立场针对的主题/命题文本")
    parser.add_argument("--threshold", type=float, default=0.5, help="立场判定置信度阈值")
    parser.add_argument("--informer_threshold", type=float, default=0.6, help="告知者判定置信度阈值")
    parser.add_argument("--total_steps", type=int, default=None, help="固定总时间步数 T。若提供，则为每个用户对齐 1..T 并对缺失步前向填充上一条非空 content")

    args = parser.parse_args()

    cfg = DetectorConfig(
        only_original=args.only_original,
        exclude_neutral=args.exclude_neutral,
        model_name_or_path=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        enable_informer=not args.no_informer,
        target_text=args.target,
        confidence_threshold=args.threshold,
        informer_threshold=args.informer_threshold,
    )
    detector = BertZeroShotStanceDetector(cfg)

    # 1) 读取 & 时间步
    posts = read_posts(args.db_path, only_original=cfg.only_original)
    posts = assign_timesteps(posts)

    # 1.1) 若提供 total_steps，则按用户对齐 1..T 并进行前向填充
    if args.total_steps is not None:
        posts = fill_missing_timesteps_with_ffill(posts, args.total_steps)

    # 2) 聚合
    agg_df, _ = compute_support_by_timestep(posts, detector)

    # 2.1) 覆盖 claim_step：从 trace 表读取 user_id=0 的 send_to_group 的 created_at
    claim_step = read_claim_step_from_trace(args.db_path)
    if claim_step != -1:
        agg_df = agg_df.sort_values("timestep").reset_index(drop=True)
        # 新增列 claim_step_fill: 在 claim_step 之前为 0，之后为 claim_step
        def _fill_val(t: int) -> int:
            return 0 if (claim_step == -1 or t < claim_step) else int(claim_step)
        agg_df["claim_step"] = agg_df["timestep"].astype(int).map(_fill_val)

    # 3) 导出
    out_csv = export_csv(agg_df, args.output_csv)

    print(f"✓ 导出 CSV: {out_csv}")
    if claim_step == -1:
        print("- 告知者步骤: 未出现（claim_step = -1）")
    else:
        print(f"- 告知者步骤: {claim_step}")


if __name__ == "__main__":
    main()
