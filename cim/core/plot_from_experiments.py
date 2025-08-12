#!/usr/bin/env python3
"""
从 ./experiments 下的实验 CSV 聚合并绘制两张图：
1) 不同 claim_step 的扩散曲线（子图网格），每个子图内按 seed_ratio 画均值曲线
2) 最后一步支持率的分组柱状图（x: claim_step，组: seed_ratio）

- CSV 命名解析参考 run.sh 与 main.py：匹配 r{ratio}_g{goc}_c{claim}
- CSV 内部列期望包含: timestep, support_ratio（可有 claim_step 列但以文件名为准）
- c0 视为无告知（在图中标为 No claim / -1）
- 将每个 CSV 作为一个 replicate 参与均值

用法示例：
  python cim/core/plot_from_experiments.py \
    --input_dir /home/lsj/Projects/CIMagent/experiments \
    --outdir /home/lsj/Projects/CIMagent/figs
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RATIO_KEYS = [0.05, 0.10, 0.15, 0.20]
CLAIM_KEYS = [-1, 3, 6, 9]  # -1 代表 No claim

# Okabe–Ito colorblind-safe palette for clear scientific plots
COLORS_LINE = {0.05: "#0072B2", 0.10: "#E69F00", 0.15: "#009E73", 0.20: "#D55E00"}
COLORS_BAR  = {0.05: "#A6C8E6", 0.10: "#F3C77A", 0.15: "#7BD1B0", 0.20: "#F29B7B"}
TITLES = {-1: "No claim", 3: "Claim at step 3", 6: "Claim at step 6", 9: "Claim at step 9"}
MARKERS = {0.05: "o", 0.10: "s", 0.15: "^", 0.20: "D"}

RE_RGC = re.compile(r"r(?P<ratio>\d+\.\d+)_g(?P<goc>\d+)_c(?P<claim>\d+)")
RE_STEPS_STYLE = re.compile(
    r"steps(?P<steps>\d+)_goc(?P<goc>\d+)_claim(?P<claim>-?\d+)_(?P<kflag>[rk])(?P<kval>\d+\.\d+|\d+)")


@dataclass
class ParsedName:
    ratio: Optional[float]
    goc: Optional[int]
    claim: Optional[int]


def parse_from_name(name: str) -> ParsedName:
    """尽量从文件名解析 ratio/goc/claim。优先 r.._g.._c.. 模式；退化支持 steps.._goc.._claim.._r/k.. 模式。
    返回 ratio=None 表示无法解析，届时跳过该文件。
    """
    m = RE_RGC.search(name)
    if m:
        ratio = float(m.group("ratio"))
        goc = int(m.group("goc"))
        claim = int(m.group("claim"))
        return ParsedName(ratio=ratio, goc=goc, claim=claim)

    m2 = RE_STEPS_STYLE.search(name)
    if m2:
        goc = int(m2.group("goc"))
        claim = int(m2.group("claim"))
        kflag = m2.group("kflag")
        kval = m2.group("kval")
        ratio: Optional[float] = None
        if kflag == "r":
            try:
                ratio = float(kval)
            except ValueError:
                ratio = None
        # kflag == 'k'（按数量）缺少比例信息，跳过
        return ParsedName(ratio=ratio, goc=goc, claim=claim)

    return ParsedName(ratio=None, goc=None, claim=None)
def load_experiment_csvs(input_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    files = sorted([p for p in input_dir.glob("*.csv")])
    for rep_id, p in enumerate(files):
        parsed = parse_from_name(p.name)
        if parsed.ratio is None:
            continue
        ratio = float(parsed.ratio)
        goc = int(parsed.goc) if parsed.goc is not None else 1
        claim = int(parsed.claim) if parsed.claim is not None else 0
        # goc=0 或 claim=0 统一视为无告知 -1
        claim_level = -1 if (goc == 0 or claim == 0) else claim

        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"timestep", "support_ratio"}.issubset(df.columns):
            # 尝试兼容字段名
            if "support" in df.columns:
                df = df.rename(columns={"support": "support_ratio"})
            else:
                continue
        for _, r in df.iterrows():
            try:
                t = int(r["timestep"])  # 可能是浮点字符串，统一转 int
                s = float(r["support_ratio"]) if pd.notna(r["support_ratio"]) else np.nan
            except Exception:
                continue
            if pd.isna(s):
                continue
            rows.append({
                "timestep": t,
                "seed_ratio": ratio,
                "claim_step": claim_level,
                "replicate": rep_id,
                "support_ratio": s,
                "_source": str(p)
            })
    if not rows:
        return pd.DataFrame(columns=["timestep", "seed_ratio", "claim_step", "replicate", "support_ratio", "_source"])
    return pd.DataFrame(rows)
def plot_diffusion(df: pd.DataFrame, out_path: Path) -> Path:
    # 与 generate_and_plot.py 一致：2x2 子图，-1,3,6,9
    claim_levels = CLAIM_KEYS
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, claim in zip(axes, claim_levels):
        sub = df[df["claim_step"] == claim]
        for seed_ratio in RATIO_KEYS:
            d = sub[sub["seed_ratio"] == seed_ratio]
            if d.empty:
                continue
            mean_curve = d.groupby("timestep")["support_ratio"].mean()
            ax.plot(mean_curve.index, mean_curve.values,
                    label=f"seed={int(seed_ratio*100)}%",
                    color=COLORS_LINE.get(seed_ratio, "#555555"), linewidth=2.5,
                    marker=MARKERS.get(seed_ratio, "o"), markersize=4, alpha=0.95)
        ax.set_title(TITLES[claim])
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Support ratio")
        ax.set_xlabel("Timestep")
        ax.legend(frameon=False)

    fig.suptitle("Diffusion speed by seed ratio and claim timing (from experiments)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
def plot_final_bars(df: pd.DataFrame, out_path: Path) -> Path:
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 创建空白图避免崩溃
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    last_t = df["timestep"].max()
    agg = (
        df[df["timestep"] == last_t]
        .groupby(["claim_step", "seed_ratio"])['support_ratio']
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = ["No claim", "3", "6", "9"]
    x_keys = [-1, 3, 6, 9]
    x = np.arange(len(x_labels))
    width = 0.22

    for i, seed_ratio in enumerate(RATIO_KEYS):
        vals = []
        for key in x_keys:
            row = agg[(agg["claim_step"] == key) & (agg["seed_ratio"] == seed_ratio)]
            vals.append(row["support_ratio"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width,
               label=f"seed={int(seed_ratio*100)}%",
               color=COLORS_BAR.get(seed_ratio, "#888888"), edgecolor="#333333", linewidth=0.6, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Final support ratio (last step)")
    ax.set_title("Earlier claim and higher seeds lead to higher final support (from experiments)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, axis='y', linestyle=":", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
def main():
    parser = argparse.ArgumentParser(description="Aggregate ./experiments CSVs and plot figures like generate_and_plot.py")
    parser.add_argument("--input_dir", default="experiments", help="输入CSV目录")
    parser.add_argument("--outdir", default="figs", help="输出图片目录")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_experiment_csvs(input_dir)
    if df.empty:
        print(f"[WARN] No valid CSV found in {input_dir}")
    else:
        # 仅保留我们关注的 ratio 集合，避免奇异值影响配色和图例
        df = df[df["seed_ratio"].isin(RATIO_KEYS)]

    fig1 = outdir / "diffusion_by_claim_and_seed.png"
    fig2 = outdir / "final_support_bars.png"
    plot_diffusion(df, fig1)
    plot_final_bars(df, fig2)

    print(f"✓ Saved figure: {fig1}")
    print(f"✓ Saved figure: {fig2}")


if __name__ == "__main__":
    main()
