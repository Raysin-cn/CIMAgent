#!/usr/bin/env python3
"""
Synthetic experiment data generator and plotting utilities.

Design (aligned with main's experiment idea):
- Seed ratio: 5%, 15%, 25% (proxy for number of seed nodes)
- Informer claim step: none, 3, 6, 9
- Expectation:
  * Higher seed ratio → faster diffusion (curves rise faster)
  * Earlier claim step → higher final support at the end

This script simulates multiple runs per (seed_ratio, claim_step) combo and plots:
1) Diffusion curves per claim_step (subplots), 3 lines for different seed ratios
2) Final support bar chart vs claim_step grouped by seed ratio

Usage examples:
  python figure_generation/generate_and_plot.py
  python figure_generation/generate_and_plot.py --steps 15 --replicates 40
Outputs (by default):
  - data/output/figure_generation/sim_results.csv
  - data/output/figure_generation/diffusion_by_claim_and_seed.png
  - data/output/figure_generation/final_support_bars.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend for headless environments (saves files without DISPLAY)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SEED_RATIOS = [0.05, 0.15, 0.25]
CLAIM_STEPS: List[Optional[int]] = [None, 3, 6, 9]


@dataclass
class SimConfig:
    steps: int = 12
    replicates: int = 30
    random_seed: int = 42
    # diffusion parameters
    base_r: float = 0.15            # base logistic growth rate
    r_per_seed: float = 1.2         # growth rate increment per unit seed ratio
    carcap_no_claim: float = 0.65   # carrying capacity without claim
    carcap_with_claim: float = 0.90 # carrying capacity with claim
    post_claim_r_boost: float = 0.08  # additional growth after claim
    noise_std: float = 0.01         # small gaussian noise per step


def simulate_one_curve(
    cfg: SimConfig,
    seed_ratio: float,
    claim_step: Optional[int],
    rng: "np.random.RandomState",
) -> np.ndarray:
    """Simulate supporters proportion over time steps [1..steps].

    Dynamics:
      s_{t+1} = s_t + r_t * s_t * (K_t - s_t) + eps
    where r_t = base_r + r_per_seed * seed_ratio (+ post_claim_r_boost after claim)
          K_t = carcap_with_claim if t >= claim_step else carcap_no_claim
    If claim_step is None → no claim boost and K_t = carcap_no_claim
    """
    s = 0.02 + 0.3 * seed_ratio  # initial support proportional to seed ratio
    series = [s]

    for t in range(1, cfg.steps + 1):
        r_t = cfg.base_r + cfg.r_per_seed * seed_ratio
        if claim_step is not None and t >= claim_step:
            r_t += cfg.post_claim_r_boost
            K_t = cfg.carcap_with_claim
        else:
            K_t = cfg.carcap_no_claim

        # logistic-like step with small noise
        s = s + r_t * s * max(K_t - s, 0.0) + rng.normal(0.0, cfg.noise_std)
        s = float(np.clip(s, 0.0, 1.0))
        series.append(s)

    # return length steps (discard the initial slot so indices 1..steps)
    return np.array(series[1:])


def simulate_grid(cfg: SimConfig) -> pd.DataFrame:
    rows = []
    # Backward-compatible RNG for older NumPy versions
    master_rs = np.random.RandomState(cfg.random_seed)
    # run replicates for each combo
    for claim in CLAIM_STEPS:
        for seed_ratio in SEED_RATIOS:
            for rep in range(cfg.replicates):
                # per-replicate RNG for reproducibility but varied
                local_rs = np.random.RandomState(master_rs.randint(0, 2**31 - 1))
                series = simulate_one_curve(cfg, seed_ratio, claim, local_rs)
                for t, val in enumerate(series, start=1):
                    rows.append({
                        "timestep": t,
                        "seed_ratio": seed_ratio,
                        "claim_step": -1 if claim is None else claim,
                        "replicate": rep,
                        "support_ratio": val,
                    })
    return pd.DataFrame(rows)


def plot_diffusion(df: pd.DataFrame, out_path: Path) -> Path:
    # Grid: 2x2 subplots by claim_step None/-1, 3, 6, 9
    claim_levels = [-1, 3, 6, 9]
    titles = {
        -1: "No claim",
        3: "Claim at step 3",
        6: "Claim at step 6",
        9: "Claim at step 9",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {0.05: "#888888", 0.15: "#666666", 0.25: "#444444"}

    for ax, claim in zip(axes, claim_levels):
        sub = df[df["claim_step"] == claim]
        for seed_ratio in SEED_RATIOS:
            d = sub[sub["seed_ratio"] == seed_ratio]
            # mean over replicates
            mean_curve = d.groupby("timestep")["support_ratio"].mean()
            ax.plot(mean_curve.index, mean_curve.values, label=f"seed={int(seed_ratio*100)}%", color=colors[seed_ratio], linewidth=2)

        ax.set_title(titles[claim])
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Support ratio")
        ax.set_xlabel("Timestep")
        ax.legend(frameon=False)

    fig.suptitle("Diffusion speed by seed ratio and claim timing")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_final_bars(df: pd.DataFrame, out_path: Path) -> Path:
    last_t = df["timestep"].max()
    agg = (
        df[df["timestep"] == last_t]
        .groupby(["claim_step", "seed_ratio"])['support_ratio']
        .mean()
        .reset_index()
    )
    # order for plotting
    agg["claim_step_label"] = agg["claim_step"].map({-1: "No claim", 3: "3", 6: "6", 9: "9"})
    agg = agg.sort_values(["claim_step", "seed_ratio"])  # ensure grouped order

    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = ["No claim", "3", "6", "9"]
    x = np.arange(len(x_labels))
    width = 0.22

    color_map = {0.05: "#bbbbbb", 0.15: "#888888", 0.25: "#555555"}

    for i, seed_ratio in enumerate(SEED_RATIOS):
        vals = []
        for key in [-1, 3, 6, 9]:
            row = agg[(agg["claim_step"] == key) & (agg["seed_ratio"] == seed_ratio)]
            vals.append(row["support_ratio"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"seed={int(seed_ratio*100)}%", color=color_map[seed_ratio])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Final support ratio (last step)")
    ax.set_title("Earlier claim and higher seeds lead to higher final support")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, axis='y', linestyle=":", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic diffusion data and plots")
    parser.add_argument("--steps", type=int, default=12, help="timesteps per simulation")
    parser.add_argument("--replicates", type=int, default=30, help="replicates per combination")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--outdir", default="data/output/figure_generation", help="output directory for CSV and plots")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(steps=args.steps, replicates=args.replicates, random_seed=args.seed)
    df = simulate_grid(cfg)

    csv_path = outdir / "sim_results.csv"
    df.to_csv(csv_path, index=False)

    fig1 = outdir / "diffusion_by_claim_and_seed.png"
    fig2 = outdir / "final_support_bars.png"
    plot_diffusion(df, fig1)
    plot_final_bars(df, fig2)

    print(f"✓ Saved CSV: {csv_path}")
    print(f"✓ Saved figure: {fig1}")
    print(f"✓ Saved figure: {fig2}")


if __name__ == "__main__":
    main()


