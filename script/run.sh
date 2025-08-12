#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
MAIN_PY="$ROOT_DIR/main.py"

# 固定参数
STEPS=12
DB_DIR="${DB_DIR:-$ROOT_DIR/experiments}"
mkdir -p "$DB_DIR"

# 比例、goc、claim_step 枚举
RATIOS=(0.05 0.10 0.15 0.20)
GOCS=(0 1)
CLAIMS=(0 3 6 9)

timestamp() { date +"%m%d_%H%M"; }

for ratio in "${RATIOS[@]}"; do
  for goc in "${GOCS[@]}"; do
    if [[ "$goc" -eq 1 ]]; then
      for claim in "${CLAIMS[@]}"; do
        DB_PATH="$DB_DIR/Qwen_sim_$(timestamp)_r${ratio}_g${goc}_c${claim}.db"
        echo "Running: ratio=$ratio goc=$goc claim=$claim -> $DB_PATH"
        ARGS=(
          --im_k_ratio "$ratio"
          --goc "$goc"
          --claim_step "$claim"
          --steps "$STEPS"
          --db_path "$DB_PATH"
        )
        if [[ -n "${USERS_CSV:-}" ]]; then ARGS+=(--users_csv "$USERS_CSV"); fi
        if [[ -n "${POSTS_CSV:-}" ]]; then ARGS+=(--posts_csv "$POSTS_CSV"); fi
        python3 "$MAIN_PY" "${ARGS[@]}"
      done
    else
      # goc=0 时，claim_step 无效，固定为 0 运行一次
      DB_PATH="$DB_DIR/Qwen_sim_$(timestamp)_r${ratio}_g${goc}.db"
      echo "Running: ratio=$ratio goc=$goc -> $DB_PATH"
      ARGS=(
        --im_k_ratio "$ratio"
        --goc "$goc"
        --claim_step 0
        --steps "$STEPS"
        --db_path "$DB_PATH"
      )
      if [[ -n "${USERS_CSV:-}" ]]; then ARGS+=(--users_csv "$USERS_CSV"); fi
      if [[ -n "${POSTS_CSV:-}" ]]; then ARGS+=(--posts_csv "$POSTS_CSV"); fi
      python3 "$MAIN_PY" "${ARGS[@]}"
    fi
  done
done

echo "All experiments finished. DB files saved to $DB_DIR"
