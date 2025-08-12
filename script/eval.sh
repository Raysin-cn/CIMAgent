#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DB_DIR="${DB_DIR:-$ROOT_DIR/experiments}"

# 可配置参数（可用环境变量覆盖）
MODEL_NAME="${MODEL_NAME:-facebook/bart-large-mnli}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-}"               # 例如: cpu 或 cuda:0；为空表示自动
TOTAL_STEPS="${TOTAL_STEPS:-12}"   # 与仿真步数保持一致，必要时覆盖
EXCLUDE_NEUTRAL="${EXCLUDE_NEUTRAL:-}"   # 设置为非空即传 --exclude_neutral
NO_INFORMER="${NO_INFORMER:-}"           # 设置为非空即传 --no_informer
TARGET_TEXT="${TARGET_TEXT:-"Should We Support the Purchase of Xinjiang Cotton Products?"}"           # 目标命题文本，非空即传 --target
THRESHOLD="${THRESHOLD:-0.5}"
INFORMER_THRESHOLD="${INFORMER_THRESHOLD:-0.6}"

if [[ ! -d "$DB_DIR" ]]; then
  echo "数据库目录不存在: $DB_DIR" >&2
  exit 1
fi

# 递归查找所有 .db 文件
mapfile -t db_files < <(find "$DB_DIR" -type f -name "*.db" | sort)
if [[ ${#db_files[@]} -eq 0 ]]; then
  echo "未在 $DB_DIR 下找到任何 .db 文件"
  exit 0
fi

processed=0
skipped=0
failed=0

for db in "${db_files[@]}"; do
  csv="${db%.db}.csv"

  if [[ -f "$csv" ]]; then
    echo "跳过（已存在 CSV）: $csv"
    ((skipped++)) || true
    continue
  fi

  echo "分析: $db -> $csv"

  ARGS=(
    --db_path "$db"
    --output_csv "$csv"
    --model_name "$MODEL_NAME"
    --batch_size "$BATCH_SIZE"
    --threshold "$THRESHOLD"
    --informer_threshold "$INFORMER_THRESHOLD"
    --total_steps "$TOTAL_STEPS"
  )

  if [[ -n "$DEVICE" ]]; then ARGS+=( --device "$DEVICE" ); fi
  if [[ -n "$TARGET_TEXT" ]]; then ARGS+=( --target "$TARGET_TEXT" ); fi
  if [[ -n "$EXCLUDE_NEUTRAL" ]]; then ARGS+=( --exclude_neutral ); fi
  if [[ -n "$NO_INFORMER" ]]; then ARGS+=( --no_informer ); fi

  set +e
  python3 "$ROOT_DIR/cim/core/stance_detector.py" "${ARGS[@]}"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "失败: $db" >&2
    ((failed++)) || true
    continue
  fi

  if [[ -f "$csv" ]]; then
    echo "✓ 生成: $csv"
    ((processed++)) || true
  else
    echo "失败（未生成 CSV）: $db" >&2
    ((failed++)) || true
  fi
done

echo "完成。生成: $processed, 跳过: $skipped, 失败: $failed"
exit 0

