#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 创建输出目录
mkdir -p data/stance

# 公共参数
TOPIC="Should AI Art Be Allowed in Traditional Art Competitions?"

# 对照组分析
python eval.py \
    --db_path data/simu/exp_no_intervene.db \
    --output data/stance/exp_no_intervene.json \
    --csv data/stance/exp_no_intervene.csv \
    --topic "$TOPIC"

# 实验组分析
for STRATEGY in topk_degree random topk_influence; do
    for NUM_TARGETS in 5 10 20; do
        DB_PATH="data/simu/exp_intervene_${STRATEGY}_${NUM_TARGETS}.db"
        OUTPUT_JSON="data/stance/exp_intervene_${STRATEGY}_${NUM_TARGETS}.json"
        OUTPUT_CSV="data/stance/exp_intervene_${STRATEGY}_${NUM_TARGETS}.csv"
        python eval.py \
            --db_path "$DB_PATH" \
            --output "$OUTPUT_JSON" \
            --csv "$OUTPUT_CSV" \
            --topic "$TOPIC"
    done
done

# 你可以继续添加更多实验组合