#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 创建输出目录
mkdir -p data/output

# 公共参数
USERS_CSV="data/raw/users_info.csv"
TOPIC="Should We Still Support Xinjiang Cotton Despite Forced Labor Allegations?"
POSTS_JSON="data/processed/generated_posts.json"
PROFILE_OUTPUT="data/processed/oasis_user_profiles.csv"
STEPS=10

# 不介入匿名智能体（对照组）
python main.py \
    --users_csv "$USERS_CSV" \
    --topic "$TOPIC" \
    --posts_json "$POSTS_JSON" \
    --profile_output "$PROFILE_OUTPUT" \
    --db_path "data/simu/exp_no_intervene.db" \
    --steps $STEPS \
    --cleanup

# 介入匿名智能体（实验组）
for STRATEGY in topk_degree random topk_influence; do
    for NUM_TARGETS in 5 10 20; do
        DB_PATH="data/simu/exp_intervene_${STRATEGY}_${NUM_TARGETS}.db"
        python main.py \
            --users_csv "$USERS_CSV" \
            --topic "$TOPIC" \
            --posts_json "$POSTS_JSON" \
            --profile_output "$PROFILE_OUTPUT" \
            --db_path "$DB_PATH" \
            --steps $STEPS \
            --cleanup \
            --intervene_dialogue \
            --target_strategy "$STRATEGY" \
            --num_targets $NUM_TARGETS \
            --num_turns 3
    done
done

# 你可以继续添加更多实验组合