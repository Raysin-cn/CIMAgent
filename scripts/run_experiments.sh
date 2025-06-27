#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p data/output

# 主题、用户、帖子等参数
USERS_CSV="data/raw/users_info.csv"
TOPIC="Should AI Art Be Allowed in Traditional Art Competitions?"
POSTS_JSON="data/processed/generated_posts.json"
PROFILE_OUTPUT="data/processed/oasis_user_profiles.csv"
STEPS=72

# 实验1：不加--intervene_dialogue
python main.py \
    --users_csv "$USERS_CSV" \
    --topic "$TOPIC" \
    --posts_json "$POSTS_JSON" \
    --profile_output "$PROFILE_OUTPUT" \
    --db_path "data/output/exp_no_intervene.db" \
    --steps $STEPS \
    --cleanup

# 实验2：加--intervene_dialogue
python main.py \
    --users_csv "$USERS_CSV" \
    --topic "$TOPIC" \
    --posts_json "$POSTS_JSON" \
    --profile_output "$PROFILE_OUTPUT" \
    --db_path "data/output/exp_with_intervene.db" \
    --steps $STEPS \
    --cleanup \
    --intervene_dialogue

# 你可以继续添加更多实验组合