#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 运行主题3的态度分析
python DBinfo/db_stance_analysis.py \
    --topic_index 3 \
    --db_dir "experiments/topic_3_Qwen3-14B_True_20250519_104621/backups" \
    --output_dir "experiments/topic_3_Qwen3-14B_True_20250519_104621" \
    --start_index 0 \
    --end_index 10

# 运行主题4的态度分析
python DBinfo/db_stance_analysis.py \
    --topic_index 4 \
    --db_dir "experiments/topic_4_Qwen3-14B_True_20250519_113304/backups" \
    --output_dir "experiments/topic_4_Qwen3-14B_True_20250519_113304" \
    --start_index 0 \
    --end_index 10

# # 运行主题5的态度分析
# python DBinfo/db_stance_analysis.py \
#     --topic_index 5 \
#     --db_dir "experiments/topic_5_Qwen3-14B_True_20250519_120926/backups" \
#     --output_dir "experiments/topic_5_Qwen3-14B_True_20250519_120926" \
#     --start_index 0 \
#     --end_index 10
