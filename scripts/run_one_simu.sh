#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 运行主程序
python main.py \
    --model_path /data/model/Qwen3-14B \
    --model_url "http://localhost:21474/v1" \
    --topic_file "data/CIM_experiments/posts/posts_topic_3.csv" \
    --users_file "data/CIM_experiments/users_info.csv"\
    --use_hidden_control True
