#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 运行主程序
# 注意:在main.py中use_hidden_control参数定义为type=bool,这会导致任何非空字符串都被解析为True
# 需要修改为type=str,然后在代码中手动转换为bool,或使用action='store_true'
python main.py \
    --model_path /data/model/Qwen3-14B \
    --model_url "http://localhost:12345/v1" \
    --topic_file "data/CIM_experiments/posts/posts_topic_3.csv" \
    --users_file "data/CIM_experiments/users_info.csv"\
    --use_hidden_control "True" \
    --seed_rate 0.1 \
    --seed_algo "Random" \