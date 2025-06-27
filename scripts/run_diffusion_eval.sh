#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 创建日志目录
mkdir -p logs
mkdir -p IMmodule/results

# 设置实验参数
# SEED_RATIOS=(0.05 0.1 0.15 0.2 0.3)
# DIFFUSION_MODELS=("LT" "IC" "SIS")
SEED_RATIOS=(0.05)
DIFFUSION_MODELS=("LT" "IC")

# 记录开始时间
echo "实验开始时间: $(date)" > logs/experiment.log

# 遍历所有参数组合
for ratio in "${SEED_RATIOS[@]}"; do
    for model in "${DIFFUSION_MODELS[@]}"; do
        echo "开始实验: 种子比例=${ratio}, 传播模型=${model}"
        echo "开始时间: $(date)" >> logs/experiment.log
        
        # 运行实验
        python IMmodule/eval_diffusion.py \
            --seed_ratio ${ratio} \
            --diffusion ${model} \
            --n_nodes 1000 \
            --num_pairs 100 \
            --epochs 200 \
            --save_dir "models/deepim_${model}_${ratio}" \
            2>&1 | tee -a "logs/${model}_${ratio}.log"
        
        # 检查实验是否成功
        if [ $? -eq 0 ]; then
            echo "实验成功完成: 种子比例=${ratio}, 传播模型=${model}"
            echo "结束时间: $(date)" >> logs/experiment.log
        else
            echo "实验失败: 种子比例=${ratio}, 传播模型=${model}"
            echo "失败时间: $(date)" >> logs/experiment.log
        fi
        
        echo "----------------------------------------" >> logs/experiment.log
    done
done

# 记录结束时间
echo "实验结束时间: $(date)" >> logs/experiment.log

# 汇总结果
echo "实验汇总报告:" > IMmodule/results/summary.txt
echo "生成时间: $(date)" >> IMmodule/results/summary.txt
echo "----------------------------------------" >> IMmodule/results/summary.txt

for ratio in "${SEED_RATIOS[@]}"; do
    for model in "${DIFFUSION_MODELS[@]}"; do
        if [ -f "IMmodule/results/results_${model}_${ratio}.txt" ]; then
            echo "参数组合: 种子比例=${ratio}, 传播模型=${model}" >> IMmodule/results/summary.txt
            cat "IMmodule/results/results_${model}_${ratio}.txt" >> IMmodule/results/summary.txt
            echo "----------------------------------------" >> IMmodule/results/summary.txt
        fi
    done
done

