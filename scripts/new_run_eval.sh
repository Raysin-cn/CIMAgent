#!/bin/bash

# CIMAgent 完整评估流程脚本
# 依次执行：帖子生成 -> 模拟运行 -> 立场检测 -> 可视化分析

echo "🚀 CIMAgent 完整评估流程开始"
echo "=================================="

# 设置工作目录
cd "$(dirname "$0")/.."

# 创建必要的目录
mkdir -p data
mkdir -p logs

# 设置日志文件
LOG_FILE="logs/eval_$(date +%Y%m%d_%H%M%S).log"

# 记录开始时间
START_TIME=$(date)
echo "开始时间: $START_TIME" | tee -a "$LOG_FILE"

# 步骤1: 生成帖子
echo "📝 步骤1: 生成帖子..." | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"

python cim/generate_public_posts.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 帖子生成完成" | tee -a "$LOG_FILE"
else
    echo "❌ 帖子生成失败" | tee -a "$LOG_FILE"
    exit 1
fi

echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 步骤2: 运行模拟并注入匿名帖子
echo "🔄 步骤2: 运行模拟并注入匿名帖子..." | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"

python main.py \
    --users_csv "data/users_info_10.csv" \
    --posts_json "data/generated_posts.json" \
    --profile_output "data/oasis_user_profiles.csv" \
    --db_path "./data/twitter_simulation.db" \
    --steps 20 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 模拟运行完成" | tee -a "$LOG_FILE"
else
    echo "❌ 模拟运行失败" | tee -a "$LOG_FILE"
    exit 1
fi

echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 步骤3: 立场检测和演化分析
echo "🔍 步骤3: 立场检测和演化分析..." | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"

python cim/stance_detection.py \
    --db_path "./data/twitter_simulation.db" \
    --output "./data/stance_detection_results.json" \
    --topic "中美贸易关税" \
    --post_limit 3 \
    --evolution 1 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 立场检测完成" | tee -a "$LOG_FILE"
else
    echo "❌ 立场检测失败" | tee -a "$LOG_FILE"
    exit 1
fi

echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 步骤4: 可视化分析
echo "📊 步骤4: 可视化分析..." | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"

python cim/visulization_stance_evol.py \
    --csv_path "./data/stance_detection_results_evolution.csv" 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 可视化分析完成" | tee -a "$LOG_FILE"
else
    echo "❌ 可视化分析失败" | tee -a "$LOG_FILE"
    exit 1
fi

echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 记录结束时间
END_TIME=$(date)
echo "结束时间: $END_TIME" | tee -a "$LOG_FILE"

# 生成结果摘要
echo "📋 评估结果摘要:" | tee -a "$LOG_FILE"
echo "==================================" | tee -a "$LOG_FILE"

# 检查生成的文件
echo "生成的文件:" | tee -a "$LOG_FILE"
if [ -f "data/generated_posts.json" ]; then
    POST_COUNT=$(jq length data/generated_posts.json 2>/dev/null || echo "未知")
    echo "  - 生成的帖子: data/generated_posts.json ($POST_COUNT 条)" | tee -a "$LOG_FILE"
fi

if [ -f "data/twitter_simulation.db" ]; then
    echo "  - 模拟数据库: data/twitter_simulation.db" | tee -a "$LOG_FILE"
fi

if [ -f "data/stance_detection_results_evolution.json" ]; then
    echo "  - 立场演化结果: data/stance_detection_results_evolution.json" | tee -a "$LOG_FILE"
fi

if [ -f "data/stance_detection_results_evolution.csv" ]; then
    echo "  - 立场演化数据: data/stance_detection_results_evolution.csv" | tee -a "$LOG_FILE"
fi

# 检查生成的图片
echo "生成的图表:" | tee -a "$LOG_FILE"
for img in data/*.png; do
    if [ -f "$img" ]; then
        echo "  - $img" | tee -a "$LOG_FILE"
    fi
done

if [ -f "data/stance_evolution_summary.txt" ]; then
    echo "  - 分析报告: data/stance_evolution_summary.txt" | tee -a "$LOG_FILE"
fi

echo "==================================" | tee -a "$LOG_FILE"
echo "🎉 CIMAgent 完整评估流程完成！" | tee -a "$LOG_FILE"
echo "📄 详细日志请查看: $LOG_FILE" | tee -a "$LOG_FILE"

# 显示日志文件位置
echo ""
echo "📄 日志文件位置: $LOG_FILE"
echo "📁 结果文件位置: data/"
echo ""
echo "🔍 快速查看结果:"
echo "  - 查看立场演化摘要: cat data/stance_evolution_summary.txt"
echo "  - 查看生成的图表: ls data/*.png"
echo "  - 查看详细日志: tail -f $LOG_FILE"