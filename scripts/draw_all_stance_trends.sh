#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.."

# 遍历experiments目录下的所有文件夹
for exp_dir in experiments/*/; do
    # 检查目录是否存在
    if [ ! -d "$exp_dir" ]; then
        continue
    fi
    
    # 检查是否存在attitude_analysis_results.csv文件
    if [ -f "${exp_dir}attitude_analysis_results.csv" ]; then
        echo "处理实验数据: $exp_dir"
        
        # 运行绘图脚本
        python DBinfo/db_stance_draw.py \
            --input_file "${exp_dir}attitude_analysis_results.csv" \
            --output_dir "$exp_dir" \
            --dpi 300
            
        echo "完成绘图: ${exp_dir}stance_trend.png"
        echo "----------------------------------------"
    fi
done

echo "所有实验数据的趋势图已生成完成！" 