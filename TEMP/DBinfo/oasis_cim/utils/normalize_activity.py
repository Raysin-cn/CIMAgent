import pandas as pd
import numpy as np
from pathlib import Path

def normalize_activity_frequency(input_file: str, output_file: str = None):
    """
    归一化activity_level_frequency列中的24维向量，结果保留两位小数
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str, optional): 输出CSV文件路径。如果为None，则覆盖原文件
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 确保activity_level_frequency列存在
    if 'activity_level_frequency' not in df.columns:
        raise ValueError("CSV文件中没有activity_level_frequency列")
    
    # 将字符串形式的列表转换为实际的列表
    df['activity_level_frequency'] = df['activity_level_frequency'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )
    
    # 归一化每个24维向量，并保留两位小数
    df['activity_level_frequency'] = df['activity_level_frequency'].apply(
        lambda x: [float(round(num, 2)) for num in list(np.array(x) / np.sum(x))] if isinstance(x, list) else x
    )
    
    # 将列表转换回字符串形式
    df['activity_level_frequency'] = df['activity_level_frequency'].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    
    # 保存结果
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"归一化完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    # 设置输入输出文件路径
    input_file = "./data/twitter_dataset/user_all_id_time.csv"
    output_file = "./data/twitter_dataset/user_all_id_time_normalized.csv"
    
    # 执行归一化
    normalize_activity_frequency(input_file, output_file) 