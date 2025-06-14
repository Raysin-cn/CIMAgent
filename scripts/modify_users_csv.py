#!/usr/bin/env python3
"""
修改用户CSV文件，将0号智能体设置为匿名智能体
"""

import pandas as pd
import sys
import os

def modify_users_csv(input_file, output_file):
    """修改用户CSV文件，将0号智能体设置为匿名智能体"""
    
    # 读取原始CSV文件
    df = pd.read_csv(input_file)
    
    print(f"原始文件包含 {len(df)} 个用户")
    
    # 创建匿名智能体（0号智能体）
    anonymous_agent = {
        'Unnamed: 0': 0,
        'user_id': 0,  # 匿名智能体使用0作为user_id
        'name': 'anonymous',
        'username': 'anonymous',
        'description': '匿名智能体 - 用于发布匿名帖子',
        'created_at': '2009-01-01 00:00:00+00:00',
        'followers_count': 0,
        'following_count': 0,
        'following_list': '[]',
        'following_agentid_list': '[]',
        'previous_tweets': '[]',
        'tweets_id': '[]',
        'activity_level_frequency': '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]',
        'activity_level': 'off_line',
        'user_char': 'anonymous',
        'followers_list': '[]'
    }
    
    # 将其他智能体的ID依次后移
    # 原来的0号智能体变成1号，1号变成2号，以此类推
    df['Unnamed: 0'] = df['Unnamed: 0'] + 1
    df['user_id'] = df['user_id']  # 保持原始user_id不变，因为这是真实的Twitter ID
    
    # 更新用户名和显示名称
    for i in range(len(df)):
        old_user_num = df.iloc[i]['Unnamed: 0'] - 1  # 原来的编号
        new_user_num = df.iloc[i]['Unnamed: 0']      # 新的编号
        df.iloc[i, df.columns.get_loc('name')] = f'user_{new_user_num}'
        df.iloc[i, df.columns.get_loc('username')] = f'user{new_user_num}'
    
    # 在开头插入匿名智能体
    anonymous_df = pd.DataFrame([anonymous_agent])
    result_df = pd.concat([anonymous_df, df], ignore_index=True)
    
    # 保存修改后的文件
    result_df.to_csv(output_file, index=False)
    
    print(f"✓ 已创建修改后的文件: {output_file}")
    print(f"✓ 匿名智能体已设置为0号智能体 (user_id: -1)")
    print(f"✓ 其他智能体ID已依次后移")
    print(f"✓ 总共包含 {len(result_df)} 个用户")
    
    # 显示前几行作为验证
    print("\n前5行数据:")
    print(result_df[['Unnamed: 0', 'user_id', 'name', 'username']].head())
    
    return result_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python modify_users_csv.py <输入文件> <输出文件>")
        print("示例: python modify_users_csv.py data/users_info_10.csv data/users_info_10_modified.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)
    
    try:
        modify_users_csv(input_file, output_file)
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        sys.exit(1) 