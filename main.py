#!/usr/bin/env python3
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import os
import pandas as pd
import shutil
from datetime import datetime
import argparse

from camel.models import ModelFactory
from camel.types import ModelPlatformType

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis import ActionType, EnvAction, SingleAction
import oasis_cim as oasis

async def backup_database(db_path, step, backup_dir):
    """备份数据库文件"""
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    backup_path = os.path.join(backup_dir, f"twitter_simulation_{step}.db")
    shutil.copy2(db_path, backup_path)
    print(f"数据库已备份到: {backup_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Twitter社交网络模拟程序')
    
    # 数据相关参数
    parser.add_argument('--db_path', type=str, default="./data/CIM_experiments/twitter_simulation.db",
                      help='数据库文件保存路径')
    parser.add_argument('--topic_file', type=str, default="data/CIM_experiments/posts/posts_topic_3.csv",
                      help='话题数据文件路径')
    parser.add_argument('--users_file', type=str, default="data/CIM_experiments/users_info.csv",
                      help='用户数据文件路径')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, default="/data/model/Qwen3-14B",
                      help='模型路径')
    parser.add_argument('--model_url', type=str, default="http://localhost:21474/v1",
                      help='模型服务URL')
    parser.add_argument('--max_tokens_1', type=int, default=12000,
                      help='第一个模型的最大token数')
    parser.add_argument('--max_tokens_2', type=int, default=1000,
                      help='第二个模型的最大token数')
    
    # 模拟相关参数
    parser.add_argument('--total_steps', type=int, default=72,
                      help='总模拟步数')
    parser.add_argument('--backup_interval', type=int, default=1,
                      help='数据库备份间隔步数')
    parser.add_argument('--seed_rate', type=float, default=0.1,
                      help='种子用户比例')
    parser.add_argument('--seed_algo', type=str, default="Random",
                      help='种子用户选择算法')
    parser.add_argument('--use_hidden_control', type=bool, default=True,
                      help='是否使用隐藏控制')
    
    # 时间戳和备份目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--backup_dir', type=str, 
                      default=f"./experiments/{timestamp}/backups",
                      help='数据保存备份目录')
    
    return parser.parse_args()

async def main():
    args = parse_args()
    
    # 配置模型
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=args.model_path,
        url=args.model_url,
        model_config_dict={"max_tokens": args.max_tokens_1}
    )
    vllm_model_2 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=args.model_path,
        url=args.model_url,
        model_config_dict={"max_tokens": args.max_tokens_2}
    )
    models = [vllm_model_1, vllm_model_2]

    # 定义可用动作
    available_actions = [
        ActionType.REFRESH, ActionType.SEARCH_USER, ActionType.SEARCH_POSTS,
        ActionType.CREATE_POST, ActionType.LIKE_POST, ActionType.UNLIKE_POST,
        ActionType.DISLIKE_POST, ActionType.UNDO_DISLIKE_POST,
        ActionType.CREATE_COMMENT, ActionType.LIKE_COMMENT, ActionType.UNLIKE_COMMENT,
        ActionType.DISLIKE_COMMENT, ActionType.UNDO_DISLIKE_COMMENT,
        ActionType.FOLLOW, ActionType.UNFOLLOW, ActionType.MUTE, ActionType.UNMUTE,
        ActionType.TREND, ActionType.REPOST, ActionType.QUOTE_POST,
        ActionType.DO_NOTHING,
    ]

    # 删除旧数据库
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    # 写入实验参数配置
    config_dir = os.path.dirname(args.backup_dir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
    config_path = os.path.join(config_dir, "experiment_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Model URL: {args.model_url}\n") 
        f.write(f"Database Path: {args.db_path}\n")
        f.write(f"Topic File: {args.topic_file}\n")
        f.write(f"Users File: {args.users_file}\n")
        f.write(f"Seed Selection Algorithm: {args.seed_algo}\n")
        f.write(f"Seed Rate: {args.seed_rate}\n")
        f.write(f"Use Hidden Control: {args.use_hidden_control}\n")
        f.write(f"Total Steps: {args.total_steps}\n")
        f.write(f"Backup Interval: {args.backup_interval}\n")
        f.write(f"Backup Directory: {args.backup_dir}\n")
        

    # 读取用户配置
    users_df = pd.read_csv(args.users_file)
    user_to_agent = {user_id: idx for idx, user_id in enumerate(users_df['user_id'].values)}
    
    # 创建环境
    env = oasis.make(
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=args.db_path,
        agent_profile_path=args.users_file,
        agent_models=models,
        available_actions=available_actions,
        time_engine="activity_level_frequency",
    )

    await env.reset()

    # 读取话题数据
    topics_df = pd.read_csv(args.topic_file)
    
    # 创建初始帖子
    initial_actions = []
    for idx, row in topics_df.iterrows():
        root_user_id = int(row['user_id'])
        agent_id = user_to_agent.get(root_user_id, idx % len(users_df))
        
        action = SingleAction(
            agent_id=agent_id,
            action=ActionType.CREATE_POST,
            args={"content": row["content"]}
        )
        initial_actions.append(action)

    # 执行初始动作
    env_actions = EnvAction(
        activate_agents=list(range(len(users_df))),
        intervention=initial_actions
    )
    await env.step(env_actions)
    

    await backup_database(args.db_path, 0, args.backup_dir)

    seeds_list_history = []
    seeds = await env.select_seeds(algos=args.seed_algo, seed_nums_rate=args.seed_rate)
    if args.use_hidden_control:
        await env.hidden_control(seeds)
    seeds_list_history.append(seeds)

    # 主模拟循环
    for step in range(args.total_steps):
        empty_action = EnvAction()
        await env.step(empty_action)
        
        if (step + 1) % args.backup_interval == 0:
            await backup_database(args.db_path, step + 1, args.backup_dir)
            seeds = await env.select_seeds(algos=args.seed_algo, seed_nums_rate=args.seed_rate)
            if args.use_hidden_control:
                await env.hidden_control(seeds)
            seeds_list_history.append(seeds)

    # 最终备份
    await backup_database(args.db_path, step="Done", backup_dir=args.backup_dir)
    await env.close()

if __name__ == "__main__":
    asyncio.run(main()) 