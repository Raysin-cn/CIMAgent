#!/usr/bin/env python3
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import os
import pandas as pd
import shutil
import json
from datetime import datetime
import argparse
import torch
from dotenv import load_dotenv
load_dotenv()

from oasis import ActionType, LLMAction, ManualAction, generate_twitter_agent_graph
from cim.generate_public_posts import PostGenerator, GeneratedPost
from cim.data_preparation import OasisPostInjector




async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Oasis社交网络模拟与帖子注入")
    parser.add_argument("--users_csv", default="data/users_info.csv", 
                       help="用户数据CSV文件路径（用于创建代理图）")
    parser.add_argument("--posts_json", default="data/generated_posts.json", 
                       help="生成的帖子JSON文件路径（将被作为匿名帖子注入）")
    parser.add_argument("--profile_output", default="data/oasis_user_profiles.csv", 
                       help="Oasis用户档案输出路径")
    parser.add_argument("--db_path", default="./data/twitter_simulation.db", 
                       help="模拟数据库路径")
    parser.add_argument("--steps", type=int, default=20, 
                       help="模拟步数（代理互动步数）")
    
    args = parser.parse_args()
    
    print("CIMAgent Oasis社交网络模拟 - 匿名帖子注入")
    print("=" * 60)
    print("注意：所有生成的帖子将作为匿名帖子注入到系统中")
    print("匿名帖子的发布者不会参与后续的社交网络演进")
    print("=" * 60)
    
    # 初始化注入器
    injector = OasisPostInjector(db_path=args.db_path)
    
    try:
        # 加载数据
        print("1. 加载数据...")
        injector.load_users_data(args.users_csv)
        injector.load_generated_posts(args.posts_json)
        
        # 创建用户档案（用于创建代理图）
        print("2. 创建用户档案...")
        profile_path = injector.create_user_profile_csv(args.profile_output)
        
        # 运行模拟（包含匿名帖子注入）
        print("3. 运行模拟并注入匿名帖子...")
        env = await injector.run_simulation_with_posts(
            profile_path=profile_path,
            posts=injector.generated_posts,  # 所有帖子将作为匿名帖子注入
            num_steps=args.steps
        )
        
        print("\n" + "=" * 60)
        print("模拟完成！")
        print("=" * 60)
        print("模拟结果:")
        print(f"- 数据库文件: {args.db_path}")
        print(f"- 用户档案: {args.profile_output}")
        print(f"- 注入匿名帖子数: {len(injector.generated_posts)}")
        print(f"- 代理互动步数: {args.steps}")
        print("\n匿名帖子说明:")
        print("- 所有帖子都以匿名用户身份发布（user_id = 0）")
        print("- 匿名用户不会参与后续的社交网络互动")
        print("- 匿名帖子会出现在推荐系统中，供其他代理查看和互动")
        print("- 可以通过数据库查询验证匿名帖子的存在")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


