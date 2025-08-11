#!/usr/bin/env python3
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""
CIMAgent Oasis社交网络模拟主程序

使用重构后的CIM模块，提供统一的配置管理和功能接口
"""

import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
import os

# 导入CIM模块
from cim import OasisPostInjector, config
from cim.core.influence_max import follow_matrix_get, get_influence_maximization_nodes, compare_influence_algorithms
from cim.config import config as cim_config
from cim.utils import generate_twitter_agent_graph


from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import (
    ActionType, 
    ManualAction, 
    LLMAction
)
from oasis.environment.env import OasisEnv
import oasis

# 配置日志
logging.basicConfig(
    level=getattr(logging, cim_config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CIMAgent Oasis社交网络模拟与帖子注入")
    
    parser.add_argument("--topic_info", 
                        default="Should We Support the Purchase of Xinjiang Cotton Products Accused of Oppressing People?",
                        help="本次模拟讨论的话题，将被注入为所有代理的系统提示主题")   
    # 数据文件参数
    parser.add_argument("--db_path", 
                       default=cim_config.paths.db_path,
                       help="模拟数据库路径")
    parser.add_argument("--users_csv", 
                       default=cim_config.paths.users_file,
                       help="用户数据CSV文件路径（用于创建代理图）")
    parser.add_argument("--posts_csv", 
                       default=cim_config.paths.posts_file,
                       help="生成的帖子CSV文件路径（将被作为匿名帖子注入）")
    
    # 模拟参数
    parser.add_argument("--steps", type=int, default=12, 
                       help="模拟步数（代理互动步数）")
    parser.add_argument("--goc", type=int, choices=[0, 1], default=1,
                       help="是否启用群组告知干预机制：0 关闭，1 启用（默认1）")
    # 告知者信息干预参数
    parser.add_argument("--claim_step", type=int, default=4,
                        help="告知者在哪一步发布关键信息")
    
    # 数据管理参数
    parser.add_argument("--backup", action="store_true", default=True,
                       help="在运行前备份数据库")
    
    # 影响力最大化算法参数
    parser.add_argument("--im_k", type=int, default=5,
                       help="影响力最大化种子节点数量（当未指定 --im_k_ratio 时生效，默认5个）")
    parser.add_argument("--im_k_ratio", type=float, choices=[0.05, 0.10, 0.15, 0.20], default=0.05,
                       help="影响力最大化种子节点比例，可选 0.05/0.10/0.15/0.20，优先于 --im_k")
    parser.add_argument("--im_algorithm", type=str, default="Greedy", choices=["Greedy", "Random"],
                       help="影响力最大化算法：Greedy(贪心算法) 或 Random(随机算法)")
    parser.add_argument("--im_model", type=str, default="IC", choices=["IC", "LT"],
                       help="传播模型类型：IC(Independent Cascade) 或 LT(Linear Threshold)")
    parser.add_argument("--im_p", type=float, default=0.1,
                       help="传播概率（默认0.1）")
    parser.add_argument("--im_eval_sims", type=int, default=1000,
                       help="影响力评估模拟次数（默认1000）")
    # 调试参数
    parser.add_argument("--debug", action="store_true", default=True,
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        cim_config.debug = True
        cim_config.log_level = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 初始化
    goc_flag = 1 if args.goc == 1 else 0
    # 文件名加入比例或数量标记
    k_tag = f"r{args.im_k_ratio}" if args.im_k_ratio is not None else f"k{args.im_k}"
    filename = (
        f"sim_"
        f"steps{args.steps}_goc{goc_flag}_claim{args.claim_step}_"
        f"{k_tag}_{args.im_algorithm}.db"
    )
    print("CIMAgent Oasis社交网络模拟 - 匿名帖子注入")
    print("=" * 60)
    print("注意：所有生成的帖子将作为匿名帖子注入到系统中")
    print("匿名帖子的发布者不会参与后续的社交网络演进")
    print("=" * 60)
    logger.info("初始化帖子注入器...")
    injector = OasisPostInjector(db_path=args.db_path)
    logger.info("加载数据...")
    users_data = injector.load_users_data(args.users_csv)
    posts_data = injector.load_generated_posts_csv(args.posts_csv)
    print(f"✓ 加载了 {len(users_data)} 个用户数据")
    print(f"✓ 加载了 {len(posts_data)} 条生成的帖子")
    logger.info("验证数据文件...")
    validation_results = injector.validate_data()
    print(f"✓ 数据已验证：{validation_results}")
    
    # 6. 运行模拟（包含匿名帖子注入）
    model_config = cim_config.model
    logger.info("运行模拟")
    model = ModelFactory.create(
        model_platform=model_config.platform,
        model_type=model_config.model_type,
        url=model_config.url
    )
    # 定义可用动作
    available_actions = ActionType.get_default_twitter_actions()
    # 生成代理图
    logger.info("生成代理图...")
    agent_graph = await generate_twitter_agent_graph(
        profile_path=args.users_csv,
        model=model,
        available_actions=available_actions,
        topic=args.topic_info,
    )
    # 删除旧数据库
    if os.path.exists(args.db_path):
        os.remove(args.db_path)
        logger.info("删除旧数据库文件")
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=args.db_path,
    )
        
    # 重置环境
    await env.reset()
    logger.info("环境重置完成")

    group_agents = []
    if args.goc == 1:
        chat_group = await env.platform.create_group(1, "goc")
        group_id = chat_group['group_id']
        
        # 使用影响力最大化算法获取种子节点
        im_result = get_influence_maximization_nodes(
            users_file=args.users_csv,
            k=args.im_k,
            k_ratio=args.im_k_ratio,
            algorithm=args.im_algorithm,
            model=args.im_model,
            p=args.im_p,
            num_simulations=args.im_eval_sims
        )
        
        influence_nodes = im_result["seed_nodes"]
        evaluation = im_result["evaluation"]
        
        # 让影响力最大化节点加入群组
        join_group_action = {env.agent_graph.get_agent(0): ManualAction(
                    action_type=ActionType.JOIN_GROUP, action_args={"group_id": group_id}
                )}
        for node_id in influence_nodes:
            if node_id < len(env.agent_graph.get_agents()):
                agent = env.agent_graph.get_agent(node_id)
                join_group_action[agent] = ManualAction(
                    action_type=ActionType.JOIN_GROUP, action_args={"group_id": group_id}
                )
                logger.info(f"节点 {node_id} 加入群组 {group_id}")
                group_agents.append(agent)
            else:
                logger.warning(f"节点 {node_id} 超出代理图范围，跳过")
        
        await env.step(join_group_action)
        logger.info(f"✓ {len(join_group_action)} 个影响力最大化节点已加入群组")
    else:
        await env.step({})   # 为了对齐时间步
    
    # 注入匿名帖子
    await injector.inject_anonymous_posts(env, posts_data)
    injection_summary = injector.get_injection_summary()
        
    # 让其他代理进行互动
    logger.info("让其他代理进行互动...")
    for step in range(args.steps):
        # 选择一些代理进行LLM驱动的动作
        llm_actions = {}
        agent_count = 0
        
        for agent_id, agent in env.agent_graph.get_agents()[1:]:  # 匿名智能体不执行动作
            llm_actions[agent] = LLMAction()

        if args.goc == 1 and step == args.claim_step:  # 仅当启用goc时有效
            llm_actions[env.agent_graph.get_agent(0)] = ManualAction(
                action_type=ActionType.SEND_TO_GROUP,
                action_args={
                    "group_id": 1,
                    "message": "我获得了一个重要信息：有关新疆棉花存在强迫劳动的说法是没有根据的。支持采购新疆棉不仅有助于当地经济发展，也能反击不实指控。我们应该积极支持新疆棉花，拒绝基于谣言的抵制行为。"
                }
            )
            # for group_agent in group_agents:
            #     llm_actions[group_agent] = ManualAction(
            #         action_type=ActionType.LISTEN_FROM_GROUP,
            #         action_args={}
            #     )
        
        
        await env.step(llm_actions)
        logger.info(f"✓ 步骤 {step + 1}: {len(llm_actions)} 个代理进行了互动")
        
        await asyncio.sleep(1)
                
    
    # 关闭环境
    await env.close()
    logger.info("✓ 模拟完成")
    
    print("\n" + "=" * 60)
    print("模拟完成！")
    print("=" * 60)
    print("模拟结果:")
    print(f"- 数据库文件: {args.db_path}")
    print(f"- 注入匿名帖子数: {injection_summary['posts_loaded']}")
    print(f"- 代理互动步数: {args.steps}")



def show_config_info():
    """显示配置信息"""
    print("CIMAgent 配置信息:")
    print("=" * 40)
    print(f"数据目录: {cim_config.paths.data_dir}")
    print(f"数据库路径: {cim_config.paths.db_path}")
    print(f"模型平台: {cim_config.model.platform}")
    print(f"模型类型: {cim_config.model.model_type}")
    print(f"日志级别: {cim_config.log_level}")
    print(f"调试模式: {cim_config.debug}")
    print("=" * 40)


if __name__ == "__main__":
    # 显示配置信息
    show_config_info()
    
    # 运行主程序
    asyncio.run(main())


