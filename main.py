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
from cim import OasisPostInjector, DataManager, config
from cim.core.stance_detector import StanceDetector
from cim.core.influence_max import follow_matrix_get, get_influence_maximization_nodes, compare_influence_algorithms
from cim.config import config as cim_config


from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import (
    ActionType, 
    ManualAction, 
    LLMAction,
    generate_twitter_agent_graph
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
    
    # 数据文件参数
    parser.add_argument("--users_csv", 
                       default=cim_config.post_generation.users_file,
                       help="用户数据CSV文件路径（用于创建代理图）")
    parser.add_argument("--posts_csv", 
                       default=cim_config.get_file_path("processed", "generated_posts.csv"),
                       help="生成的帖子CSV文件路径（将被作为匿名帖子注入）")
    parser.add_argument("--profile_output", 
                       default=cim_config.get_file_path("processed", "oasis_user_profiles.csv"),
                       help="Oasis用户档案输出路径")
    
    # 数据库参数
    parser.add_argument("--db_path", 
                       default=cim_config.database.path,
                       help="模拟数据库路径")
    
    # 模拟参数
    parser.add_argument("--steps", type=int, default=10, 
                       help="模拟步数（代理互动步数）")
    parser.add_argument("--goc", action="store_true", default=True,
                       help="启用群组意见交流模式（默认启用）")
    
    # 数据管理参数
    parser.add_argument("--backup", action="store_true", default=False,
                       help="在运行前备份数据库")
    parser.add_argument("--cleanup", action="store_true", default=False,
                       help="运行后清理临时文件")
    
    # 影响力最大化算法参数
    parser.add_argument("--im_k", type=int, default=5,
                       help="影响力最大化种子节点数量（默认5个）")
    parser.add_argument("--im_algorithm", type=str, default="Greedy", choices=["Greedy", "Random"],
                       help="影响力最大化算法：Greedy(贪心算法) 或 Random(随机算法)")
    parser.add_argument("--im_model", type=str, default="IC", choices=["IC", "LT"],
                       help="传播模型类型：IC(Independent Cascade) 或 LT(Linear Threshold)")
    parser.add_argument("--im_p", type=float, default=0.1,
                       help="传播概率（默认0.1）")
    parser.add_argument("--im_eval_sims", type=int, default=1000,
                       help="影响力评估模拟次数（默认1000）")

    # 告知者信息发布参数
    parser.add_argument("--claim_step", type=int, default=4,
                        help="告知者在哪一步发布关键信息")
    
    
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
    
    # 若使用默认 db_path，则根据超参数自动生成区分不同实验的DB文件名，并保存到 ./data/simu/
    try:
        default_db_path = cim_config.database.path
        if args.db_path == default_db_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p_str = f"{args.im_p:.3f}".replace(".", "p")
            goc_flag = 1 if args.goc else 0
            filename = (
                f"oasis_sim_"
                f"steps{args.steps}_goc{goc_flag}_claim{args.claim_step}_"
                f"k{args.im_k}_{args.im_algorithm}_{args.im_model}_"
                f"p{p_str}_sims{args.im_eval_sims}_{ts}.db"
            )
            # 保存到 ./data/simu/ 目录下（若不存在则创建）
            simu_dir = Path("./data/simu").resolve()
            simu_dir.mkdir(parents=True, exist_ok=True)
            args.db_path = str(simu_dir / filename)
            logger.info(f"自动生成实验数据库路径: {args.db_path}")
    except Exception as e:
        logger.warning(f"自动命名数据库文件失败，使用原始路径: {args.db_path}. 错误: {e}")
    
    print("CIMAgent Oasis社交网络模拟 - 匿名帖子注入")
    print("=" * 60)
    print("注意：所有生成的帖子将作为匿名帖子注入到系统中")
    print("匿名帖子的发布者不会参与后续的社交网络演进")
    print("=" * 60)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 1. 数据备份（如果启用）
    if args.backup:
        logger.info("备份现有数据库...")
        backup_path = data_manager.backup_database(args.db_path)
        print(f"✓ 数据库已备份到: {backup_path}")
    
    # 2. 初始化注入器
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
    logger.info("创建用户档案...")
    profile_path = injector.create_user_profile_csv(args.profile_output)
    print(f"✓ 用户档案已创建: {profile_path}")
    
    # 6. 运行模拟（包含匿名帖子注入）
    model_config = cim_config.model
    logger.info("运行模拟")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=model_config.model_type,
        url=model_config.url
    )
    # 定义可用动作
    available_actions = ActionType.get_default_twitter_actions()
    # 生成代理图
    logger.info("生成代理图...")
    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=available_actions,
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
    if args.goc:
        chat_group = await env.platform.create_group(1, "goc")
        group_id = chat_group['group_id']
        
        # 使用影响力最大化算法获取种子节点
        im_result = get_influence_maximization_nodes(
            config=cim_config,
            k=args.im_k,
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
    await injector.inject_anonymous_posts(env, injector.generated_posts)
    injection_summary = injector.get_injection_summary()
        
    # 让其他代理进行互动
    logger.info("让其他代理进行互动...")
    for step in range(args.steps):
        # 选择一些代理进行LLM驱动的动作
        llm_actions = {}
        agent_count = 0
        
        for agent_id, agent in env.agent_graph.get_agents()[1:]:  # 匿名智能体不执行动作
            llm_actions[agent] = LLMAction()

        if step == args.claim_step:
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
    print(f"- 用户档案: {args.profile_output}")
    print(f"- 注入匿名帖子数: {injection_summary['posts_loaded']}")
    print(f"- 代理互动步数: {args.steps}")
    print(f"- 运行时间: {injection_summary['injection_time']}")

    # 7. 模拟后立场分析与可视化
    logger.info("开始立场分析...")
    detector = StanceDetector(db_path=args.db_path)
    stance_results = await detector.detect_stance_for_all_users(topic=None, post_limit=None)
    # 结果保存到 ./data/output 下，文件名关联本次实验
    output_dir = Path(cim_config.paths.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    db_stem = Path(args.db_path).stem
    stance_json_path = str(output_dir / f"{db_stem}_stance_results.json")
    await detector.save_stance_results(stance_results, stance_json_path)
    logger.info(f"立场分析结果已保存到: {stance_json_path}")

    # 立场演化绘图
    evolution_fig_path = stance_json_path.replace('.json', '_evolution.png')
    try:
        detector.plot_users_stance_evolution(stance_results, save_path=evolution_fig_path, alpha=0.7)
        logger.info(f"立场演化图已保存到: {evolution_fig_path}")
    except Exception as e:
        logger.warning(f"立场演化绘图失败: {e}")
    
    # 显示影响力最大化算法结果
    if args.goc:
        print(f"- 影响力最大化算法: {args.im_algorithm}")
        print(f"- 种子节点数: {args.im_k}")
        print(f"- 传播模型: {args.im_model}")
        print(f"- 传播概率: {args.im_p}")
        print(f"- 影响力评估: {evaluation['influence']:.2f} 节点 ({evaluation['influence_ratio']:.2%})")
        print(f"- 种子节点列表: {influence_nodes}")
    
    print("\n匿名帖子说明:")
    print("- 所有帖子都以匿名用户身份发布（user_id = 0）")
    print("- 匿名用户不会参与后续的社交网络互动")
    print("- 匿名帖子会出现在推荐系统中，供其他代理查看和互动")
    print("- 可以通过数据库查询验证匿名帖子的存在")
    
    # 8. 数据清理（如果启用）
    if args.cleanup:
        logger.info("清理临时文件...")
        deleted_count = data_manager.cleanup_temp_files()
        print(f"✓ 清理了 {deleted_count} 个临时文件")
    
    # 9. 生成数据摘要报告
    logger.info("生成数据摘要报告...")
    summary_path = data_manager.export_data_summary()
    print(f"✓ 数据摘要报告已生成: {summary_path}")
    
    print("\n" + "=" * 60)
    print("模拟操作完成！")
    print("=" * 60)
        

def show_config_info():
    """显示配置信息"""
    print("CIMAgent 配置信息:")
    print("=" * 40)
    print(f"数据目录: {cim_config.paths.data_dir}")
    print(f"数据库路径: {cim_config.database.path}")
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


