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

# 导入CIM模块
from cim import OasisPostInjector, DataManager, config
from cim.config import config as cim_config


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
                       help="用户数据CSV文件路径（用于创建智能体图）")
    parser.add_argument("--posts_json", 
                       default=cim_config.get_file_path("processed", "generated_posts.json"),
                       help="生成的帖子JSON文件路径（将被作为匿名帖子注入）")
    parser.add_argument("--profile_output", 
                       default=cim_config.get_file_path("processed", "oasis_user_profiles.csv"),
                       help="Oasis用户档案输出路径")
    
    # 数据库参数
    parser.add_argument("--db_path", 
                       default=cim_config.database.path,
                       help="模拟数据库路径")
    
    # 模拟参数
    parser.add_argument("--steps", type=int, default=8,
                       help="模拟步数（代理互动步数）")
    
    # 数据管理参数
    parser.add_argument("--backup", action="store_true",
                       help="在运行前备份数据库")
    parser.add_argument("--cleanup", action="store_true",
                       help="运行后清理临时文件")
    
    # 调试参数
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        cim_config.debug = True
        cim_config.log_level = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    print("CIMAgent Oasis社交网络模拟 - 匿名帖子注入")
    print("=" * 60)
    print("注意：所有生成的帖子将作为匿名帖子注入到系统中")
    print("匿名帖子的发布者不会参与后续的社交网络演进")
    print("=" * 60)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    try:
        # 1. 数据备份（如果启用）
        if args.backup:
            logger.info("1. 备份现有数据库...")
            backup_path = data_manager.backup_database(args.db_path)
            print(f"✓ 数据库已备份到: {backup_path}")
        
        # 2. 初始化注入器
        logger.info("2. 初始化帖子注入器...")
        injector = OasisPostInjector(db_path=args.db_path)
        
        # 3. 加载数据
        logger.info("4. 加载数据...")
        users_data = injector.load_users_data(args.users_csv)
        posts_data = injector.load_generated_posts(args.posts_json)
        
        print(f"✓ 加载了 {len(users_data)} 个用户数据")
        print(f"✓ 加载了 {len(posts_data)} 条生成的帖子")

        # 4. 验证数据文件
        logger.info("3. 验证数据文件...")
        validation_results = injector.validate_data()
        
        if not validation_results["users_data_valid"]:
            raise FileNotFoundError(f"用户数据文件不存在或无效: {args.users_csv}")
        
        if not validation_results["posts_data_valid"]:
            raise FileNotFoundError(f"帖子数据文件不存在或无效: {args.posts_json}")
        
        print("✓ 数据文件验证通过")
        
        # 5. 创建用户档案（用于创建代理图）
        logger.info("5. 创建用户档案...")
        profile_path = injector.create_user_profile_csv(args.profile_output)
        print(f"✓ 用户档案已创建: {profile_path}")
        
        # 6. 运行模拟（包含匿名帖子注入）
        logger.info("6. 运行模拟并注入匿名帖子...")
        env = await injector.run_simulation_with_posts(
            
            profile_path=profile_path,
            posts=injector.generated_posts,  # 所有帖子将作为匿名帖子注入
            num_steps=args.steps
        )
        
        # 7. 生成运行摘要
        injection_summary = injector.get_injection_summary()
        
        print("\n" + "=" * 60)
        print("模拟完成！")
        print("=" * 60)
        print("模拟结果:")
        print(f"- 数据库文件: {args.db_path}")
        print(f"- 用户档案: {args.profile_output}")
        print(f"- 注入匿名帖子数: {injection_summary['posts_loaded']}")
        print(f"- 代理互动步数: {args.steps}")
        print(f"- 运行时间: {injection_summary['injection_time']}")
        
        print("\n匿名帖子说明:")
        print("- 所有帖子都以匿名用户身份发布（user_id = 0）")
        print("- 匿名用户不会参与后续的社交网络互动")
        print("- 匿名帖子会出现在推荐系统中，供其他代理查看和互动")
        print("- 可以通过数据库查询验证匿名帖子的存在")
        
        # 8. 数据清理（如果启用）
        if args.cleanup:
            logger.info("8. 清理临时文件...")
            deleted_count = data_manager.cleanup_temp_files()
            print(f"✓ 清理了 {deleted_count} 个临时文件")
        
        # 9. 生成数据摘要报告
        logger.info("9. 生成数据摘要报告...")
        summary_path = data_manager.export_data_summary()
        print(f"✓ 数据摘要报告已生成: {summary_path}")
        
        print("\n" + "=" * 60)
        print("所有操作完成！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件不存在: {e}")
        print(f"❌ 文件不存在: {e}")
        print("请检查文件路径是否正确，或使用 --help 查看参数说明")
        
    except Exception as e:
        logger.error(f"❌ 运行出错: {e}")
        print(f"❌ 运行出错: {e}")
        if cim_config.debug:
            import traceback
            traceback.print_exc()


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


