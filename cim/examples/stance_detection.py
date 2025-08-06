#!/usr/bin/env python3
"""
立场检测模块
针对Oasis社交网络模拟数据库中每个用户最近发布的帖子进行立场检测
支持时间步演化分析
"""

from StanceDetector import StanceDetector
import json
import pandas as pd


async def main():
    """主函数 - 运行立场检测"""
    import argparse
    
    parser = argparse.ArgumentParser(description="用户帖子立场检测")
    parser.add_argument("--db_path", default="./data/twitter_simulation.db", 
                       help="数据库文件路径")
    parser.add_argument("--output", default="./data/stance_detection_results.json", 
                       help="输出结果文件路径")
    parser.add_argument("--topic", default="中美贸易关税", 
                       help="检测主题")
    parser.add_argument("--post_limit", type=int, default=1, 
                       help="每个用户检测的帖子数量限制")
    parser.add_argument("--user_id", type=int, default=None, 
                       help="指定用户ID进行检测（不指定则检测所有用户）")
    parser.add_argument("--evolution", type=int, default=1, 
                       help="启用立场演化分析模式")
    parser.add_argument("--timestep", type=int, default=10, 
                       help="指定时间步进行检测（仅在evolution模式下有效）")
    
    args = parser.parse_args()
    
    print("用户帖子立场检测")
    print("=" * 50)
    
    # 初始化检测器
    detector = StanceDetector(args.db_path)
    
    try:
        if args.evolution:
            # 立场演化分析模式
            print("🔍 开始立场演化分析...")
            evolution_results = await detector.analyze_stance_evolution(
                args.topic, args.post_limit
            )
            
            if "error" in evolution_results:
                print(f"❌ 演化分析失败: {evolution_results['error']}")
                return
            
            # 保存演化结果
            output_path = args.output.replace('.json', '_evolution.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_results, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 演化分析结果已保存到: {output_path}")
            
            # 生成演化摘要
            print("\n📈 立场演化分析摘要:")
            print(f"- 分析主题: {evolution_results['topic']}")
            print(f"- 时间步数量: {len(evolution_results['timesteps'])}")
            print(f"- 用户数量: {evolution_results['total_users']}")
            
            # 生成CSV格式的演化数据
            csv_data = []
            for user_id, user_evolution in evolution_results['user_evolution'].items():
                for stance_info in user_evolution['timestep_stances']:
                    csv_data.append({
                        'user_id': user_id,
                        'user_name': user_evolution['user_name'],
                        'name': user_evolution['name'],
                        'timestep': stance_info['timestep'],
                        'posts_analyzed': stance_info['posts_analyzed'],
                        'stance': stance_info['stance'],
                        'confidence': stance_info['confidence'],
                        'reasoning': stance_info['reasoning']
                    })
            
            csv_path = output_path.replace('.json', '.csv')
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 演化数据CSV已保存到: {csv_path}")
            
        elif args.timestep is not None:
            # 指定时间步分析
            print(f"🔍 分析时间步 {args.timestep} 的用户立场...")
            
            # 先分析所有帖子的立场
            post_stances = await detector.analyze_all_posts_stance(args.topic)
            
            if args.user_id:
                # 分析指定用户在指定时间步的立场
                stance_result = detector.get_user_stance_at_timestep(
                    args.user_id, args.timestep, post_stances, args.post_limit
                )
                results = [stance_result]
            else:
                # 分析所有用户在指定时间步的立场
                users = detector.get_all_users_with_posts()
                results = []
                for user_id in users:
                    if user_id == -1:  # 跳过匿名用户
                        continue
                    stance_result = detector.get_user_stance_at_timestep(
                        user_id, args.timestep, post_stances, args.post_limit
                    )
                    results.append(stance_result)
            
            # 保存结果
            detector.save_stance_results(results, args.output)
            
            # 生成摘要
            summary = detector.generate_stance_summary(results)
            print(f"\n时间步 {args.timestep} 检测结果摘要:")
            print(f"- 总用户数: {summary['total_users']}")
            print(f"- 有效用户数: {summary['valid_users']}")
            print(f"- 分析帖子总数: {summary['total_posts_analyzed']}")
            print(f"- 平均置信度: {summary['average_confidence']:.2f}")
            print(f"- 最常见立场: {summary['most_common_stance']}")
            print("\n立场分布:")
            for stance, count in summary['stance_distribution'].items():
                print(f"  {stance}: {count} 人")
        
        else:
            # 传统模式：检测当前所有帖子的立场
            if args.user_id:
                # 检测指定用户
                print(f"检测用户 {args.user_id} 的立场...")
                result = await detector.detect_stance_for_user(args.user_id, args.topic, args.post_limit)
                results = [result]
            else:
                # 检测所有用户
                print("检测所有用户的立场...")
                results = await detector.detect_stance_for_all_users(args.topic, args.post_limit)
            
            # 保存结果
            detector.save_stance_results(results, args.output)
            
            # 生成摘要
            summary = detector.generate_stance_summary(results)
            print("\n检测结果摘要:")
            print(f"- 总用户数: {summary['total_users']}")
            print(f"- 有效用户数: {summary['valid_users']}")
            print(f"- 分析帖子总数: {summary['total_posts_analyzed']}")
            print(f"- 平均置信度: {summary['average_confidence']:.2f}")
            print(f"- 最常见立场: {summary['most_common_stance']}")
            print("\n立场分布:")
            for stance, count in summary['stance_distribution'].items():
                print(f"  {stance}: {count} 人")
        
    except Exception as e:
        print(f"❌ 检测过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
