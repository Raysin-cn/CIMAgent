#!/usr/bin/env python3
"""
读取Oasis数据库文件中的trace表并结构化打印信息
"""

import sqlite3
import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import argparse
import os

class TraceTableReader:
    """Trace表读取器"""
    
    def __init__(self, db_path: str):
        """
        初始化Trace表读取器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        
    def check_db_exists(self) -> bool:
        """检查数据库文件是否存在"""
        return os.path.exists(self.db_path)
    
    def get_trace_data(self, limit: Optional[int] = None, 
                      user_id: Optional[int] = None,
                      action_type: Optional[str] = None) -> List[Dict]:
        """
        获取trace表数据
        
        Args:
            limit: 限制返回的记录数量
            user_id: 过滤特定用户ID
            action_type: 过滤特定动作类型
            
        Returns:
            trace数据列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构建查询语句
            query = """
                SELECT user_id, created_at, action, info
                FROM trace
            """
            params = []
            
            # 添加过滤条件
            conditions = []
            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if action_type is not None:
                conditions.append("action = ?")
                params.append(action_type)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # 添加排序和限制
            query += " ORDER BY created_at DESC, user_id DESC"
            
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # 转换为字典格式
            trace_data = []
            for row in rows:
                trace_record = {
                    'user_id': row[0],
                    'created_at': row[1],
                    'action': row[2],
                    'info': row[3]
                }
                trace_data.append(trace_record)
            
            conn.close()
            return trace_data
            
        except Exception as e:
            print(f"❌ 读取trace表失败: {e}")
            return []
    
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """获取用户信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT user_id, agent_id, user_name, name, bio FROM user WHERE user_id = ?"
            cursor.execute(query, (user_id,))
            user = cursor.fetchone()
            
            conn.close()
            
            if user:
                return {
                    'user_id': user[0],
                    'agent_id': user[1],
                    'user_name': user[2],
                    'name': user[3],
                    'bio': user[4]
                }
            return None
            
        except Exception as e:
            print(f"❌ 获取用户 {user_id} 信息失败: {e}")
            return None
    
    def get_trace_summary(self) -> Dict:
        """获取trace表摘要统计"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM trace")
            total_records = cursor.fetchone()[0]
            
            # 用户数量
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM trace")
            unique_users = cursor.fetchone()[0]
            
            # 动作类型统计
            cursor.execute("SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY COUNT(*) DESC")
            action_stats = dict(cursor.fetchall())
            
            # 时间范围
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM trace")
            time_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_records': total_records,
                'unique_users': unique_users,
                'action_statistics': action_stats,
                'time_range': {
                    'start': time_range[0] if time_range[0] else 'N/A',
                    'end': time_range[1] if time_range[1] else 'N/A'
                }
            }
            
        except Exception as e:
            print(f"❌ 获取trace摘要失败: {e}")
            return {}
    
    def parse_info_field(self, info: str) -> Dict:
        """解析info字段的JSON数据"""
        try:
            if info and info.strip():
                return json.loads(info)
            return {}
        except json.JSONDecodeError:
            # 如果不是JSON格式，返回原始字符串
            return {'raw_info': info}
    
    def print_trace_data(self, trace_data: List[Dict], show_user_info: bool = True):
        """结构化打印trace数据"""
        if not trace_data:
            print("📭 没有找到trace数据")
            return
        
        print(f"\n📊 找到 {len(trace_data)} 条trace记录")
        print("=" * 80)
        
        for i, record in enumerate(trace_data, 1):
            print(f"\n🔍 记录 #{i}")
            print("-" * 40)
            
            # 基本信息
            print(f"👤 用户ID: {record['user_id']}")
            print(f"⏰ 时间: {record['created_at']}")
            print(f"🎯 动作: {record['action']}")
            
            # 用户信息
            if show_user_info and record['user_id'] != -1:
                user_info = self.get_user_info(record['user_id'])
                if user_info:
                    print(f"📝 用户名: {user_info.get('user_name', 'N/A')}")
                    print(f"🏷️  显示名: {user_info.get('name', 'N/A')}")
                    if user_info.get('bio'):
                        print(f"💬 简介: {user_info['bio'][:100]}{'...' if len(user_info['bio']) > 100 else ''}")
            
            # 解析info字段
            if record['info']:
                parsed_info = self.parse_info_field(record['info'])
                print(f"📋 详细信息:")
                for key, value in parsed_info.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"   {key}: {value[:100]}...")
                    else:
                        print(f"   {key}: {value}")
            else:
                print("📋 详细信息: 无")
    
    def print_summary(self):
        """打印trace表摘要"""
        summary = self.get_trace_summary()
        if not summary:
            return
        
        print("\n📈 Trace表摘要统计")
        print("=" * 50)
        print(f"📊 总记录数: {summary['total_records']}")
        print(f"👥 涉及用户数: {summary['unique_users']}")
        
        if summary['time_range']['start'] != 'N/A':
            print(f"⏰ 时间范围: {summary['time_range']['start']} 到 {summary['time_range']['end']}")
        
        print(f"\n🎯 动作类型统计:")
        for action, count in summary['action_statistics'].items():
            print(f"   {action}: {count} 次")
    
    def export_to_csv(self, trace_data: List[Dict], output_path: str):
        """导出trace数据到CSV文件"""
        try:
            # 准备CSV数据
            csv_data = []
            for record in trace_data:
                # 获取用户信息
                user_info = self.get_user_info(record['user_id']) if record['user_id'] != -1 else None
                
                # 解析info字段
                parsed_info = self.parse_info_field(record['info'])
                
                csv_record = {
                    'user_id': record['user_id'],
                    'user_name': user_info.get('user_name', 'anonymous') if user_info else 'anonymous',
                    'name': user_info.get('name', 'anonymous') if user_info else 'anonymous',
                    'created_at': record['created_at'],
                    'action': record['action'],
                    'info': record['info'],
                    'parsed_info': json.dumps(parsed_info, ensure_ascii=False) if parsed_info else ''
                }
                csv_data.append(csv_record)
            
            # 保存为CSV
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ 数据已导出到: {output_path}")
            
        except Exception as e:
            print(f"❌ 导出CSV失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="读取Oasis数据库trace表")
    parser.add_argument("--db_path", default="./data/twitter_simulation.db", 
                       help="数据库文件路径")
    parser.add_argument("--limit", type=int, default=20, 
                       help="显示记录数量限制")
    parser.add_argument("--user_id", type=int, default=None, 
                       help="过滤特定用户ID")
    parser.add_argument("--action", default=None, 
                       help="过滤特定动作类型")
    parser.add_argument("--no_user_info", action="store_true", 
                       help="不显示用户详细信息")
    parser.add_argument("--export_csv", default=None, 
                       help="导出到CSV文件路径")
    parser.add_argument("--summary_only", action="store_true", 
                       help="只显示摘要统计")
    
    args = parser.parse_args()
    
    print("🔍 Oasis数据库Trace表读取器")
    print("=" * 50)
    
    # 初始化读取器
    reader = TraceTableReader(args.db_path)
    
    # 检查数据库文件
    if not reader.check_db_exists():
        print(f"❌ 数据库文件不存在: {args.db_path}")
        return
    
    try:
        # 显示摘要
        reader.print_summary()
        
        if not args.summary_only:
            # 获取trace数据
            trace_data = reader.get_trace_data(
                limit=args.limit,
                user_id=args.user_id,
                action_type=args.action
            )
            
            # 显示trace数据
            reader.print_trace_data(trace_data, show_user_info=not args.no_user_info)
            
            # 导出CSV
            if args.export_csv:
                reader.export_to_csv(trace_data, args.export_csv)
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
