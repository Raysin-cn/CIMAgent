import sqlite3
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage
from camel.agents import ChatAgent
import asyncio


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StanceDetector:
    """立场检测器"""
    
    def __init__(self, db_path: str, model_name: str = "gpt-4o-mini"):
        """
        初始化立场检测器
        
        Args:
            db_path: 数据库文件路径
            model_name: 使用的模型名称
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        
    def _init_model(self):
        """初始化LLM模型"""
        if self.model is None:
            try:
                self.model = ModelFactory.create(
                    model_platform=ModelPlatformType.VLLM,
                    model_type="/data/model/Qwen3-14B",
                    url="http://localhost:12345/v1",
                    model_config_dict={"max_tokens":10000}
                )
                self.agent = ChatAgent(
                    system_message="你是一个社交媒体帖子立场识别助手，能够根据帖子内容和话题识别帖子表达的立场。",
                    model=self.model
                )
                logger.info(f"✓ 成功初始化模型: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ 模型初始化失败: {e}")
                raise
    
    def get_user_recent_posts(self, user_id: int, limit: int = 5) -> List[Dict]:
        """
        获取用户最近发布的帖子
        
        Args:
            user_id: 用户ID
            limit: 获取的帖子数量限制
            
        Returns:
            帖子列表，每个帖子包含post_id, content, created_at等信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询用户最近发布的原创帖子（排除转发和引用）
            query = """
                SELECT post_id, content, created_at, num_likes, num_dislikes, num_shares
                FROM post 
                WHERE user_id = ? AND original_post_id IS NULL
                ORDER BY created_at DESC
                LIMIT ?
            """
            
            cursor.execute(query, (user_id, limit))
            posts = cursor.fetchall()
            
            # 转换为字典格式
            post_list = []
            for post in posts:
                post_dict = {
                    'post_id': post[0],
                    'content': post[1],
                    'created_at': post[2],
                    'num_likes': post[3],
                    'num_dislikes': post[4],
                    'num_shares': post[5]
                }
                post_list.append(post_dict)
            
            conn.close()
            return post_list
            
        except Exception as e:
            logger.error(f"❌ 获取用户 {user_id} 的帖子失败: {e}")
            return []
    
    def get_all_users_with_posts(self) -> List[int]:
        """
        获取所有发布过帖子的用户ID列表
        
        Returns:
            用户ID列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询所有发布过原创帖子的用户
            query = """
                SELECT DISTINCT user_id 
                FROM post 
                WHERE original_post_id IS NULL AND user_id != -1
                ORDER BY user_id
            """
            
            cursor.execute(query)
            users = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return users
            
        except Exception as e:
            logger.error(f"❌ 获取用户列表失败: {e}")
            return []
    
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典
        """
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
            logger.error(f"❌ 获取用户 {user_id} 信息失败: {e}")
            return None
    
    async def detect_stance_for_text(self, text: str, topic: str = "中美贸易关税", max_retries: int = 5) -> Dict:
        """
        对单个文本进行立场检测
        
        Args:
            text: 要检测的文本
            topic: 检测的主题
            max_retries: 最大重试次数
            
        Returns:
            立场检测结果，包含立场、置信度、理由等
        """
        self._init_model()
        
        prompt = f"""
请分析以下关于"{topic}"的文本的立场。

文本内容：{text}

请从以下立场中选择一个：
1. 支持 - 明确支持或赞同相关观点
2. 反对 - 明确反对或批评相关观点  
3. 中立 - 保持中立态度，不明确支持或反对
4. 混合 - 同时包含支持和反对的观点

请以JSON格式返回结果：
{{
    "stance": "立场类别",
    "confidence": "置信度(0-1)",
    "reasoning": "分析理由",
    "keywords": ["关键词1", "关键词2"]
}}

只返回JSON格式的结果，不要其他内容。
"""
        for attempt in range(max_retries):
            try:
                response = await self.agent.astep(
                                    BaseMessage.make_user_message(
                    role_name="User",
                    content=(prompt)))
                response_parse = response.msgs[0].content.strip()
                if '<think>' in response_parse and '</think>' in response_parse:
                    start_idx = response_parse.find('</think>') + len('</think>')
                    response_parse = response_parse[start_idx:].strip()
                if response_parse.startswith('\n'):
                    response_parse = response_parse.lstrip()
                result = json.loads(response_parse)
                return result
            
            except json.JSONDecodeError as e:
                logger.warning(f"❌ 第{attempt + 1}次尝试JSON解析失败: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ 立场检测失败，已重试{max_retries}次: {e}")
                    return {
                        "stance": "未知",
                        "confidence": 0.0,
                        "reasoning": f"JSON解析失败: {str(e)}",
                        "keywords": []
                    }
                
            except Exception as e:
                logger.warning(f"❌ 第{attempt + 1}次尝试失败: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ 立场检测失败，已重试{max_retries}次: {e}")
                    return {
                        "stance": "未知",
                        "confidence": 0.0,
                        "reasoning": f"检测失败: {str(e)}",
                        "keywords": []
                    }
    
    async def detect_stance_for_user(self, user_id: int, topic: str = "中美贸易关税", 
                                   post_limit: int = 3) -> Dict:
        """
        对单个用户的所有最近帖子进行立场检测
        
        Args:
            user_id: 用户ID
            topic: 检测的主题
            post_limit: 检测的帖子数量限制
            
        Returns:
            用户立场检测结果
        """
        # 获取用户信息
        user_info = self.get_user_info(user_id)
        if not user_info:
            return {"error": f"用户 {user_id} 不存在"}
        
        # 获取用户最近帖子
        posts = self.get_user_recent_posts(user_id, post_limit)
        if not posts:
            return {
                "user_id": user_id,
                "user_name": user_info.get('user_name', 'Unknown'),
                "name": user_info.get('name', 'Unknown'),
                "posts_analyzed": 0,
                "overall_stance": "无帖子",
                "confidence": 0.0,
                "reasoning": "该用户没有发布过帖子",
                "post_details": []
            }
        
        # 对每个帖子进行立场检测
        post_stances = []
        stance_counts = {"支持": 0, "反对": 0, "中立": 0, "混合": 0, "未知": 0}
        
        for post in posts:
            stance_result = await self.detect_stance_for_text(post['content'], topic)
            post_stance = {
                'post_id': post['post_id'],
                'content': post['content'][:100] + "..." if len(post['content']) > 100 else post['content'],
                'created_at': post['created_at'],
                'stance': stance_result['stance'],
                'confidence': stance_result['confidence'],
                'reasoning': stance_result['reasoning'],
                'keywords': stance_result.get('keywords', [])
            }
            post_stances.append(post_stance)
            
            # 统计立场分布
            stance = stance_result['stance']
            if stance in stance_counts:
                stance_counts[stance] += 1
        
        # 计算整体立场
        total_posts = len(post_stances)
        if total_posts == 0:
            overall_stance = "无帖子"
            confidence = 0.0
        else:
            # 选择出现次数最多的立场作为整体立场
            max_stance = max(stance_counts.items(), key=lambda x: x[1])
            overall_stance = max_stance[0]
            confidence = max_stance[1] / total_posts
        
        # 生成整体分析理由
        reasoning = f"分析了{total_posts}条帖子，立场分布："
        for stance, count in stance_counts.items():
            if count > 0:
                reasoning += f" {stance}({count})"
        
        return {
            "user_id": user_id,
            "user_name": user_info.get('user_name', 'Unknown'),
            "name": user_info.get('name', 'Unknown'),
            "posts_analyzed": total_posts,
            "overall_stance": overall_stance,
            "confidence": confidence,
            "reasoning": reasoning,
            "stance_distribution": stance_counts,
            "post_details": post_stances
        }
    
    async def detect_stance_for_all_users(self, topic: str = "中美贸易关税", 
                                        post_limit: int = 1) -> List[Dict]:
        """
        对所有用户进行立场检测
        
        Args:
            topic: 检测的主题
            post_limit: 每个用户检测的帖子数量限制
            
        Returns:
            所有用户的立场检测结果列表
        """
        users = self.get_all_users_with_posts()
        logger.info(f"开始对 {len(users)} 个用户进行立场检测...")
        
        results = []
        for i, user_id in enumerate(users):
            if i==0: continue  # 跳过匿名智能体
            logger.info(f"正在检测用户 {user_id} ({i+1}/{len(users)})")
            try:
                result = await self.detect_stance_for_user(user_id, topic, post_limit)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ 用户 {user_id} 立场检测失败: {e}")
                results.append({
                    "user_id": user_id,
                    "error": str(e)
                })
        
        return results
    
    def save_stance_results(self, results: List[Dict], output_path: str):
        """
        保存立场检测结果到文件
        
        Args:
            results: 立场检测结果列表
            output_path: 输出文件路径
        """
        try:
            # 保存为JSON格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存为CSV格式（简化版本）
            csv_data = []
            for result in results:
                if 'error' not in result:
                    csv_data.append({
                        'user_id': result['user_id'],
                        'user_name': result['user_name'],
                        'name': result['name'],
                        'posts_analyzed': result['posts_analyzed'],
                        'overall_stance': result['overall_stance'],
                        'confidence': result['confidence'],
                        'reasoning': result['reasoning']
                    })
            
            csv_path = output_path.replace('.json', '.csv')
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"✓ 结果已保存到: {output_path}")
            logger.info(f"✓ CSV格式已保存到: {csv_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
    
    def generate_stance_summary(self, results: List[Dict]) -> Dict:
        """
        生成立场检测结果摘要
        
        Args:
            results: 立场检测结果列表
            
        Returns:
            摘要统计信息
        """
        total_users = len(results)
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {"error": "没有有效的检测结果"}
        
        # 统计立场分布
        stance_distribution = {}
        total_posts = 0
        
        for result in valid_results:
            stance = result['overall_stance']
            stance_distribution[stance] = stance_distribution.get(stance, 0) + 1
            total_posts += result['posts_analyzed']
        
        # 计算平均置信度
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        
        return {
            "total_users": total_users,
            "valid_users": len(valid_results),
            "total_posts_analyzed": total_posts,
            "stance_distribution": stance_distribution,
            "average_confidence": avg_confidence,
            "most_common_stance": max(stance_distribution.items(), key=lambda x: x[1])[0] if stance_distribution else "无"
        }
    
    def get_user_posts_before_timestep(self, user_id: int, timestep: str, limit: int = 5) -> List[Dict]:
        """
        获取用户在指定时间步之前最近发布的帖子
        
        Args:
            user_id: 用户ID
            timestep: 时间步（时间字符串）
            limit: 获取的帖子数量限制
            
        Returns:
            帖子列表，每个帖子包含post_id, content, created_at等信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询用户在指定时间步之前最近发布的原创帖子
            query = """
                SELECT post_id, content, created_at, num_likes, num_dislikes, num_shares
                FROM post 
                WHERE user_id = ? AND original_post_id IS NULL
                AND created_at <= ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            
            cursor.execute(query, (user_id, timestep, limit))
            posts = cursor.fetchall()
            
            # 转换为字典格式
            post_list = []
            for post in posts:
                post_dict = {
                    'post_id': post[0],
                    'content': post[1],
                    'created_at': post[2],
                    'num_likes': post[3],
                    'num_dislikes': post[4],
                    'num_shares': post[5]
                }
                post_list.append(post_dict)
            
            conn.close()
            return post_list
            
        except Exception as e:
            logger.error(f"❌ 获取用户 {user_id} 在时间步 {timestep} 之前的帖子失败: {e}")
            return []
    
    def get_all_posts_with_timestamps(self) -> List[Dict]:
        """
        获取所有帖子及其时间戳信息
        
        Returns:
            所有帖子的列表，包含时间信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询所有原创帖子，按时间排序
            query = """
                SELECT p.post_id, p.user_id, p.content, p.created_at, 
                       p.num_likes, p.num_dislikes, p.num_shares
                FROM post p
                WHERE p.original_post_id IS NULL AND p.user_id != -1
                ORDER BY p.created_at ASC
            """
            
            cursor.execute(query)
            posts = cursor.fetchall()
            
            # 转换为字典格式
            post_list = []
            for post in posts:
                post_dict = {
                    'post_id': post[0],
                    'user_id': post[1],
                    'content': post[2],
                    'created_at': post[3],
                    'num_likes': post[4],
                    'num_dislikes': post[5],
                    'num_shares': post[6]
                }
                post_list.append(post_dict)
            
            conn.close()
            return post_list
            
        except Exception as e:
            logger.error(f"❌ 获取所有帖子失败: {e}")
            return []
    
    def get_timesteps_from_post(self) -> List[int]:
        """
        从post表中获取所有时间步
        
        Returns:
            时间步列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询所有帖子的创建时间作为时间步信息
            query = """
                SELECT DISTINCT created_at
                FROM post 
                WHERE original_post_id IS NULL AND user_id != -1
                ORDER BY created_at ASC
            """
            
            cursor.execute(query)
            timestep_records = cursor.fetchall()
            timesteps = [x[0] for x in timestep_records]
            return timesteps
            
        except Exception as e:
            logger.error(f"❌ 获取时间步失败: {e}")
            return []
    
    async def analyze_all_posts_stance(self, topic: str = "中美贸易关税", 
                                     batch_size: int = 50, 
                                     max_concurrent: int = 10) -> Dict[int, Dict]:
        """
        分析所有帖子的立场（优化版本，支持分批处理和并发控制）
        
        Args:
            topic: 检测的主题
            batch_size: 每批处理的帖子数量
            max_concurrent: 最大并发数量
            
        Returns:
            帖子ID到立场结果的映射
        """
        posts = self.get_all_posts_with_timestamps()
        logger.info(f"开始分析 {len(posts)} 个帖子的立场...")
        
        # 过滤掉匿名用户的帖子
        valid_posts = [post for post in posts if post['user_id'] != 0]
        logger.info(f"有效帖子数量: {len(valid_posts)}")
        
        if not valid_posts:
            logger.warning("没有找到有效的帖子")
            return {}
        
        # 分批处理
        total_batches = (len(valid_posts) + batch_size - 1) // batch_size
        logger.info(f"将分 {total_batches} 批处理，每批最多 {batch_size} 个帖子")
        
        post_stances = {}
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(valid_posts))
            batch_posts = valid_posts[start_idx:end_idx]
            
            logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch_posts)} 个帖子")
            
            # 创建当前批次的任务
            tasks = []
            for post in batch_posts:
                tasks.append(self.detect_stance_for_text(post['content'], topic))
            
            # 使用信号量控制并发数量
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await task
            
            # 执行当前批次的任务
            batch_tasks = [process_with_semaphore(task) for task in tasks]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理当前批次的结果
            for i, (post, result) in enumerate(zip(batch_posts, batch_results)):
                post_id = post['post_id']
                
                if isinstance(result, Exception):
                    logger.error(f"❌ 帖子 {post_id} 立场分析失败: {result}")
                    post_stances[post_id] = {
                        'user_id': post['user_id'],
                        'content': post['content'][:100] + "..." if len(post['content']) > 100 else post['content'],
                        'created_at': post['created_at'],
                        'stance': "未知",
                        'confidence': 0.0,
                        'reasoning': f"分析失败: {str(result)}",
                        'keywords': []
                    }
                else:
                    post_stances[post_id] = {
                        'user_id': post['user_id'],
                        'content': post['content'][:100] + "..." if len(post['content']) > 100 else post['content'],
                        'created_at': post['created_at'],
                        'stance': result['stance'],
                        'confidence': result['confidence'],
                        'reasoning': result['reasoning'],
                        'keywords': result.get('keywords', [])
                    }
                
                # 每处理10个帖子输出一次进度
                if (i + 1) % 10 == 0:
                    logger.info(f"批次 {batch_idx + 1} 进度: {i + 1}/{len(batch_posts)}")
            
            logger.info(f"✅ 第 {batch_idx + 1} 批处理完成")
            
            # 批次间短暂休息，避免API限制
            if batch_idx < total_batches - 1:
                await asyncio.sleep(1)
        
        logger.info(f"✅ 所有帖子立场分析完成，共处理 {len(post_stances)} 个帖子")
        return post_stances
    
    def get_user_stance_at_timestep(self, user_id: int, timestep: int, 
                                  post_stances: Dict[int, Dict], 
                                  limit: int = 3) -> Dict:
        """
        获取用户在指定时间步的立场
        
        Args:
            user_id: 用户ID
            timestep: 时间步
            post_stances: 所有帖子的立场分析结果
            limit: 考虑的最多帖子数量
            
        Returns:
            用户在指定时间步的立场信息
        """
        # 获取用户在时间步之前的帖子
        posts = self.get_user_posts_before_timestep(user_id, timestep, limit)
        
        if not posts:
            return {
                "user_id": user_id,
                "timestep": timestep,
                "posts_analyzed": 0,
                "stance": "未知",
                "confidence": 0.0,
                "reasoning": "该用户在此时刻之前没有发布过帖子"
            }
        
        # 统计立场分布
        stance_counts = {"支持": 0, "反对": 0, "中立": 0, "混合": 0, "未知": 0}
        analyzed_posts = []
        
        post = posts[-1]
        if post['post_id'] in post_stances:
            stance_info = post_stances[post['post_id']]
            stance = stance_info['stance']
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
            analyzed_posts.append(stance_info)
        
        # 计算整体立场
        total_posts = len(analyzed_posts)
        if total_posts == 0:
            return {
                "user_id": user_id,
                "timestep": timestep,
                "posts_analyzed": 0,
                "stance": "未知",
                "confidence": 0.0,
                "reasoning": "没有可分析的帖子"
            }
        
        # 选择出现次数最多的立场作为整体立场
        max_stance = max(stance_counts.items(), key=lambda x: x[1])
        overall_stance = max_stance[0]
        confidence = max_stance[1] / total_posts
        
        # 生成分析理由
        reasoning = f"时间步{timestep}时分析了{total_posts}条帖子，立场分布："
        for stance, count in stance_counts.items():
            if count > 0:
                reasoning += f" {stance}({count})"
        
        return {
            "user_id": user_id,
            "timestep": timestep,
            "posts_analyzed": total_posts,
            "stance": overall_stance,
            "confidence": confidence,
            "reasoning": reasoning,
            "stance_distribution": stance_counts,
            "post_details": analyzed_posts
        }
    
    async def analyze_stance_evolution(self, topic: str = "中美贸易关税", 
                                     post_limit: int = 3) -> Dict:
        """
        分析用户立场随时间步的演化
        
        Args:
            topic: 检测的主题
            post_limit: 每个时间步考虑的最多帖子数量
            
        Returns:
            用户立场演化分析结果
        """
        # 获取所有时间步
        timesteps = self.get_timesteps_from_post()
        if not timesteps:
            logger.error("❌ 没有找到时间步信息")
            return {"error": "没有找到时间步信息"}
        
        # 获取所有用户
        users = self.get_all_users_with_posts()
        if not users:
            logger.error("❌ 没有找到用户")
            return {"error": "没有找到用户"}
        
        # 分析所有帖子的立场
        post_stances = await self.analyze_all_posts_stance(topic)
        
        # 分析每个用户在每个时间步的立场
        evolution_results = {}
        
        for user_id in users:
            if user_id == 0:  # 跳过匿名用户
                continue
                
            user_info = self.get_user_info(user_id)
            if not user_info:
                continue
            
            user_evolution = {
                "user_id": user_id,
                "user_name": user_info.get('user_name', 'Unknown'),
                "name": user_info.get('name', 'Unknown'),
                "timestep_stances": []
            }
            
            for timestep in timesteps:
                stance_at_step = self.get_user_stance_at_timestep(
                    user_id, timestep, post_stances, post_limit
                )
                user_evolution["timestep_stances"].append(stance_at_step)
            
            evolution_results[user_id] = user_evolution
        
        return {
            "topic": topic,
            "timesteps": timesteps,
            "total_users": len(evolution_results),
            "user_evolution": evolution_results
        }