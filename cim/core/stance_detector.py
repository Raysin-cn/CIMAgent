"""
立场检测器模块

提供社交媒体帖子立场检测功能，支持：
- 单文本立场检测
- 用户立场分析
- 时间演化分析
- 批量处理
"""

import sqlite3
import json
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage
from camel.agents import ChatAgent

from ..config import config


# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class StanceDetector:
    """立场检测器"""
    
    def __init__(self, db_path: Optional[str] = None, model_config: Optional[Dict] = None):
        """
        初始化立场检测器
        
        Args:
            db_path: 数据库文件路径，如果为None则使用配置中的默认路径
            model_config: 模型配置，如果为None则使用配置中的默认配置
        """
        self.db_path = db_path or config.database.path
        self.model_config = model_config or config.model.__dict__
        self.model = None
        self.agent = None
        
        # 立场类别定义
        self.stance_categories = ["支持", "反对", "中立", "混合"]
        
        logger.info(f"初始化立场检测器，数据库路径: {self.db_path}")
    
    def _init_model(self):
        """初始化LLM模型"""
        if self.model is None:
            try:
                # 根据配置创建模型
                if self.model_config["platform"].upper() == "VLLM":
                    self.model = ModelFactory.create(
                        model_platform=ModelPlatformType.VLLM,
                        model_type=self.model_config["model_type"],
                        url=self.model_config["url"],
                        model_config_dict={"max_tokens": self.model_config["max_tokens"]}
                    )
                else:
                    # 默认使用OpenAI
                    self.model = ModelFactory.create(
                        model_platform=ModelPlatformType.OPENAI,
                        model_type=ModelType.GPT_4O_MINI,
                    )
                
                # 创建ChatAgent
                self.agent = ChatAgent(
                    system_message="你是一个社交媒体帖子立场识别助手，能够根据帖子内容和话题识别帖子表达的立场。",
                    model=self.model
                )
                
                logger.info(f"✓ 成功初始化模型: {self.model_config['platform']}")
            except Exception as e:
                logger.error(f"❌ 模型初始化失败: {e}")
                raise
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            raise
    
    def get_user_recent_posts(self, user_id: int, limit: int = None) -> List[Dict]:
        """
        获取用户最近发布的帖子
        
        Args:
            user_id: 用户ID
            limit: 获取的帖子数量限制，如果为None则使用配置中的默认值
            
        Returns:
            帖子列表，每个帖子包含post_id, content, created_at等信息
        """
        limit = limit or config.stance.post_limit
        
        try:
            conn = self._get_db_connection()
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
            logger.debug(f"获取用户 {user_id} 的 {len(post_list)} 条帖子")
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
            conn = self._get_db_connection()
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
            logger.info(f"找到 {len(users)} 个发布过帖子的用户")
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
            conn = self._get_db_connection()
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
    
    async def detect_stance_for_text(self, text: str, topic: str = None, max_retries: int = None) -> Dict:
        """
        对单个文本进行立场检测
        
        Args:
            text: 要检测的文本
            topic: 检测的主题，如果为None则使用配置中的默认主题
            max_retries: 最大重试次数，如果为None则使用配置中的默认值
            
        Returns:
            立场检测结果，包含立场、置信度、理由等
        """
        topic = topic or config.stance.default_topic
        max_retries = max_retries or config.stance.max_retries
        
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
                        content=prompt
                    )
                )
                
                if response.msgs and len(response.msgs) > 0:
                    content = response.msgs[0].content.strip()
                    
                    # 尝试解析JSON
                    try:
                        result = json.loads(content)
                        
                        # 验证结果格式
                        if all(key in result for key in ["stance", "confidence", "reasoning"]):
                            result["text"] = text
                            result["topic"] = topic
                            result["detection_time"] = datetime.now().isoformat()
                            
                            logger.debug(f"立场检测成功: {result['stance']} (置信度: {result['confidence']})")
                            return result
                        else:
                            logger.warning(f"立场检测结果格式不完整: {result}")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析JSON响应: {content}")
                        
            except Exception as e:
                logger.warning(f"立场检测尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # 重试前等待
        
        # 所有重试都失败，返回默认结果
        logger.error(f"立场检测失败，返回默认结果")
        return {
            "stance": "中立",
            "confidence": 0.0,
            "reasoning": "检测失败，使用默认中立立场",
            "keywords": [],
            "text": text,
            "topic": topic,
            "detection_time": datetime.now().isoformat()
        }
    
    async def detect_stance_for_user(self, user_id: int, topic: str = None, post_limit: int = None) -> Dict:
        """
        检测指定用户的立场
        
        Args:
            user_id: 用户ID
            topic: 检测主题
            post_limit: 帖子数量限制
            
        Returns:
            用户立场检测结果
        """
        topic = topic or config.stance.default_topic
        post_limit = post_limit or config.stance.post_limit
        
        # 获取用户信息
        user_info = self.get_user_info(user_id)
        if not user_info:
            logger.warning(f"用户 {user_id} 不存在")
            return {"error": f"用户 {user_id} 不存在"}
        
        # 获取用户最近帖子
        posts = self.get_user_recent_posts(user_id, post_limit)
        if not posts:
            logger.warning(f"用户 {user_id} 没有发布过帖子")
            return {"error": f"用户 {user_id} 没有发布过帖子"}
        
        # 检测每个帖子的立场
        post_stances = []
        for post in posts:
            stance = await self.detect_stance_for_text(post['content'], topic)
            stance['post_id'] = post['post_id']
            stance['post_content'] = post['content']
            post_stances.append(stance)
        
        # 综合用户立场
        stance_counts = {}
        total_confidence = 0
        
        for stance in post_stances:
            stance_type = stance['stance']
            stance_counts[stance_type] = stance_counts.get(stance_type, 0) + 1
            total_confidence += float(stance['confidence'])
        
        # 确定主要立场
        main_stance = max(stance_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = total_confidence / len(post_stances) if post_stances else 0
        
        result = {
            "user_id": user_id,
            "user_name": user_info.get('user_name', ''),
            "name": user_info.get('name', ''),
            "topic": topic,
            "posts_analyzed": len(posts),
            "main_stance": main_stance,
            "stance_distribution": stance_counts,
            "average_confidence": avg_confidence,
            "post_stances": post_stances,
            "analysis_time": datetime.now().isoformat()
        }
        
        logger.info(f"用户 {user_id} 立场检测完成: {main_stance}")
        return result
    
    async def detect_stance_for_all_users(self, topic: str = None, post_limit: int = None) -> List[Dict]:
        """
        检测所有用户的立场
        
        Args:
            topic: 检测主题
            post_limit: 每个用户的帖子数量限制
            
        Returns:
            所有用户的立场检测结果列表
        """
        topic = topic or config.stance.default_topic
        post_limit = post_limit or config.stance.post_limit
        
        users = self.get_all_users_with_posts()
        if not users:
            logger.warning("没有找到发布过帖子的用户")
            return []
        
        logger.info(f"开始检测 {len(users)} 个用户的立场")
        
        # 并发检测所有用户
        semaphore = asyncio.Semaphore(config.stance.max_concurrent)
        
        async def process_user(user_id):
            async with semaphore:
                return await self.detect_stance_for_user(user_id, topic, post_limit)
        
        tasks = [process_user(user_id) for user_id in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉错误结果
        valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
        
        logger.info(f"立场检测完成，成功检测 {len(valid_results)} 个用户")
        return valid_results
    
    def save_stance_results(self, results: List[Dict], output_path: str):
        """
        保存立场检测结果
        
        Args:
            results: 检测结果列表
            output_path: 输出文件路径
        """
        try:
            # 保存为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存为CSV
            csv_path = output_path.replace('.json', '.csv')
            csv_data = []
            
            for result in results:
                if "error" in result:
                    continue
                    
                csv_data.append({
                    'user_id': result['user_id'],
                    'user_name': result['user_name'],
                    'name': result['name'],
                    'topic': result['topic'],
                    'posts_analyzed': result['posts_analyzed'],
                    'main_stance': result['main_stance'],
                    'average_confidence': result['average_confidence'],
                    'analysis_time': result['analysis_time']
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"结果已保存: {output_path}, {csv_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
    
    def generate_stance_summary(self, results: List[Dict]) -> Dict:
        """
        生成立场检测结果摘要
        
        Args:
            results: 检测结果列表
            
        Returns:
            摘要信息
        """
        if not results:
            return {"error": "没有检测结果"}
        
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "没有有效的检测结果"}
        
        # 统计信息
        total_users = len(valid_results)
        total_posts = sum(r['posts_analyzed'] for r in valid_results)
        avg_confidence = sum(r['average_confidence'] for r in valid_results) / total_users
        
        # 立场分布
        stance_distribution = {}
        for result in valid_results:
            stance = result['main_stance']
            stance_distribution[stance] = stance_distribution.get(stance, 0) + 1
        
        # 最常见立场
        most_common_stance = max(stance_distribution.items(), key=lambda x: x[1])[0]
        
        summary = {
            "total_users": total_users,
            "valid_users": len(valid_results),
            "total_posts_analyzed": total_posts,
            "average_confidence": avg_confidence,
            "most_common_stance": most_common_stance,
            "stance_distribution": stance_distribution,
            "summary_time": datetime.now().isoformat()
        }
        
        return summary 