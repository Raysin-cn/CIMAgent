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
        Initialize StanceDetector
        Args:
            db_path: database file path, use default if None
            model_config: model config, use default if None
        """
        self.db_path = db_path or config.database.path
        self.model_config = model_config or config.model.__dict__
        self.model = None
        self.agent = None
        # stance categories in English
        self.stance_categories = ["favor", "against", "none", "mixed"]
        logger.info(f"Initialize StanceDetector, db path: {self.db_path}")
    
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
                    system_message="You are a social media post stance recognition assistant, able to recognize the stance of a post based on its content and topic.",
                    model=self.model
                )
                
                logger.info(f"✓ Successfully initialized model: {self.model_config['platform']}")
            except Exception as e:
                logger.error(f"❌ Model initialization failed: {e}")
                raise
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
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
            logger.debug(f"Retrieved {len(post_list)} posts for user {user_id}")
            return post_list
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve posts for user {user_id}: {e}")
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
            logger.info(f"Found {len(users)} users who have posted")
            return users
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve user list: {e}")
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
            logger.error(f"❌ Failed to retrieve user {user_id} information: {e}")
            return None
    
    # Chain-of-Thought few-shot prompt template (English)
    cot_prompt_template = '''Q: What is the tweet's stance on the target?
The options are:
- against
- favor
- none

tweet: <I'm sick of celebrities who think being a well known actor makes them an authority on anything else. #robertredford #UN>
target: Liberal Values
reasoning: the author is implying that celebrities should not be seen as authorities on political issues, which is often associated with liberal values such as Robert Redford who is a climate change activist -> the author is against liberal values
stance: against

tweet: <I believe in a world where people are free to move and choose where they want to live>
target: Immigration
reasoning: the author is expressing a belief in a world with more freedom of movement -> the author is in favor of immigration
stance: favor

tweet: <I love the way the sun sets every day. #Nature #Beauty>
target: Taxes
reasoning: the author is in favor of nature and beauty -> the author is neutral towards taxes
stance: none

tweet: <If a woman chooses to pursue a career instead of staying at home, is she any less of a mother?>
target: Conservative Party
reasoning: the author is questioning traditional gender roles, which are often supported by the conservative party -> the author is against the conservative party
stance: against

tweet: <We need to make sure that mentally unstable people can't become killers #protect #US>
target: Gun Control
reasoning: the author is advocating for measures to prevent mentally unstable people from accessing guns -> the author is in favor of gun control
stance: favor

tweet: <There is no shortcut to success, there's only hard work and dedication #Success #SuccessMantra>
target: Open Borders
reasoning: the author is in favor of hard work and dedication -> the author is neutral towards open borders
stance: none

tweet: <{text}>
target: {target}
reasoning:
'''

    def _parse_reasoning_and_stance(self, content: str) -> tuple[str, str]:
        """
        Parse reasoning and stance from model output
        """
        import re
        reasoning = ""
        stance = ""
        reasoning_match = re.search(r"reasoning:(.*?)(?:stance:|$)", content, re.DOTALL | re.IGNORECASE)
        stance_match = re.search(r"stance:\s*([a-zA-Z]+)", content, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        if stance_match:
            stance = stance_match.group(1).strip().lower()
        # Map stance to standard English categories
        stance_map = {
            "favor": "favor",
            "support": "favor",
            "against": "against",
            "oppose": "against",
            "none": "none",
            "neutral": "none",
            "mixed": "mixed"
        }
        stance = stance_map.get(stance, "none")
        return reasoning, stance

    async def detect_stance_for_text(self, text: str, topic: str = None, max_retries: int = None, n: int = 3) -> Dict:
        """
        Stance detection for a single text, with CoT and self-consistency
        Args:
            text: text to detect
            topic: detection topic
            max_retries: max retry times
            n: sample times, default 3
        Returns:
            structured stance detection result
        """
        topic = topic or config.stance.default_topic
        max_retries = max_retries or config.stance.max_retries
        self._init_model()
        prompt = self.cot_prompt_template.format(text=text, target=topic)
        reasonings, stances = [], []
        for _ in range(n):
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
                        reasoning, stance = self._parse_reasoning_and_stance(content)
                        if stance:
                            reasonings.append(reasoning)
                            stances.append(stance)
                            break
                except Exception as e:
                    logger.warning(f"Stance detection sample failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
            else:
                reasonings.append("")
                stances.append("none")
        from collections import Counter
        stance_counter = Counter(stances)
        pred, count = stance_counter.most_common(1)[0]
        confidence = count / n
        return {
            "reasonings": reasonings,
            "stances": stances,
            "pred": pred,
            "confidence": confidence,
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
            logger.warning(f"User {user_id} does not exist")
            return {"error": f"User {user_id} does not exist"}
        
        # 获取用户最近帖子
        posts = self.get_user_recent_posts(user_id, post_limit)
        if not posts:
            logger.warning(f"User {user_id} has not posted")
            return {"error": f"User {user_id} has not posted"}
        
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
        
        logger.info(f"User {user_id} stance detection completed: {main_stance}")
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
            logger.warning("No users found who have posted")
            return []
        
        logger.info(f"Starting stance detection for {len(users)} users")
        
        # 并发检测所有用户
        semaphore = asyncio.Semaphore(config.stance.max_concurrent)
        
        async def process_user(user_id):
            async with semaphore:
                return await self.detect_stance_for_user(user_id, topic, post_limit)
        
        tasks = [process_user(user_id) for user_id in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉错误结果
        valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
        
        logger.info(f"Stance detection completed, successfully detected {len(valid_results)} users")
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
            
            logger.info(f"Results saved: {output_path}, {csv_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def generate_stance_summary(self, results: List[Dict]) -> Dict:
        """
        生成立场检测结果摘要
        
        Args:
            results: 检测结果列表
            
        Returns:
            摘要信息
        """
        if not results:
            return {"error": "No detection results"}
        
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid detection results"}
        
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