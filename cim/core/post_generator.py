"""
帖子生成器模块

提供社交媒体帖子生成功能，支持：
- 基于用户档案的个性化帖子生成
- 多话题支持
- 批量生成
- 内容质量控制
"""

import asyncio
import json
import pandas as pd
import random
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from ..config import config


# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """用户档案数据类"""
    user_id: str
    name: str
    username: str
    description: str
    followers_count: int
    following_count: int
    user_char: str
    activity_level: str
    activity_level_frequency: List[int]


@dataclass
class Topic:
    """话题数据类"""
    title: str
    description: str
    keywords: List[str]
    related_topics: List[str]


@dataclass
class GeneratedPost:
    """生成的帖子数据类"""
    user_id: str
    username: str
    content: str
    topic: str
    timestamp: str
    generation_time: str = None
    
    def __post_init__(self):
        if self.generation_time is None:
            self.generation_time = datetime.now().isoformat()


class PostGenerator:
    """帖子生成器"""
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        初始化帖子生成器
        
        Args:
            model_config: 模型配置，如果为None则使用配置中的默认配置
        """
        self.model_config = model_config or config.model.__dict__
        self.model = None
        self.agent = None
        self.users_data = {}
        self.topics_data = {}
        
        logger.info("初始化帖子生成器")
    
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
                    system_message="你是一个社交媒体帖子生成助手，能够根据用户档案和话题生成个性化的帖子内容。",
                    model=self.model
                )
                
                logger.info(f"✓ 成功初始化模型: {self.model_config['platform']}")
            except Exception as e:
                logger.error(f"❌ 模型初始化失败: {e}")
                raise
    
    def load_users_data(self, csv_path: str = None) -> Dict[str, UserProfile]:
        """
        加载用户数据
        
        Args:
            csv_path: 用户数据CSV文件路径，如果为None则使用配置中的默认路径
            
        Returns:
            用户档案字典
        """
        csv_path = csv_path or config.post_generation.users_file
        
        try:
            df = pd.read_csv(csv_path)
            users = {}
            
            for _, row in df.iterrows():
                # 解析活动频率数据
                activity_frequency = eval(row['activity_level_frequency']) if pd.notna(row['activity_level_frequency']) else []
                activity_levels = eval(row['activity_level']) if pd.notna(row['activity_level']) else []
                
                user_profile = UserProfile(
                    user_id=str(row['user_id']),
                    name=row['name'] if pd.notna(row['name']) else "",
                    username=row['username'] if pd.notna(row['username']) else "",
                    description=row['description'] if pd.notna(row['description']) else "",
                    followers_count=int(row['followers_count']) if pd.notna(row['followers_count']) else 0,
                    following_count=int(row['following_count']) if pd.notna(row['following_count']) else 0,
                    user_char=row['user_char'] if pd.notna(row['user_char']) else "",
                    activity_level=activity_levels[-1] if activity_levels else "normal",
                    activity_level_frequency=activity_frequency
                )
                users[user_profile.user_id] = user_profile
                
            self.users_data = users
            logger.info(f"✓ 加载了 {len(users)} 个用户档案")
            return users
            
        except Exception as e:
            logger.error(f"❌ 加载用户数据失败: {e}")
            return {}
    
    def load_topics_data(self, json_path: str = None) -> Dict[str, Topic]:
        """
        加载话题数据
        
        Args:
            json_path: 话题数据JSON文件路径，如果为None则使用配置中的默认路径
            
        Returns:
            话题字典
        """
        json_path = json_path or config.post_generation.topics_file
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                topics_raw = json.load(f)
            
            topics = {}
            for topic_id, topic_data in topics_raw.items():
                topic = Topic(
                    title=topic_data['title'],
                    description=topic_data['description'],
                    keywords=topic_data['keywords'],
                    related_topics=topic_data['related_topics']
                )
                topics[topic_id] = topic
                
            self.topics_data = topics
            logger.info(f"✓ 加载了 {len(topics)} 个话题")
            return topics
            
        except Exception as e:
            logger.error(f"❌ 加载话题数据失败: {e}")
            return {}
    
    def _create_user_prompt(self, user: UserProfile, topic: Topic) -> str:
        """
        为用户和话题创建提示词
        
        Args:
            user: 用户档案
            topic: 话题信息
            
        Returns:
            生成的提示词
        """
        prompt = f"""
你是一个社交媒体用户，需要基于给定的用户档案和话题生成一条个性化的帖子。

用户档案：
- 用户名：{user.username}
- 真实姓名：{user.name}
- 个人描述：{user.description}
- 用户特征：{user.user_char}
- 粉丝数：{user.followers_count}
- 关注数：{user.following_count}
- 活跃度：{user.activity_level}

话题信息：
- 标题：{topic.title}
- 描述：{topic.description}
- 关键词：{', '.join(topic.keywords)}
- 相关话题：{', '.join(topic.related_topics)}

请生成一条关于这个话题的帖子，要求：
1. 内容要符合用户的个人特征和描述
2. 语言风格要符合用户的活跃度水平
3. 帖子长度适中（{config.post_generation.min_length}-{config.post_generation.max_length}字符）
4. 内容要有观点和态度
5. 要自然、真实，避免过于官方化的语言

请直接返回帖子内容，不要包含其他说明文字。
"""
        return prompt
    
    async def generate_post_for_user(self, user_id: str, topic_id: str) -> Optional[GeneratedPost]:
        """
        为指定用户生成关于指定话题的帖子
        
        Args:
            user_id: 用户ID
            topic_id: 话题ID
            
        Returns:
            生成的帖子对象
        """
        if user_id not in self.users_data:
            logger.warning(f"用户 {user_id} 不存在")
            return None
            
        if topic_id not in self.topics_data:
            logger.warning(f"话题 {topic_id} 不存在")
            return None
        
        user = self.users_data[user_id]
        topic = self.topics_data[topic_id]
        
        # 初始化模型
        self._init_model()
        
        # 创建提示词
        prompt = self._create_user_prompt(user, topic)
        
        try:
            # 调用大模型生成内容
            response = await self.agent.astep(
                BaseMessage.make_user_message(
                    role_name="User",
                    content=prompt
                )
            )
            
            if response.msgs and len(response.msgs) > 0:
                content = response.msgs[0].content.strip()
                
                # 创建帖子对象
                post = GeneratedPost(
                    user_id=user_id,
                    username=user.username,
                    content=content,
                    topic=topic.title,
                    timestamp=str(datetime.now().timestamp())
                )
                
                logger.debug(f"为用户 {user_id} 生成帖子成功")
                return post
            else:
                logger.warning(f"为用户 {user_id} 生成帖子失败：没有返回内容")
                return None
            
        except Exception as e:
            logger.error(f"为用户 {user_id} 生成帖子时出错: {e}")
            return None
    
    async def generate_multiple_posts(self, 
                                    user_ids: List[str], 
                                    topic_id: str, 
                                    num_posts: int = 1) -> List[GeneratedPost]:
        """
        为多个用户生成帖子
        
        Args:
            user_ids: 用户ID列表
            topic_id: 话题ID
            num_posts: 每个用户生成的帖子数量
            
        Returns:
            生成的帖子列表
        """
        all_posts = []
        
        for user_id in user_ids:
            for _ in range(num_posts):
                post = await self.generate_post_for_user(user_id, topic_id)
                if post:
                    all_posts.append(post)
        
        logger.info(f"为 {len(user_ids)} 个用户生成了 {len(all_posts)} 条帖子")
        return all_posts
    
    def save_posts_to_json(self, posts: List[GeneratedPost], output_path: str):
        """
        将生成的帖子保存为JSON格式
        
        Args:
            posts: 帖子列表
            output_path: 输出文件路径
        """
        try:
            # 转换为字典格式
            posts_data = []
            for post in posts:
                post_dict = {
                    'user_id': post.user_id,
                    'username': post.username,
                    'content': post.content,
                    'topic': post.topic,
                    'timestamp': post.timestamp,
                    'generation_time': post.generation_time
                }
                posts_data.append(post_dict)
            
            # 保存为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(posts_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ 帖子已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存帖子失败: {e}")
    
    def save_posts_to_csv(self, posts: List[GeneratedPost], output_path: str):
        """
        将生成的帖子保存为CSV格式
        
        Args:
            posts: 帖子列表
            output_path: 输出文件路径
        """
        try:
            # 转换为DataFrame
            data = []
            for post in posts:
                data.append({
                    'user_id': post.user_id,
                    'username': post.username,
                    'content': post.content,
                    'topic': post.topic,
                    'timestamp': post.timestamp,
                    'generation_time': post.generation_time
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"✓ 帖子已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存帖子失败: {e}")
    
    def get_available_users(self) -> List[str]:
        """获取可用的用户ID列表"""
        return list(self.users_data.keys())
    
    def get_available_topics(self) -> List[str]:
        """获取可用的话题ID列表"""
        return list(self.topics_data.keys())
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取指定用户的档案"""
        return self.users_data.get(user_id)
    
    def get_topic_info(self, topic_id: str) -> Optional[Topic]:
        """获取指定话题的信息"""
        return self.topics_data.get(topic_id) 