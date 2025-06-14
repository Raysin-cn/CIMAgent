import asyncio
import json
import pandas as pd
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()


from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.messages import BaseMessage

import oasis
from oasis import ActionType, ManualAction


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


class PostGenerator:
    """帖子生成器"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 1):
        """
        初始化帖子生成器
        
        Args:
            model_name: 使用的模型名称
            temperature: 模型温度参数，控制生成多样性
        """
        # 创建模型
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
        )
        
        # 创建ChatAgent
        self.agent = ChatAgent(
            system_message="你是一个社交媒体帖子生成助手，能够根据用户档案和话题生成个性化的帖子内容。",
            model=model
        )
        
        self.temperature = temperature
        self.users_data = {}
        self.topics_data = {}
        
    def load_users_data(self, csv_path: str) -> Dict[str, UserProfile]:
        """加载用户数据"""
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
        return users
    
    def load_topics_data(self, json_path: str) -> Dict[str, Topic]:
        """加载话题数据"""
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
        return topics
    
    def _create_user_prompt(self, user: UserProfile, topic: Topic) -> str:
        """为用户和话题创建提示词"""
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
3. 帖子长度适中（100-280字符）
4. 内容要有观点和态度
5. 可以适当使用表情符号和标签
6. 要自然、真实，避免过于官方化的语言

请直接返回帖子内容，不要包含其他说明文字。
"""
        return prompt
    
    async def generate_post_for_user(self, user_id: str, topic_id: str) -> Optional[GeneratedPost]:
        """为指定用户生成关于指定话题的帖子"""
        if user_id not in self.users_data:
            print(f"用户 {user_id} 不存在")
            return None
            
        if topic_id not in self.topics_data:
            print(f"话题 {topic_id} 不存在")
            return None
        
        user = self.users_data[user_id]
        topic = self.topics_data[topic_id]
        
        # 创建提示词
        prompt = self._create_user_prompt(user, topic)
        
        try:
            # 调用大模型生成内容
            response = await self.agent.astep(
                BaseMessage.make_user_message(
                role_name="User",
                content=(prompt))
            )
            
            if response.msgs and len(response.msgs) > 0:
                content = response.msgs[0].content.strip()
                # 创建帖子对象
                post = GeneratedPost(
                    user_id="None",
                    username="None",
                    content=content,
                    topic=topic.title,
                    timestamp=0
                )
                
                return post
            else:
                print(f"生成帖子失败：没有返回内容")
                return None
            
        except Exception as e:
            print(f"生成帖子时出错: {e}")
            return None
    
    async def generate_multiple_posts(self, 
                                    user_ids: List[str], 
                                    topic_id: str, 
                                    num_posts: int) -> List[GeneratedPost]:
        """生成多个帖子"""
        # 如果用户数量少于需要的帖子数量，重复使用用户
        if len(user_ids) < num_posts:
            user_ids = user_ids * (num_posts // len(user_ids) + 1)
        
        # 随机选择用户生成帖子
        selected_users = random.sample(user_ids, min(num_posts, len(user_ids)))
        
        post_task = []
        for user_id in selected_users:
            post_task.append(self.generate_post_for_user(user_id, topic_id))
            
        results = await asyncio.gather(*post_task)
        
        return results
    
    def save_posts_to_json(self, posts: List[GeneratedPost], output_path: str):
        """将生成的帖子保存到JSON文件"""
        posts_data = []
        for post in posts:
            posts_data.append({
                "user_id": post.user_id,
                "username": post.username,
                "content": post.content,
                "topic": post.topic,
                "timestamp": post.timestamp
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(posts)} 条帖子到 {output_path}")
    
    def save_posts_to_csv(self, posts: List[GeneratedPost], output_path: str):
        """将生成的帖子保存到CSV文件"""
        posts_data = []
        for post in posts:
            posts_data.append({
                "user_id": post.user_id,
                "username": post.username,
                "content": post.content,
                "topic": post.topic,
                "timestamp": post.timestamp
            })
        
        df = pd.DataFrame(posts_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"已保存 {len(posts)} 条帖子到 {output_path}")


async def main():
    """主函数示例"""
    # 初始化生成器
    generator = PostGenerator(temperature=0.9)  # 高温度以获得更多样性
    
    # 加载数据
    users = generator.load_users_data("data/users_info.csv")
    topics = generator.load_topics_data("data/topics.json")
    
    print(f"加载了 {len(users)} 个用户")
    print(f"加载了 {len(topics)} 个话题")
    
    # 选择一些用户和话题
    user_ids = list(users.keys())
    topic_id = "topic_3"  # 第一个话题
    
    # 生成帖子
    posts = await generator.generate_multiple_posts(
        user_ids=user_ids,
        topic_id=topic_id,
        num_posts=50
    )
    
    # 保存结果
    generator.save_posts_to_json(posts, "data/generated_posts.json")
    generator.save_posts_to_csv(posts, "data/generated_posts.csv")
    
    # 打印示例
    for post in posts[:3]:
        print(f"\n用户: {post.username}")
        print(f"内容: {post.content}")
        print(f"话题: {post.topic}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
