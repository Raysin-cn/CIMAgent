import pandas as pd
import json
from typing import List, Dict
import asyncio
from datetime import datetime
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import re

class PostGenerator:
    def __init__(self, users_csv_path: str):
        """
        初始化帖子生成器
        
        Args:
            users_csv_path: 用户数据CSV文件路径
        """
        self.users_csv_path = users_csv_path
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="/data/model/Qwen3-14B",
            url="http://localhost:21474/v1",
            model_config_dict={"max_tokens": 32000}
        )
        self.df = None
        self.user_descriptions = {}
        
    def load_data(self):
        """加载用户CSV数据"""
        self.df = pd.read_csv(self.users_csv_path)
        # 创建用户描述字典
        for _, row in self.df.iterrows():
            self.user_descriptions[str(row["user_id"])] = {
                "description": row["description"],
                "user_char": row["user_char"]
            }
            
    def parse_llm_response(self, response_text: str) -> Dict:
        """
        解析LLM返回的文本，提取帖子内容、情感和立场
        
        Args:
            response_text: LLM返回的文本
            
        Returns:
            包含帖子信息的字典
        """
        try:
            # 尝试直接解析JSON
            return json.loads(response_text)
        except:
            # 如果JSON解析失败，使用正则表达式提取信息
            content_match = re.search(r'content["\s:]+([^"]+)', response_text)
            sentiment_match = re.search(r'sentiment["\s:]+([^"]+)', response_text)
            stance_match = re.search(r'stance["\s:]+([^"]+)', response_text)
            
            content = content_match.group(1) if content_match else "无法生成内容"
            sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
            stance = stance_match.group(1) if stance_match else "neutral"
            
            return {
                "content": content,
                "sentiment": sentiment,
                "stance": stance
            }
            
    async def generate_post(self, user_id: str, topic: str) -> Dict:
        """
        为指定用户生成关于特定话题的帖子
        
        Args:
            user_id: 用户ID
            topic: 话题内容
            
        Returns:
            包含帖子信息的字典
        """
        user_desc = self.user_descriptions.get(user_id, {})
        if not user_desc:
            return None
            
        try:
            messages = [
                {"role": "system", "content": """你是一个社交媒体内容生成专家。请根据用户的特征和描述，生成一条关于特定话题的帖子。
                要求：
                1. 内容要符合用户的性格特征和兴趣
                2. 语言风格要自然，像真实用户
                3. 可以包含表情符号
                4. 长度控制在280字符以内
                5. 可以包含URL占位符
                6. 要有观点和态度
                
                请严格按照以下JSON格式返回，不要添加任何其他内容：
                {
                    "content": "帖子内容",
                    "sentiment": "情感倾向(positive/negative/neutral)",
                    "stance": "立场(for/against/neutral)"
                }"""},
                {"role": "user", "content": f"""用户信息：
                描述：{user_desc['description']}
                特征：{user_desc['user_char']}
                
                话题：{topic}
                
                请生成一条符合该用户特征的帖子。"""}
            ]
            
            response = await self.model.arun(messages)
            response_text = response.choices[0].message.content.strip()
            
            # 解析响应
            post_data = self.parse_llm_response(response_text)
            
            return {
                "root_user": int(user_id),  # 确保是整数类型
                "source_tweet": post_data["content"],
                "sentiment": post_data["sentiment"],
                "stance": post_data["stance"],
                "topic": topic,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating post for user {user_id}: {e}")
            print(f"Response text: {response_text if 'response_text' in locals() else 'No response'}")
            return None
            
    async def generate_posts_for_topic(self, topic: str, posts_per_user: int = 1) -> List[Dict]:
        """
        为所有用户生成关于特定话题的帖子
        
        Args:
            topic: 话题内容
            posts_per_user: 每个用户生成的帖子数量
            
        Returns:
            帖子列表
        """
        if self.df is None:
            self.load_data()
            
        posts = []
        tasks = []
        
        # 为每个用户创建生成任务
        for user_id in self.df["user_id"]:
            for _ in range(posts_per_user):
                tasks.append(self.generate_post(str(user_id), topic))
                
        # 并发执行所有任务
        print(f"开始为{topic}生成帖子")
        results = await asyncio.gather(*tasks)
        print("帖子生成完成")
        
        # 过滤掉生成失败的结果
        posts = [post for post in results if post is not None]
        
        return posts
        
    async def save_posts(self, posts: List[Dict], output_path: str):
        """
        保存生成的帖子到CSV文件
        
        Args:
            posts: 帖子列表
            output_path: 输出文件路径
        """
        df = pd.DataFrame(posts)
        # 确保列的顺序与twitter_simulation_CIM.py中的格式一致
        df = df[["root_user", "source_tweet", "sentiment", "stance", "topic", "created_at"]]
        df.to_csv(output_path, index=False)
        print(f"帖子已保存到: {output_path}")

async def main():
    # 初始化生成器
    generator = PostGenerator("./data/twitter_dataset_CIM/processed_users.csv")
    
    # 设置话题
    topic = "现在出现的LLM智能体是否具备了可以模拟人类社交网络媒体的能力？"
    
    # 生成帖子
    posts = await generator.generate_posts_for_topic(topic, posts_per_user=1)
    
    # 保存结果
    output_path = f"./data/twitter_dataset_CIM/posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    await generator.save_posts(posts, output_path)

if __name__ == "__main__":
    asyncio.run(main())