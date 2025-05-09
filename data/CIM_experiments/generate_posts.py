#!/usr/bin/env python3
import json
import pandas as pd
import argparse
from typing import Dict, List
import os
from datetime import datetime
import asyncio
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from tqdm import tqdm
import re


class PostGenerator:
    def __init__(self, model_path: str = "/data/model/Qwen3-14B", model_url: str = "http://localhost:21474/v1"):
        """初始化生成器，设置模型配置"""
        # 内容生成模型
        self.content_model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=model_path,
            url=model_url,
            model_config_dict={"max_tokens": 5000}
        )
        # 分析模型，增加max_tokens以确保完整输出
        self.analysis_model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=model_path,
            url=model_url,
            model_config_dict={
                "max_tokens": 1500,  # 增加token数量
                "temperature": 0.1  # 降低温度使输出更确定性
            }
        )

    async def generate_post_content(self, user_info: Dict, topic_info: Dict) -> str:
        """使用大模型生成帖子内容"""
        prompt = f"""As a social media user with the following characteristics:
Description: {user_info['description']}
Character: {user_info['user_char']}

Please write a post about this topic:
Title: {topic_info['title']}
Description: {topic_info['description']}
Keywords: {', '.join(topic_info['keywords'])}

Requirements:
1. Write in a natural, personal style matching the user's characteristics
2. Keep it under 280 characters
3. You can use emojis if appropriate
4. Express your genuine opinion based on your character
5. You can include placeholder URLs if relevant

Write only the post content, nothing else."""

        response = await self.content_model.arun([
            {"role": "system", "content": "You are a social media user creating a post."},
            {"role": "user", "content": prompt}
        ])
        
        content = response.choices[0].message.content.strip()
        
        # 移除可能的<think>标签及其内容
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()
        
        return content

    async def analyze_stance(self, post_content: str, topic_info: Dict) -> Dict:
        """分析帖子的立场和情感"""
        prompt = f"""Analyze this social media post and determine its stance and sentiment.

Post: "{post_content}"
Topic: {topic_info['title']}
Topic Description: {topic_info['description']}

Instructions:
1. First, analyze the overall stance (support/oppose/neutral) towards the topic
2. Then, analyze the emotional sentiment (positive/negative/neutral)
3. Return ONLY a JSON object in this exact format, nothing else:
{{
    "stance": "support/oppose/neutral",
    "sentiment": "positive/negative/neutral"
}}

IMPORTANT: DO NOT include any explanations, thoughts, or analysis. ONLY return the JSON object. NO additional text before or after the JSON.

Stance Criteria:
- 'support': Clearly agrees with or advocates for the proposal/change mentioned in the topic
- 'oppose': Clearly disagrees with or argues against the proposal/change
- 'neutral': Discusses both sides, asks questions, or presents balanced viewpoints

Sentiment Criteria:
- 'positive': Uses enthusiastic language, optimistic tone, positive emojis (✨🎉👍), or expresses hope/excitement
- 'negative': Uses concerned/worried language, negative emojis (❌😠👎), or expresses fear/anger
- 'neutral': Uses factual language, balanced tone, or focuses on objective discussion"""

        response = await self.analysis_model.arun([
            {
                "role": "system", 
                "content": "You are a post analyzer. You must ONLY return a JSON object with stance and sentiment. No other text allowed."
            },
            # {  # few-shot
            #     "role": "user", 
            #     "content": f"Analyze this post about remote work: 'Working from home has improved my life-work balance significantly! 🏡✨'\nReturn ONLY JSON."
            # },
            # {
            #     "role": "assistant",
            #     "content": '{"stance": "support", "sentiment": "positive"}'
            # },
            # {
            #     "role": "user", 
            #     "content": f"Analyze this post about political ads: 'Social media platforms should focus on facts, not political manipulation. Ban all political ads! 🚫'\nReturn ONLY JSON."
            # },
            # {
            #     "role": "assistant",
            #     "content": '{"stance": "support", "sentiment": "negative"}'
            # },
            {
                "role": "user", 
                "content": prompt
            }
        ])
        
        analysis = response.choices[0].message.content.strip()
        
        # 使用正则表达式提取stance和sentiment
        stance_pattern = r'"stance":\s*"(support|oppose|neutral)"'
        sentiment_pattern = r'"sentiment":\s*"(positive|negative|neutral)"'
        
        stance_match = re.search(stance_pattern, analysis)
        sentiment_match = re.search(sentiment_pattern, analysis)
        
        if not stance_match or not sentiment_match:
            print(f"Warning: Could not parse analysis properly. Raw response: {analysis}")
            
        stance = stance_match.group(1) if stance_match else "neutral"
        sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
        
        return {
            "stance": stance,
            "sentiment": sentiment
        }

async def generate_posts_for_topic(topic_id: str, topics: Dict, users_df: pd.DataFrame, 
                                 generator: PostGenerator) -> List[Dict]:
    """为指定话题生成所有用户的帖子"""

    
    topic_info = topics[topic_id]
    posts_data = []
    
    # 创建一个包含用户信息和任务的列表
    user_content_pairs = []
    content_tasks = []
    
    print("\n准备生成用户帖子...")
    for _, user in users_df.iterrows():
        user_info = {
            "description": user["description"],
            "user_char": user["user_char"],
            "user_id": user["user_id"]
        }
        # 创建生成任务
        task = generator.generate_post_content(user_info, topic_info)
        content_tasks.append(task)
        user_content_pairs.append({"user_info": user_info, "task": task})
    
    # 使用tqdm包装gather的结果
    print("\n开始生成用户帖子...")
    contents = []
    pbar = tqdm(total=len(content_tasks), desc="生成帖子")
    for coro in asyncio.as_completed(content_tasks):
        result = await coro
        contents.append(result)
        pbar.update(1)
    pbar.close()
    
    # 对contents按照原始任务顺序重新排序
    ordered_contents = []
    for pair in user_content_pairs:
        task = pair["task"]
        idx = content_tasks.index(task)
        ordered_contents.append(contents[idx])
    
    # 创建立场分析任务，保持相同顺序
    print("\n开始分析帖子立场...")
    analysis_tasks = [generator.analyze_stance(content, topic_info) for content in ordered_contents]
    
    # 使用tqdm包装立场分析任务
    analyses = []
    pbar = tqdm(total=len(analysis_tasks), desc="分析立场")
    for coro in asyncio.as_completed(analysis_tasks):
        result = await coro
        analyses.append(result)
        pbar.update(1)
    pbar.close()
    
    # 对analyses按照原始任务顺序重新排序
    ordered_analyses = []
    for task in analysis_tasks:
        idx = analysis_tasks.index(task)
        ordered_analyses.append(analyses[idx])
    
    # 按顺序创建最终的帖子数据
    for i, (user_content_pair, content, analysis) in enumerate(zip(user_content_pairs, ordered_contents, ordered_analyses)):
        user_info = user_content_pair["user_info"]
        # 从users_df中获取对应用户的完整信息
        user_row = users_df[users_df['user_id'] == user_info["user_id"]].iloc[0]
        
        post = {
            "user_id": user_info["user_id"],
            "name": user_row["name"],
            "username": user_row["username"],
            "content": content,
            "stance": analysis["stance"],
            "sentiment": analysis["sentiment"],
            "topic_id": topic_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        posts_data.append(post)
    
    return posts_data

def load_topics(topics_file: str) -> Dict:
    """加载话题配置"""
    with open(topics_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_available_topics(topics: Dict):
    """列出所有可用话题"""
    print("\nAvailable topics:")
    for topic_id, topic_info in topics.items():
        print(f"\n{topic_id}:")
        print(f"Title: {topic_info['title']}")
        print(f"Description: {topic_info['description']}")
        print("Keywords:", ", ".join(topic_info['keywords']))
        print("Related topics:", ", ".join(topic_info['related_topics']))
        print("-" * 80)

async def main():
    parser = argparse.ArgumentParser(description='Generate posts based on topics and user information')
    parser.add_argument('--topics_file', type=str, 
                      default='data/CIM_experiments/topics.json',
                      help='Path to the topics configuration file')
    parser.add_argument('--users_file', type=str, 
                      default='data/CIM_experiments/users_info.csv',
                      help='Path to the users information file')
    parser.add_argument('--output_dir', type=str, 
                      default='data/CIM_experiments/posts',
                      help='Directory to save generated posts')
    parser.add_argument('--list_topics', action='store_true',
                      help='List all available topics and exit')
    parser.add_argument('--topic_id', type=str, default='topic_3',
                      help='Specific topic ID to generate posts for')
    parser.add_argument('--model_path', type=str,
                      default='/data/model/Qwen3-14B',
                      help='Path to the language model')
    parser.add_argument('--model_url', type=str,
                      default='http://localhost:21474/v1',
                      help='URL for the model service')
    
    args = parser.parse_args()
    
    # 加载话题配置
    topics = load_topics(args.topics_file)
    
    # 如果只是列出话题
    if args.list_topics:
        list_available_topics(topics)
        return
    
    # 如果没有指定话题ID
    if not args.topic_id:
        print("Please specify a topic ID using --topic_id")
        list_available_topics(topics)
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载用户信息
    users_df = pd.read_csv(args.users_file)
    
    # 初始化生成器
    generator = PostGenerator(args.model_path, args.model_url)
    
    # 生成帖子
    posts_data = await generate_posts_for_topic(args.topic_id, topics, users_df, generator)
    
    # 保存生成的帖子
    output_file = os.path.join(args.output_dir, f'posts_{args.topic_id}.csv')
    posts_df = pd.DataFrame(posts_data)
    posts_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated {len(posts_df)} posts for topic {args.topic_id}")
    print(f"Posts saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())