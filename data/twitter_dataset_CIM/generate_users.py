import pandas as pd
import json
from typing import List, Dict, Tuple
import asyncio
from datetime import datetime
import os
from collections import defaultdict
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import random
import matplotlib.pyplot as plt
import numpy as np

def visualize_adj_matrix(adj_matrix, title="Adj_Maxtrix Visilization"):
    """
    可视化邻接矩阵
    
    Args:
        adj_matrix: 邻接矩阵
        title: 图表标题
    """
    # 确保邻接矩阵是numpy数组且类型为float
    adj_matrix = np.array(adj_matrix, dtype=float)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(adj_matrix, cmap='Blues')
    plt.colorbar()
    plt.title(title)
    plt.savefig('./data/twitter_dataset_CIM/adj_matrix.png')
    plt.close()

class UserDataProcessor:
    def __init__(self, csv_path: str):
        """
        初始化用户数据处理器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="/data/model/Qwen3-14B",
            url="http://localhost:21474/v1",
            model_config_dict={"max_tokens": 32000}
        )
        self.df = None
        self.user_descriptions = {}
        
    def load_data(self):
        """加载CSV数据"""
        self.df = pd.read_csv(self.csv_path)
        # 创建用户描述字典
        for _, row in self.df.iterrows():
            self.user_descriptions[str(row["user_id"])] = {
                "description": row["description"],
                "user_char": row["user_char"]
            }
        
    async def enhance_user_description(self, description: str) -> str:
        """
        使用LLM增强用户描述
        
        Args:
            description: 原始用户描述
            
        Returns:
            增强后的用户描述
        """
        if not description or pd.isna(description):
            return "No description available"
            
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的社交媒体用户分析专家。请根据用户的原始描述，生成一个更详细、更专业的用户画像描述。保持原始信息的同时，添加更多专业见解。"},
                {"role": "user", "content": f"请分析并增强以下用户描述：{description}"}
            ]
            response = await self.model.arun(messages)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error enhancing description: {e}")
            return description

    async def analyze_potential_followers(self, user_id: str, candidate_users: List[Dict]) -> List[Tuple[str, float, float]]:
        """
        分析用户可能关注和被关注的用户
        
        Args:
            user_id: 当前用户ID
            candidate_users: 候选用户列表，每个元素包含id和描述信息
            
        Returns:
            列表，每个元素为(目标用户ID, 关注可能性分数, 被关注可能性分数)
        """
        user_desc = self.user_descriptions.get(user_id, {})
        if not user_desc:
            return []
            
        try:
            # 构建候选用户描述
            candidates_desc = "\n".join([
                f"用户ID: {c['id']}\n描述: {c['description']}\n特征: {c['user_char']}"
                for c in candidate_users
            ])
            
            messages = [
                {"role": "system", "content": """你是一个社交媒体关系分析专家。请分析当前用户与候选用户之间的双向关系。
                对于每个候选用户，请分别判断：
                1. 当前用户是否可能关注该候选用户
                2. 该候选用户是否可能关注当前用户
                
                请考虑以下因素：
                1. 共同的兴趣爱好
                2. 专业领域
                3. 生活方式
                4. 价值观
                5. 社交圈子
                6. 影响力差异
                7. 内容相关性
                
                请严格按照以下格式回复，每个候选用户一行：
                用户ID: xxx, 关注可能性: 0.8, 被关注可能性: 0.3"""},
                {"role": "user", "content": f"""当前用户描述：
                {user_desc['description']}
                {user_desc['user_char']}
                
                候选用户列表：
                {candidates_desc}
                
                请分析当前用户与每个候选用户之间的双向关注关系，并给出可能性分数。"""}
            ]
            
            response = await self.model.arun(messages)
            
            # 解析响应
            results = []
            content = response.choices[0].message.content
            print(content)
            for line in content.strip().split('\n'):
                try:
                    # 解析每行的结果
                    parts = line.split(',')
                    target_id = parts[0].split(':')[1].strip()
                    follow_score = float(parts[1].split(':')[1].strip())
                    followed_score = float(parts[2].split(':')[1].strip())
                    results.append((target_id, follow_score, followed_score))
                except:
                    continue
                    
            return results
                
        except Exception as e:
            print(f"Error analyzing potential followers: {e}")
            return []
            
    async def generate_relationships(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        生成用户之间的关注关系
        
        Returns:
            (关注关系字典, 被关注关系字典)
        """
        following_relationships = defaultdict(list)
        followers_relationships = defaultdict(list)
        threshold = 0.6  # 关注可能性阈值
        
        # 为每个用户选择约10%的其他用户作为候选
        sample_size = max(1, int(len(self.df) * 0.1))
        relation_tasks = []
        
        for _, row in self.df.iterrows():
            user_id = str(row["user_id"])
            
            # 随机选择候选用户
            other_users = self.df[self.df['user_id'] != user_id].sample(n=sample_size)
            
            # 构建候选用户列表
            candidate_users = []
            for _, target_row in other_users.iterrows():
                target_id = str(target_row["user_id"])
                candidate_users.append({
                    "id": target_id,
                    "description": self.user_descriptions[target_id]["description"],
                    "user_char": self.user_descriptions[target_id]["user_char"]
                })
            
            # 分析潜在关注关系
            relation_tasks.append(self.analyze_potential_followers(user_id, candidate_users))
            
        print("开始分析潜在关注关系")
        results = await asyncio.gather(*relation_tasks)
        print("分析潜在关注关系完成")
        
        # 处理结果
        for user_id, relations in zip(self.df["user_id"], results):
            for target_id, follow_score, followed_score in relations:
                # 处理关注关系
                if follow_score >= threshold:
                    following_relationships[str(user_id)].append(target_id)
                
                # 处理被关注关系
                if followed_score >= threshold:
                    followers_relationships[target_id].append(str(user_id))
                        
        return dict(following_relationships), dict(followers_relationships)
            
    async def process_user_data(self) -> List[Dict]:
        """
        处理用户数据并返回增强后的用户列表
        
        Returns:
            处理后的用户数据列表
        """
        if self.df is None:
            self.load_data()
            
        # 生成关注关系
        following_relationships, followers_relationships = await self.generate_relationships()
        
        # 创建增强后的用户数据
        enhance_user_data = self.df.copy()
        
        # 初始化关注列表和粉丝列表列
        enhance_user_data['following_list'] = [[] for _ in range(len(enhance_user_data))]
        enhance_user_data['followers_list'] = [[] for _ in range(len(enhance_user_data))]
        
        # 更新关注关系
        for user_id, following_list in following_relationships.items():
            mask = enhance_user_data['user_id'] == int(user_id)
            if mask.any():
                idx = enhance_user_data[mask].index[0]
                enhance_user_data.at[idx, 'following_list'] = following_list
                enhance_user_data.at[idx, 'following_agentid_list'] = [enhance_user_data[enhance_user_data['user_id'] == int(following_id)].index[0] for following_id in following_list]
        
        # 更新粉丝关系
        for user_id, followers_list in followers_relationships.items():
            mask = enhance_user_data['user_id'] == int(user_id)
            if mask.any():
                idx = enhance_user_data[mask].index[0]
                enhance_user_data.at[idx, 'followers_list'] = followers_list
        
        # 生成邻接矩阵并可视化
        n_users = len(enhance_user_data)
        adj_matrix = np.zeros((n_users, n_users))
        for i, row in enhance_user_data.iterrows():
            for following_id in row['following_list']:
                j = enhance_user_data[enhance_user_data['user_id'] == int(following_id)].index
                if len(j) > 0:
                    adj_matrix[i, j[0]] = 1
        
        visualize_adj_matrix(adj_matrix)
        
        return enhance_user_data
    
    async def save_processed_data(self, output_path: str):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出文件路径
        """
        enhance_user_data = await self.process_user_data()
        # 将follower_list和following_list从字符串列表转换为整数列表
        enhance_user_data['followers_list'] = enhance_user_data['followers_list'].apply(
            lambda x: [int(i) for i in x]
        )
        enhance_user_data['following_list'] = enhance_user_data['following_list'].apply(
            lambda x: [int(i) for i in x]
        )
        enhance_user_data.to_csv(output_path, index=False)

async def main():
    # 初始化处理器
    processor = UserDataProcessor(
        csv_path="data/twitter_dataset/user_all_id_time.csv"
    )
    
    # 处理数据并保存
    output_path = "data/twitter_dataset_CIM/processed_users.csv"
    await processor.save_processed_data(output_path)
    print(f"处理完成，数据已保存到: {output_path}")

def following_list_to_following_agentid_list(csv_path: str):
    user_data = pd.read_csv(csv_path)
    for index, row in user_data.iterrows():
        if row['following_list'] is not None and isinstance(row['following_list'], str):
            # 将字符串形式的列表转换为实际的列表
            following_list = eval(row['following_list'])
            following_agentid_list = []
            for following_id in following_list:
                # 检查是否存在匹配的用户ID
                matched_users = user_data[user_data['user_id'] == int(following_id)]
                if not matched_users.empty:
                    following_agentid_list.append(int(matched_users.index[0]))
            user_data.at[index, 'following_agentid_list'] = following_agentid_list
    user_data.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # asyncio.run(main())
    following_list_to_following_agentid_list("data/twitter_dataset_CIM/processed_users.csv")
