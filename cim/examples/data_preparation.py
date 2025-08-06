import pandas as pd
import json
import os
import asyncio
from datetime import datetime
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import (
    ActionType, 
    ManualAction, 
    LLMAction,
    generate_twitter_agent_graph
)
from oasis.environment.env import OasisEnv

class OasisPostInjector:
    """将生成的帖子注入到Oasis社交网络模拟框架中"""
    
    def __init__(self, db_path: str = "./data/twitter_simulation.db"):
        self.db_path = db_path
        self.users_data = {}
        self.generated_posts = []
        
    def load_users_data(self, csv_path: str):
        """加载用户数据"""
        df = pd.read_csv(csv_path)
        self.users_data = df.set_index('user_id').to_dict('index')
        print(f"✓ 加载了 {len(self.users_data)} 个用户数据")
        return self.users_data
    
    def load_generated_posts(self, json_path: str):
        """加载生成的帖子数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.generated_posts = json.load(f)
        print(f"✓ 加载了 {len(self.generated_posts)} 条生成的帖子")
        return self.generated_posts
    
    def create_user_profile_csv(self, output_path: str):
        """创建符合oasis格式的用户档案CSV文件"""

        # 创建用户档案数据
        profile_data = []
        for user_id in list(self.users_data):
            user_info = self.users_data[user_id]
            
            # 构建用户档案
            profile = {
                'user_id': user_id,
                'name': user_info.get('name', f'user_{user_id}'),
                'username': user_info.get('username', f'user{user_id}'),
                'description': user_info.get('description', ''),
                'followers_count': user_info.get('followers_count', 0),
                'following_count': user_info.get('following_count', 0),
                'user_char': user_info.get('user_char', ''),
                'activity_level': user_info.get('activity_level', 'normal'),
                'created_at': user_info.get('created_at', '2009-01-01 00:00:00+00:00'),
                'following_list': user_info.get('following_list', '[]'),
                'following_agentid_list': user_info.get('following_agentid_list', '[]'),
                'previous_tweets': user_info.get('previous_tweets', '[]'),
                'tweets_id': user_info.get('tweets_id', '[]'),
                'activity_level_frequency': user_info.get('activity_level_frequency', '[1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,2,2,2,1,1,2,1,1,1]'),
                'followers_list': user_info.get('followers_list', '[]')
            }
            profile_data.append(profile)
        
        # 保存为CSV
        df = pd.DataFrame(profile_data)
        df.to_csv(output_path, index=False)
        print(f"✓ 创建了用户档案文件: {output_path}")
        return output_path
    
    def create_posts_actions(self, posts: list) -> list:
        """将生成的帖子转换为oasis动作列表"""
        actions = []
        
        for post in posts:
            # 创建发帖动作
            action = {
                'user_id': post['user_id'],
                'action_type': ActionType.CREATE_POST,
                'content': post['content'],
                'timestamp': post.get('timestamp', 0)
            }
            actions.append(action)
        
        return actions
    
    async def inject_anonymous_posts(self, env:OasisEnv, posts: list):
        """直接向平台注入匿名帖子，不通过代理系统"""
        print("正在注入匿名帖子...")
        
        # 为匿名帖子创建特殊的用户ID（0号为匿名智能体避免与正常用户冲突）
        anonymous_user_id = 0
        current_time = 0
        
        # 注入所有匿名帖子
        injected_count = 0
        for i, post in enumerate(posts):
            try:
                # 直接向数据库插入帖子
                post_insert_query = (
                    "INSERT INTO post (user_id, content, created_at, num_likes, "
                    "num_dislikes, num_shares) VALUES (?, ?, ?, ?, ?, ?)")
                
                env.platform.pl_utils._execute_db_command(
                    post_insert_query, 
                    (anonymous_user_id, post['content'], current_time, 0, 0, 0),
                    commit=True
                )
                
                post_id = env.platform.db_cursor.lastrowid
                
                # 记录到trace表
                action_info = {"content": post['content'], "post_id": post_id}
                env.platform.pl_utils._record_trace(
                    anonymous_user_id, 
                    ActionType.CREATE_POST.value,
                    action_info, 
                    current_time
                )
                
                injected_count += 1
                if (i + 1) % 10 == 0:
                    print(f"✓ 已注入 {i + 1} 条匿名帖子...")
                    
            except Exception as e:
                print(f"❌ 注入第 {i + 1} 条匿名帖子时出错: {e}")
        
        print(f"✓ 成功注入了 {injected_count} 条匿名帖子")
        return injected_count
    
    async def run_simulation_with_posts(self, 
                                      profile_path: str,
                                      posts: list,
                                      num_steps: int = 5):
        """运行包含生成帖子的社交网络模拟"""
        
        # 创建模型
        # model = ModelFactory.create(
        #     model_platform=ModelPlatformType.OPENAI,
        #     model_type=ModelType.GPT_4O,
        # )

        model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="/data/model/Qwen3-14B",
            url="http://localhost:12345/v1"
        )
        
        # 定义可用动作
        available_actions = ActionType.get_default_twitter_actions()
        
        # 生成代理图
        agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=available_actions,
        )
        
        # 删除旧数据库
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        # 创建环境
        import oasis
        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=self.db_path,
        )
        
        # 重置环境
        await env.reset()
        
        # 注入匿名帖子
        await self.inject_anonymous_posts(env, posts)
        
        # 让其他代理进行互动
        print("让其他代理进行互动...")
        for step in range(num_steps):
            try:
                # 选择一些代理进行LLM驱动的动作
                llm_actions = {}
                agent_count = 0
                
                for agent_id, agent in env.agent_graph.get_agents()[1:]:  #匿名智能体不执行动作
                    llm_actions[agent] = LLMAction()
                
                await env.step(llm_actions)
                print(f"✓ 步骤 {step + 1}: {len(llm_actions)} 个代理进行了互动")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ 步骤 {step + 1} 出错: {e}")
        
        # 关闭环境
        await env.close()
        print("✓ 模拟完成")
        
        return env