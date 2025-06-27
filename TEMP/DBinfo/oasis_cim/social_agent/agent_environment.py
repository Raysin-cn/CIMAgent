# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")
    
    env_template = Template(
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")
    
    # NOTE 隐藏智能体获取环境信息
    specific_agent_env_template = Template(
        "Here is the information of the agent you are interested in:\n"
        "Agent Information:\n"
        "ID: $agent_id\n"
        "Bio: $bio\n"
        "Recent Activities:\n"
        "Posts: $recent_posts\n"
        "Comments: $recent_comments\n"
        "Interactions: $recent_interactions"
    )
    


    def __init__(self, action: SocialAction):
        self.action = action

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = False,
        include_follows: bool = False,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""


        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
        )

    async def get_specific_agent_env(self, agent_id: int) -> str:
        r"""Get the environment information of a specific agent.
        
        Args:
            agent_id (int): The ID of the agent to get information about.
            
        Returns:
            str: A formatted string containing the agent's information including:
                - Bio
                - Recent posts
                - Recent comments
                - Recent interactions (likes, reposts, etc.)
        """
        # 搜索用户信息
        user_info = await self.action.search_user(f'{agent_id}')
        if len(user_info.get("users", [])) != 0:
            for info in user_info.get("users", []):
                if info.get("user_id") == agent_id:
                    bio = info.get("bio", "No bio available")
                    break
            else:
                bio = "No bio available"
        else:
            bio = "No bio available"
        # 获取该用户的最近帖子
        search_result = await self.action.search_posts(f"{agent_id}")
        recent_posts = []
        if len(search_result.get("posts", [])) != 0:
            for posts in search_result.get("posts"):
                if posts['user_id'] == agent_id:
                    recent_posts.append(posts)
        if len(recent_posts) > 3:
            recent_posts = recent_posts[-3:]

        # 由于没有直接的评论搜索API，我们可以从帖子中提取评论
        recent_comments = "Comments information not available"  # 这部分需要额外的API支持
        
        # 获取用户的最近交互
        recent_interactions = "Interaction information not available"  # 这部分也需要额外的API支持
        
        return self.specific_agent_env_template.substitute(
            agent_id=agent_id,
            bio=bio,
            recent_posts=recent_posts,
            recent_comments=recent_comments,
            recent_interactions=recent_interactions
        )
