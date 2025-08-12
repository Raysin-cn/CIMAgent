from __future__ import annotations

import ast
import asyncio
import json
from typing import List, Optional, Union

import pandas as pd
import tqdm
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.types import OpenAIBackendRole
from camel.prompts import TextPrompt

from oasis.social_agent import AgentGraph, SocialAgent
from oasis.social_platform import Channel, Platform
from oasis.social_platform.config import Neo4jConfig, UserInfo
from oasis.social_platform.typing import ActionType


def build_user_info_template_for_topic(topic: str) -> TextPrompt:
    """根据给定话题构造用户系统提示模板。

    先将话题文本直接嵌入模板（不作为占位符），再保留 `{description}` 占位符，
    使得 `SocialAgent` 通过 `UserInfo.to_custom_system_message` 自动从
    `UserInfo.profile['description']` 填充描述。
    """
    template = f"""
# OBJECTIVE
You are a Twitter user currently participating in a debate discussion on the following topic: {topic}
After reading posts related to this topic, you are expected to express your own stance clearly—your stance should be one of: support, oppose, or neutral. Please ensure your actions and statements are focused on this topic, and always make your position explicit in your responses.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{{description}}

# RESPONSE METHOD
Please perform actions by tool calling.
"""
    return TextPrompt(template)


async def generate_twitter_agent_graph(
    profile_path: str,
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None,
    topic: str | None = None,
) -> AgentGraph:
    agent_info = pd.read_csv(profile_path)
    agent_graph = AgentGraph()

    for agent_id in range(len(agent_info)):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info["user_char"][
            agent_id]
        # 供自定义模板中的 {description} 自动填充
        profile["description"] = agent_info["description"][agent_id]

        user_info = UserInfo(
            name=agent_info["username"][agent_id],
            description=agent_info["description"][agent_id],
            profile=profile,
            recsys_type='twitter',
        )

        # 基于话题构造模板；若无话题，则使用默认系统提示
        user_info_tmpl = (
            build_user_info_template_for_topic(topic)
            if topic is not None else None
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            user_info_template=user_info_tmpl,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.add_agent(agent)
    return agent_graph