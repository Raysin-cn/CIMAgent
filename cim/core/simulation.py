import pandas as pd
import json
import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
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
import oasis
from ..config import config
    


# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class InterviewResult:
    """采访结果数据类"""
    def __init__(self, agent_id: int, agent_name: str, topic: str, 
                 prompt: str, response: str, timestamp: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.topic = topic
        self.prompt = prompt
        self.response = response
        self.timestamp = timestamp


async def conduct_interview(env: OasisEnv, agent_id: int, topic: str, 
                          prompt: str = None) -> Optional[InterviewResult]:
    """
    对指定智能体进行采访
    
    Args:
        env: Oasis环境对象
        agent_id: 智能体ID
        topic: 采访话题
        prompt: 采访问题，如果为None则使用默认问题
        
    Returns:
        采访结果对象
    """
    try:
        # 获取智能体信息
        agent = env.agent_graph.get_agent(agent_id)
        if agent is None:
            logger.warning(f"智能体 {agent_id} 不存在")
            return None
        
        # 如果没有提供问题，使用默认问题
        if prompt is None:
            prompt = f"关于'{topic}'这个话题，请分享您的观点和立场。"
        
        # 创建采访动作
        interview_action = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": prompt}
        )
        
        # 执行采访
        actions = {agent: interview_action}
        await env.step(actions)
        
        # 记录采访结果
        timestamp = datetime.now().isoformat()
        result = InterviewResult(
            agent_id=agent_id,
            agent_name=agent.name if hasattr(agent, 'name') else f"Agent_{agent_id}",
            topic=topic,
            prompt=prompt,
            response="采访已执行，响应将在数据库中记录",  # 实际响应会存储在数据库中
            timestamp=timestamp
        )
        
        logger.info(f"✓ 完成对智能体 {agent_id} 的采访")
        return result
        
    except Exception as e:
        logger.error(f"❌ 采访智能体 {agent_id} 时出错: {e}")
        return None


async def conduct_batch_interviews(env: OasisEnv, agent_ids: List[int], 
                                 topic: str, prompt: str = None) -> List[InterviewResult]:
    """
    对多个智能体进行批量采访
    
    Args:
        env: Oasis环境对象
        agent_ids: 智能体ID列表
        topic: 采访话题
        prompt: 采访问题
        
    Returns:
        采访结果列表
    """
    results = []
    
    for agent_id in agent_ids:
        result = await conduct_interview(env, agent_id, topic, prompt)
        if result:
            results.append(result)
        await asyncio.sleep(0.5)  # 避免过于频繁的采访
    
    return results


def save_interview_results(results: List[InterviewResult], output_path: str = None):
    """
    保存采访结果到文件
    
    Args:
        results: 采访结果列表
        output_path: 输出文件路径，如果为None则使用默认路径
    """
    if not results:
        logger.warning("没有采访结果需要保存")
        return
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/output/interview_results_{timestamp}.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为字典格式
    data = []
    for result in results:
        data.append({
            'agent_id': result.agent_id,
            'agent_name': result.agent_name,
            'topic': result.topic,
            'prompt': result.prompt,
            'response': result.response,
            'timestamp': result.timestamp
        })
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ 采访结果已保存到: {output_path}")


async def run_simulation_steps(env, num_steps: int = 5, 
                             interview_interval: int = 2,
                             interview_topic: str = "当前热门话题",
                             interview_prompt: str = None,
                             target_agents: List[int] = None):
    """
    让代理进行多步互动模拟，并在指定间隔进行采访
    
    Args:
        env: Oasis环境对象
        num_steps: 模拟步数
        interview_interval: 采访间隔（每几步进行一次采访）
        interview_topic: 采访话题
        interview_prompt: 采访问题
        target_agents: 目标智能体ID列表，如果为None则采访所有非匿名智能体
    """
    logger.info("开始社交平台模拟...")
    

    for step in range(num_steps):
        try:
            # 执行常规社交互动
            llm_actions = {}
            for agent_id, agent in env.agent_graph.get_agents()[1:]:  # 匿名智能体不执行动作
                llm_actions[agent] = LLMAction()
            await env.step(llm_actions)
            logger.info(f"✓ 步骤 {step + 1}: {len(llm_actions)} 个代理进行了互动")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ 步骤 {step + 1} 出错: {e}")
    
    await env.close()
    logger.info("✓ 模拟完成")


async def run_simulation_with_interviews(env, num_steps: int = 5, 
                                       interview_config: Dict = None):
    """
    运行带采访功能的模拟
    
    Args:
        env: Oasis环境对象
        num_steps: 模拟步数
        interview_config: 采访配置字典，包含以下字段：
            - interval: 采访间隔（默认: 2）
            - topic: 采访话题（默认: "当前热门话题"）
            - prompt: 采访问题（可选）
            - target_agents: 目标智能体ID列表（可选）
    """
    if interview_config is None:
        interview_config = {}
    
    await run_simulation_steps(
        env=env,
        num_steps=num_steps,
        interview_interval=interview_config.get('interval', 2),
        interview_topic=interview_config.get('topic', "当前热门话题"),
        interview_prompt=interview_config.get('prompt'),
        target_agents=interview_config.get('target_agents')
    )