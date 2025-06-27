# cim/core/anonymous_agent_controller.py

import random
import asyncio
from oasis.social_agent.agent_action import SocialAction
from oasis import ManualAction, ActionType
from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole
import re

class AnonymousAgentController:
    def __init__(self, env):
        self.env = env
        self.anon_agent = env.agent_graph.get_agent(0)  # 假设匿名智能体id为0

    def select_target_agents(self, strategy="random", k=1):
        """
        Select target agents using the agent_graph structure (supports igraph/neo4j backend):
        - random: randomly select k agents
        - topk_degree: select top-k nodes with highest degree
        - topk_influence: select k nodes covering most neighbors (greedy)
        """
        agent_graph = self.env.agent_graph
        backend = getattr(agent_graph, 'backend', 'igraph')
        if backend == 'igraph':
            all_ids = [v.index for v in agent_graph.graph.vs if v.index != 0]
        else:
            all_ids = [aid for aid in agent_graph.get_all_nodes() if aid != 0]
        if not all_ids or k <= 0:
            return []
        if strategy == "random":
            return random.sample(all_ids, min(k, len(all_ids)))
        elif strategy == "topk_degree":
            if backend == 'igraph':
                degree_list = sorted(all_ids, key=lambda n: agent_graph.graph.degree(n), reverse=True)
            else:
                edge_list = agent_graph.get_all_edges()
                from collections import Counter
                degree_count = Counter()
                for src, dst in edge_list:
                    degree_count[src] += 1
                    degree_count[dst] += 1
                degree_list = sorted(all_ids, key=lambda n: degree_count[n], reverse=True)
            return degree_list[:k]
        elif strategy == "topk_influence":
            selected = set()
            covered = set()
            for _ in range(min(k, len(all_ids))):
                best = None
                best_cover = -1
                for node in all_ids:
                    if node in selected:
                        continue
                    if backend == 'igraph':
                        neighbors = set(agent_graph.graph.neighbors(node)) - covered
                    else:
                        edge_list = agent_graph.get_all_edges()
                        neighbors = set([dst for src, dst in edge_list if src == node]) - covered
                    if len(neighbors) > best_cover:
                        best = node
                        best_cover = len(neighbors)
                if best is not None:
                    selected.add(best)
                    if backend == 'igraph':
                        covered.update(agent_graph.graph.neighbors(best))
                    else:
                        edge_list = agent_graph.get_all_edges()
                        covered.update([dst for src, dst in edge_list if src == best])
            return list(selected)
        else:
            return random.sample(all_ids, min(k, len(all_ids)))

    def get_user_profile(self, target_id):
        try:
            agent_obj = self.env.agent_graph.get_agent(target_id)
            return {
                "user_id": getattr(agent_obj, 'social_agent_id', target_id),
                "user_name": getattr(agent_obj, 'user_name', ''),
                "name": getattr(agent_obj, 'name', ''),
                "bio": getattr(agent_obj, 'bio', ''),
            }
        except Exception:
            return {"user_id": target_id, "user_name": "", "name": "", "bio": ""}

    def generate_persuasion_prompt(self, user_profile, stance_goal, topic):
        # 可扩展为LLM prompt模板
        name = user_profile.get("name") or user_profile.get("user_name") or "friend"
        bio = user_profile.get("bio", "")
        stance_map = {
            "favor": "support",
            "against": "oppose",
            "none": "remain neutral",
            "mixed": "think dialectically"
        }
        stance_text = stance_map.get(stance_goal, "support")
        intro = f"Hi {name}, I noticed your profile: {bio}. " if bio else f"Hi {name}. "
        persuade = f"I'd like to talk with you about '{topic}'. On this topic, I tend to {stance_text}. What do you think?"
        return f"{intro}{persuade}"

    def clean_think_tags(self, text):
        # 去除 <think>、</think>、<|think|>、<THINK> 等标签及其内容
        # 1. 去除 <think>...</think> 结构
        text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # 2. 去除 <|think|> 及类似标签
        text = re.sub(r"<\|?/?\s*think\s*\|?>", "", text, flags=re.IGNORECASE)
        # 3. 去除多余空行和首尾空格
        return text.strip()

    async def multi_turn_persuasion_dialogue(self, target_id, stance_goal, topic, num_turns=3):
        """
        匿名智能体与目标用户进行多轮LLM驱动的双向记忆对话。
        返回完整对话历史。
        """
        anon_agent = self.anon_agent
        target_agent = self.env.agent_graph.get_agent(target_id)
        # 保存原始 system prompt
        original_anon_system_message = anon_agent.system_message
        original_target_system_message = target_agent.system_message

        # 设置匿名体 system prompt，包含目标和总轮数
        anon_prompt = (
            f"You are an anonymous persuader. Your goal is to persuade the other user to adopt the '{stance_goal}' stance on the topic '{topic}'. "
            f"You will have a total of {num_turns} rounds to persuade the user. "
            "Each time, you will be told which round it is."
        )
        anon_agent._system_message = BaseMessage.make_assistant_message(
            role_name="system", content=anon_prompt
        )
        anon_agent.init_messages()

        # 设置目标体 system prompt（可选，或保持原有）
        target_agent._system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=(
                "You are a social media user. Please respond naturally to the persuasion. "
                "During this conversation, you must only reply in plain text and must not use any platform tools, functions, or actions. "
                "Do not call any tool or function. Just chat."
            )
        )
        target_agent.init_messages()

        dialogue_history = []
        user_profile = self.get_user_profile(target_id)

        # 第一轮，匿名体根据目标profile和stance_goal生成"说服开场白"
        anon_text = self.generate_persuasion_prompt(user_profile, stance_goal, topic)
        # 构造一条assistant消息
        assistant_msg = BaseMessage.make_assistant_message(
            role_name="AnonymousAgent",  # 或你的agent名
            content=anon_text
        )
        anon_agent.update_memory(assistant_msg, OpenAIBackendRole.ASSISTANT)
        dialogue_history.append({
            "from": 0, "to": target_id, "content": anon_text, "turn": 1, "round": 1
        })

        for turn in range(1, num_turns+1):
            # 目标体回应
            target_response = await target_agent.astep(anon_text)
            target_text = self.clean_think_tags(target_response.msgs[0].content)
            dialogue_history.append({
                "from": target_id, "to": 0, "content": target_text, "turn": turn*2, "round": turn
            })
            # 匿名体下一轮继续说服（带上轮次提示）
            round_info = f"(This is round {turn+1} of {num_turns})"
            anon_input = f"{round_info}\n{target_text}"
            anon_response = await anon_agent.astep(anon_input)
            anon_text = self.clean_think_tags(anon_response.msgs[0].content)
            dialogue_history.append({
                "from": 0, "to": target_id, "content": anon_text, "turn": turn*2+1, "round": turn+1
            })

        # 恢复原始 system prompt
        anon_agent._system_message = original_anon_system_message
        anon_agent.init_messages()
        target_agent._system_message = original_target_system_message
        return dialogue_history

    async def batch_multi_turn_persuasion(self, stance_goal, topic, strategy="topk_degree", k=5, num_turns=3):
        """
        并发对多个目标用户进行多轮LLM对话，返回{target_id: dialogue_history}
        """
        target_ids = self.select_target_agents(strategy, k)
        tasks = [self.multi_turn_persuasion_dialogue(target_id, stance_goal, topic, num_turns) for target_id in target_ids]
        results = await asyncio.gather(*tasks)
        return {target_id: history for target_id, history in zip(target_ids, results)}

    def plan_group_persuasion(self, stance_goal, strategy="topk_degree", k=5, num_turns=3):
        """
        返回一个动作计划列表，每个元素为{agent: Action}，可直接传给env.step
        """
        target_ids = self.select_target_agents(strategy, k)
        actions_plan = []
        for target_id in target_ids:
            
            user_profile = self.get_user_profile(target_id)
            for turn in range(num_turns):
                message = self.generate_persuasion_prompt(user_profile, stance_goal, "COVID-19 vaccination")
                group_name = f"persuasion_group_{target_id}"
                # 1. 创建群聊
                actions_plan.append({
                    self.anon_agent: ManualAction(
                        action_type=ActionType.CREATE_GROUP,
                        action_args={"group_name": group_name}
                    )
                })
                # 2. 邀请目标用户加入群聊（如有INVITE_TO_GROUP动作，否则直接让目标用户join_group）
                # 3. 目标用户加入群聊
                actions_plan.append({
                    self.env.agent_graph.get_agent(target_id): ManualAction(
                        action_type=ActionType.JOIN_GROUP,
                        action_args={"group_id": f"${group_name}_id"}  # 需在env.step后获取真实id
                    )
                })
                # 4. 匿名智能体多轮发言
                actions_plan.append({
                    self.anon_agent: ManualAction(
                        action_type=ActionType.SEND_TO_GROUP,
                        action_args={"group_id": f"${group_name}_id", "content": message}
                    )
                })
        return actions_plan