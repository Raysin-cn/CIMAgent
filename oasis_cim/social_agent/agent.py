# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import inspect
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union
import json
from pydantic import BaseModel, ValidationError
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from camel.agents._types import ModelResponse, ToolCallRequest
from camel.types.agents import ToolCallingRecord
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.memories import MemoryRecord
from camel.types import OpenAIBackendRole

# from oasis.social_agent.agent_action import SocialAction
# from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

# modified_oasis_cim
from oasis_cim.social_agent.agent_action import SocialAction
from oasis_cim.social_agent.agent_environment import SocialEnvironment

if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")

    if not agent_log.handlers:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            f"./log/social.agent-{str(now)}.log")
        file_handler.setLevel("DEBUG")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
        agent_log.addHandler(file_handler)


class HiddenAgentResponse(BaseModel):
    """隐藏智能体的响应格式模型"""
    say2user: str  # 智能体对用户说的话


class SocialAgent(ChatAgent):
    r"""Social Agent."""

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        twitter_channel: Channel,
        model: Optional[Union[BaseModelBackend,
                              List[BaseModelBackend]]] = None,
        agent_graph: "AgentGraph" = None,
        available_actions: list[ActionType] = None,
    ):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))

        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=self.user_info.to_system_message(),  # system prompt
        )

        if not available_actions:
            agent_log.info("No available actions defined, using all actions.")
            self.action_tools = self.env.action.get_openai_function_list()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        super().__init__(system_message=system_message,
                         model=model,
                         scheduling_strategy='random_model',
                         tools=self.action_tools,
                         single_iteration=True)
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")

    async def perform_action_by_llm(self):
        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"))
        try:
            # agent_log.info(
            #     f"Agent {self.social_agent_id} observing environment: "
            #     f"{env_prompt}")
            response = await self.astep(user_msg)
            for tool_call in response.info['tool_calls']:
                action_name = tool_call.tool_name
                args = tool_call.args
                agent_log.info(f"Agent {self.social_agent_id} performed "
                               f"action: {action_name} with args: {args}")
                # Abort graph action for if 100w Agent
                self.perform_agent_graph_action(action_name, args)
            agent_log.info("+"*10 + f"  Agent {self.social_agent_id} performed actions: {response}  " + "+"*10)
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")

    async def perform_test(self):
        """
        doing test for all agents.
        TODO: rewrite the function according to the ChatAgent.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(role_name="User",
                                          content=("You are a twitter user."))
        # Test memory should not be writed to memory.
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": self.test_prompt
        }])

        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.
        response = self._get_model_response(openai_messages=openai_messages,
                                            num_tokens=num_tokens)
        content = response.output_messages[0].content
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_action_by_hci(self) -> Any:
        print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            agent_log.info(f"Agent {self.social_agent_id} function: "
                           f"{function_list[i].func.__name__}")

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.social_agent_id} invalid input.")
            return
        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        func_name = func_name.value if isinstance(func_name,
                                                  ActionType) else func_name
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                agent_log.info(f"Agent {self.social_agent_id}: {result}")
                return result
        raise ValueError(f"Function {func_name} not found in the list.")




    #TODO：需要更改agent执行关注与取关的动作，通过CIM
    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} unfollowed Agent {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} followed Agent {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
    


class SocialHiddenAgent(ChatAgent):

    def __init__(
        self,
        agent_id: int,
        user_info: Dict[str, Any],
        twitter_channel: Channel,
        model: Optional[Union[BaseModelBackend,
                              List[BaseModelBackend]]] = None,
        agent_graph: "AgentGraph" = None,
        available_actions: list[ActionType] = None,
    ):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))

        system_content = (f"# Objective: {self.user_info.get('description')}\n"
                            f"# Profile: {self.user_info.get('profile')}\n"
                            f"# Recsys Type: {self.user_info.get('recsys_type')}\n"
                            f"# RESPONSE METHOD: Please say something to the user, what you say will be recorded in the user's memory."
                            "# RESPONSE FORMAT: {'say2user': 'what you say to the user'}")
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=system_content,  # system prompt
        )

        if not available_actions:
            agent_log.info("No available actions defined, using all actions.")
            self.action_tools = self.env.action.get_openai_function_list()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        super().__init__(system_message=system_message,
                         model=model,
                         scheduling_strategy='random_model',
                         tools=self.action_tools,
                         single_iteration=True)
        self.agent_graph = agent_graph

    async def astep(self, input_message: Union[BaseMessage, str], response_format: Optional[Type[BaseModel]] = None):
        """
        重写ChatAgent的astep方法,用于处理输入消息并生成响应
        
        Args:
            input_message: 输入消息,可以是BaseMessage对象或字符串
            response_format: 可选的响应格式模型类
            
        Returns:
            ChatAgentResponse: 包含输出消息、工具调用记录等信息的响应对象
        """
        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        
        try:
            openai_messages, num_tokens = self.memory.get_context()
        except RuntimeError as e:
            return self._step_token_exceed(
                e.args[1], tool_call_records, "max_tokens_exceeded"
            )

        # 如果指定了响应格式，则不使用工具调用
        if response_format is not None:
            response = await self._aget_model_response(
                openai_messages,
                num_tokens,
                response_format,
                None,  # 不使用工具
            )
        else:
            response = await self._aget_model_response(
                openai_messages,
                num_tokens,
                response_format,
                self._get_full_tool_schemas(),
            )

            # if tool_call_requests := response.tool_call_requests:
            #     # Process all tool calls
            #     for tool_call_request in tool_call_requests:
            #         if (
            #             tool_call_request.tool_name
            #             in self._external_tool_schemas
            #         ):
            #             if external_tool_call_requests is None:
            #                 external_tool_call_requests = []
            #             external_tool_call_requests.append(tool_call_request)

            #         tool_call_record = await self._aexecute_tool(
            #             tool_call_request
            #         )
            #         tool_call_records.append(tool_call_record)

            #     # If we found an external tool call, break the loop
            #     if external_tool_call_requests:
            #         break

            #     if self.single_iteration:
            #         break

            #     # If we're still here, continue the loop
            #     continue


        await self._aformat_response_if_needed(response, response_format)
        self._record_final_output(response.output_messages)

        return self._convert_to_chatagent_response(
            response,
            tool_call_records,
            num_tokens,
            external_tool_call_requests,
        )

    async def perform_action_by_hidden_agent(self, agent_id: int, *args, **kwargs):
        env_prompt = await self.env.get_specific_agent_env(agent_id)
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"
                f"And your goal is {kwargs.get('goal', '')}"
                )
        )

        # 获取hidden agent的响应，使用HiddenAgentResponse作为响应格式
        response = await self.astep(user_msg, response_format=HiddenAgentResponse)
        
        # 获取响应内容
        try:
            response_content = json.loads(response.msgs[0].content) if response.msgs else {"say2user": "Nothing to say."}
        except (json.JSONDecodeError, AttributeError):
            response_content = {"say2user": response.msgs[0].content if response.msgs else "Nothing to say."}
        
        # 创建记忆记录
        # 记录用户的查询消息
        self.memory.write_record(
            MemoryRecord(
                message=user_msg,
                role=OpenAIBackendRole.USER,
                role_at_backend=OpenAIBackendRole.USER
            )
        )
        
        # 记录助手的响应消息
        assistant_msg = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=json.dumps({
                'action': 'hidden_agent_response',
                'response_content': response_content,
                'target_agent_id': agent_id,
                'timestamp': datetime.now().isoformat()
            }, ensure_ascii=False, indent=2)
        )
        self.memory.write_record(
            MemoryRecord(
                message=assistant_msg,
                role=OpenAIBackendRole.ASSISTANT,
                role_at_backend=OpenAIBackendRole.ASSISTANT
            )
        )
        
        # 如果目标智能体存在，也在其记忆中记录这次交互
        target_agent = self.agent_graph.get_agent(agent_id)
        if target_agent and hasattr(target_agent, 'memory'):
            # 记录用户（观察者）的消息
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=f"Someone says this to me: {response_content.get('say2user', '')}"
            )
            target_agent.memory.write_record(
                MemoryRecord(
                    message=user_msg,
                    role=OpenAIBackendRole.USER,
                    role_at_backend=OpenAIBackendRole.USER
                )
            )
            
            # # 记录系统元数据（可选）
            # metadata_msg = BaseMessage.make_system_message(
            #     role_name="System",
            #     content=json.dumps({
            #         'event': 'being_observed',
            #         'observer_agent_id': 'Some hidden person',
            #         'observation_time': datetime.now().isoformat()
            #     }, ensure_ascii=False)
            # )
            # target_agent.memory.write_record(
            #     MemoryRecord(
            #         message=metadata_msg,
            #         role=OpenAIBackendRole.SYSTEM,
            #         role_at_backend=OpenAIBackendRole.SYSTEM
            #     )
            # )
        
        return response
