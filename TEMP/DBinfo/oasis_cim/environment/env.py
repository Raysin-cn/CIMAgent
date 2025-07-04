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
import asyncio
import logging
import os
import random
from datetime import datetime
from typing import List, Optional, Union

from camel.models import BaseModelBackend

# from oasis.environment.env_action import EnvAction, SingleAction
# from oasis.social_agent.agents_generator import (generate_agents,
#                                                  generate_reddit_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)

# modified_oasis_cim
from oasis_cim.environment.env_action import EnvAction, SingleAction
from oasis_cim.social_agent.agents_generator import (generate_agents, generate_reddit_agents, generate_hidden_agents)

# Create log directory if it doesn't exist
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logger
env_log = logging.getLogger("oasis.env")
env_log.setLevel("INFO")

# Add file handler to save logs to file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_handler = logging.FileHandler(f"{log_dir}/oasis-{current_time}.log",
                                   encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
env_log.addHandler(file_handler)


class OasisEnv:

    def __init__(
        self,
        platform: Union[DefaultPlatformType, Platform],
        agent_profile_path: str,
        database_path: str = None,
        agent_models: Optional[Union[BaseModelBackend,
                                     List[BaseModelBackend]]] = None,
        available_actions: list[ActionType] = None,
        semaphore: int = 128,
        time_engine: str = None,
    ) -> None:
        r"""Init the oasis environment.

        Args:
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
            agent_profile_path: The path to the agent profile. Make sure the
                data format is align with the `platform`.
            agent_models: The model backend to use for all agents to generate
                responses. (default: :obj:`ModelPlatformType.DEFAULT` with
                `ModelType.DEFAULT`)
            available_actions: The actions to use for the agents. Choose from
                `ActionType`.
            time_engine: The type of time engine to use. Can be "activity_level" or 
                "activity_level_frequency". If None, no time engine is used.
        """
        self.agent_profile_path = agent_profile_path
        self.agent_models = agent_models
        self.available_actions = available_actions
        self.time_engine = time_engine
        # Use a semaphore to limit the number of concurrent requests
        self.llm_semaphore = asyncio.Semaphore(semaphore)
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError(
                    "database_path is required for DefaultPlatformType")
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="twhin-bert",
                    refresh_rec_post_count=2,
                    max_rec_post_len=2,
                    following_post_count=3,
                )
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="reddit",
                    allow_self_rating=True,
                    show_score=True,
                    max_rec_post_len=100,
                    refresh_rec_post_count=5,
                )
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f"Invalid platform: {platform}. Only "
                                 "DefaultPlatformType.TWITTER or "
                                 "DefaultPlatformType.REDDIT are supported.")
        elif isinstance(platform, Platform):
            if database_path != platform.db_path:
                env_log.warning("database_path is not the same as the "
                                "platform.db_path, using the platform.db_path")
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                self.platform_type = DefaultPlatformType.TWITTER
        else:
            raise ValueError(
                f"Invalid platform: {platform}. You should pass a "
                "DefaultPlatformType or a Platform instance.")

    async def reset(self) -> None:
        r"""Start the platform and sign up the agents.
        """
        self.platform_task = asyncio.create_task(self.platform.running())
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.agent_graph = await generate_agents(
                agent_info_path=self.agent_profile_path,
                twitter_channel=self.channel,
                model=self.agent_models,
                recsys_type=RecsysType.TWHIN,
                start_time=self.platform.sandbox_clock.time_step,
                available_actions=self.available_actions,
                twitter=self.platform,
            )
        elif self.platform_type == DefaultPlatformType.REDDIT:
            self.agent_graph = await generate_reddit_agents(
                agent_info_path=self.agent_profile_path,
                twitter_channel=self.channel,
                model=self.agent_models,
                available_actions=self.available_actions,
            )
        # NOTE 生成隐藏智能体
        self.hidden_agent = await generate_hidden_agents(
            agent_info_path=self.agent_profile_path,
            twitter_channel=self.channel,
            model=self.agent_models,
            available_actions=self.available_actions,
            twitter=self.platform,
            agent_graph=self.agent_graph,
        )

    async def _perform_control_action(self, action: SingleAction) -> None:
        r"""Perform a control action.

        Args:
            action(SingleAction): The action to perform.
        """
        control_agent = self.agent_graph.get_agent(action.agent_id)
        await control_agent.perform_action_by_data(action.action,
                                                   **action.args)

    #TODO: 需要更改LLM_action，关注动作等改变graph结构的需要用CIM框架
    async def _perform_llm_action(self, agent):
        r"""Send the request to the llm model and execute the action.
        """
        async with self.llm_semaphore:
            return await agent.perform_action_by_llm()

    async def step(self, action: EnvAction) -> None:
        r"""Perform some control actions, update the recommendation system,
        and let some llm agents perform actions.

        Args:
            action(EnvAction): The activate agents and control actions to
                perform.
        """
        # Control some agents to perform actions
        if action.intervention:
            control_tasks = [
                self._perform_control_action(single_action)
                for single_action in action.intervention
            ]
            await asyncio.gather(*control_tasks)
            env_log.info("performed control actions.")

        # Update the recommendation system
        await self.platform.update_rec_table()
        env_log.info("update rec table.")

        # Some llm agents perform actions
        #在此处需要引入时间引擎，使得特定的agent在特定的时间才能执行动作.
        if action.activate_agents:
            activate_agents = action.activate_agents
        elif self.time_engine == "activity_level":
            current_time = self.platform.sandbox_clock.time_step%24
            print("Sandbox clock current_time: ", current_time)
            activate_agents = [
                agent_id for agent_id, _ in self.agent_graph.get_agents()
                if self.agent_graph.get_agent(agent_id).user_info.profile["other_info"].get("activity_level", ["off_line"]*24)[current_time] != "off_line"
            ]
        elif self.time_engine == "activity_level_frequency":
            current_time = self.platform.sandbox_clock.time_step%24
            print("Sandbox clock current_time: ", current_time)
            activate_agents = []
            for agent_id, _ in self.agent_graph.get_agents():
                activity_level_frequency = self.agent_graph.get_agent(agent_id).user_info.profile["other_info"].get("activity_level_frequency", [0]*24)[current_time]
                if random.random()/24 < activity_level_frequency:
                    activate_agents.append(agent_id)
        else:
            env_log.warning(
                "activate_agents and time_engine are None, default to activate all agents.")
            activate_agents = [
                agent_id for agent_id, _ in self.agent_graph.get_agents()
            ]

        llm_tasks = []
        for agent_id in activate_agents:
            agent = self.agent_graph.get_agent(agent_id)
            llm_tasks.append(self._perform_llm_action(agent))

        await asyncio.gather(*llm_tasks)
        env_log.info("performed llm actions.")
        # Update the clock
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.sandbox_clock.time_step += 1

    async def close(self) -> None:
        r"""Stop the platform and close the environment.
        """
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")

    # NOTE 使用隐藏智能体针对社交网络中的种子节点进行交互
    async def select_seeds(self, seed_nums_rate: float = 0.1) -> list[int]:
        r"""Select the seeds for the control action.
        """
        seeds = self.agent_graph.select_seeds(seed_nums_rate = seed_nums_rate)
        return seeds

    async def hidden_control(self, seeds: list[int]) -> None:
        r"""control the hidden_agent to spread the information to the seeds.
        """
        # for agent_id in seeds:
            # specific_action = SingleAction(
            #     agent_id=self.hidden_agent.agent_id,
            #     action=ActionType.FOLLOW,
            #     args={"followee_id": agent_id}
            # )
            # await self._perform_control_action(specific_action)
            
        goal = "Through positive interaction and guidance, help target users develop an optimistic and supportive stance on the topic. Build trust and influence their perspective to gradually shift towards a positive and pro-stance position."
        specific_action = [
            self.hidden_agent.perform_action_by_hidden_agent(agent_id, goal=goal)
            for agent_id in seeds
        ]
        await asyncio.gather(*specific_action)
