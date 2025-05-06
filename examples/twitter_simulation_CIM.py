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
import os
import pandas as pd

from camel.models import ModelFactory
from camel.types import ModelPlatformType

# import oasis
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis import ActionType, EnvAction, SingleAction
import oasis_cim as oasis



async def main():
    # NOTE: You need to deploy the vllm server first
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="/data/model/Qwen3-14B",
        url="http://localhost:21474/v1",
        model_config_dict={"max_tokens": 32000}
    )
    vllm_model_2 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="/data/model/Qwen3-14B",
        url="http://localhost:21474/v1",
        model_config_dict={"max_tokens": 10000}
    )
    # Define the models for agents. Agents will select models based on
    # pre-defined scheduling strategies
    models = [vllm_model_1, vllm_model_2]

    # Define the available actions for the agents
    available_actions = [
        ActionType.REFRESH,
        ActionType.SEARCH_USER,
        ActionType.SEARCH_POSTS,
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.UNLIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.UNDO_DISLIKE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.UNLIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.UNDO_DISLIKE_COMMENT,
        ActionType.FOLLOW,
        ActionType.UNFOLLOW,
        ActionType.MUTE,
        ActionType.UNMUTE,
        ActionType.TREND,
        ActionType.REPOST,
        ActionType.QUOTE_POST,
        ActionType.DO_NOTHING,
    ]

    # Define the path to the database
    db_path = "./data/twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Read user profiles
    users_df = pd.read_csv("data/twitter_dataset_CIM/processed_users.csv")
    
    # Create a mapping from user_id to agent_id (0-based index)
    user_to_agent = {user_id: idx for idx, user_id in enumerate(users_df['user_id'].values)}
    
    # Make the environment
    env = oasis.make(
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        agent_profile_path=("data/twitter_dataset_CIM/processed_users.csv"),
        agent_models=models,
        available_actions=available_actions,
        time_engine="activity_level_frequency",
    )

    # Run the environment
    await env.reset()

    # Read all_topics.csv
    topics_df = pd.read_csv("data/twitter_dataset_CIM/post_topics.csv")
    
    # Create initial posts from topics
    initial_actions = []
    for idx, row in topics_df.iterrows():
        # Get the original poster's user_id and map it to agent_id
        root_user_id = int(row['root_user'])
        agent_id = user_to_agent.get(root_user_id, idx % len(users_df))  # Fallback to modulo if user not found
        
        action = SingleAction(
            agent_id=agent_id,
            action=ActionType.CREATE_POST,
            args={"content": row["source_tweet"]}
        )
        initial_actions.append(action)

    # Create environment action with all initial posts
    env_actions = EnvAction(
        activate_agents=list(range(len(users_df))),  # Activate all agents
        intervention=initial_actions
    )

    # Perform initial actions
    await env.step(env_actions)

    # Run for 1 days (24 * 1 = 24 timesteps)
    for _ in range(24):
        # Create empty action to let all agents act
        empty_action = EnvAction()
        await env.step(empty_action)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    # test vllm model
    # async def test_model():
    #     vllm_model = ModelFactory.create(
    #         model_platform=ModelPlatformType.VLLM,
    #         model_type="Qwen2.5-14B-Instruct",
    #         url="http://localhost:21474/v1",
    #     )
    #     print(dir(vllm_model))
    #     # 假设模型有 chat 接口
    #     response = await vllm_model.run([{"role": "user", "content": "你好！"}])
    #     print(response)
    # asyncio.run(test_model())
    asyncio.run(main())
