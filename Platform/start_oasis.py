import asyncio
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
import Platform.oasis_cim as oasis
from oasis import ActionType, EnvAction, SingleAction

import os
import dotenv

dotenv.load_dotenv(override=True)

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))

async def main():
  # Define the model for the agents
  openai_model = ModelFactory.create(
      model_platform=ModelPlatformType.OPENAI,
      model_type=ModelType.GPT_4O_MINI,
      api_key=os.getenv("OPENAI_API_KEY"),
      url=os.getenv("OPENAI_BASE_URL"),
  )

  # Define the available actions for the agents
  available_actions = [
      ActionType.LIKE_POST,
      ActionType.CREATE_POST,
      ActionType.CREATE_COMMENT,
      ActionType.FOLLOW
  ]

  # Make the environment
  env = oasis.make(
      platform=oasis.DefaultPlatformType.REDDIT,
      database_path="reddit_simulation.db",
      agent_profile_path="./data/reddit/user_data_36.json",
      agent_models=openai_model,
      available_actions=available_actions,
  )

  # Run the environment
  await env.reset()

  action = SingleAction(
    agent_id=0,
    action=ActionType.CREATE_POST,
    args={"content": "Welcome to the OASIS World!"}
  )

  env_actions = EnvAction(
    activate_agents=list(range(10)),  # activate the first 10 agents
    intervention=[action]
  )

  # Apply interventions to the environment, refresh the recommendation system, and LLM agent perform actions
  await env.step(env_actions)

  # Close the environment
  await env.close()

if __name__ == "__main__":
  asyncio.run(main())