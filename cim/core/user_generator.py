"""
用户生成器模块

目标：生成与 `data/raw/users_info_10.csv` 相同字段结构的用户CSV。

- 基于LLM生成核心档案字段（name/username/description/user_char）
- 随机模拟粉丝/关注、活跃频率与活跃等级
- 构造关注与被关注关系字段
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 将项目根目录加入路径，便于相对导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from cim.config import config


logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


@dataclass
class GeneratedUser:
    user_id: int
    name: str
    username: str
    description: str
    user_char: str
    created_at: str
    followers_count: int
    following_count: int
    activity_level_frequency: List[int]
    activity_level: List[str]
    following_list: List[int]
    following_agentid_list: List[int]
    followers_list: List[int]
    previous_tweets: List[int]
    tweets_id: List[int]


class UserGenerator:
    """用户CSV生成器。"""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        self.model_config = model_config or config.model.__dict__
        self.model = None
        self.agent: Optional[ChatAgent] = None
        logger.info("初始化用户生成器")

    def _init_model(self) -> None:
        if self.model is not None:
            return
        try:
            if self.model_config["platform"].upper() == "VLLM":
                self.model = ModelFactory.create(
                    model_platform=ModelPlatformType.VLLM,
                    model_type=self.model_config["model_type"],
                    url=self.model_config["url"],
                    model_config_dict={
                        "max_tokens": self.model_config.get("max_tokens", 20480),
                        "temperature": self.model_config.get("temperature", 1.0),
                    },
                )
            else:
                self.model = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4O_MINI,
                )

            self.agent = ChatAgent(
                system_message=(
                    "You are an assistant that creates realistic social media user profiles. "
                    "Return strictly valid JSON without any additional text."
                ),
                model=self.model,
            )
            logger.info(f"✓ 成功初始化模型: {self.model_config['platform']}")
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise

    @staticmethod
    def _random_created_at(start_year: int = 2007, end_year: int = 2014) -> str:
        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        end = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        delta = end - start
        rand_dt = start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
        return rand_dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    @staticmethod
    def _generate_activity_pattern() -> Tuple[List[int], List[str]]:
        """生成24小时活跃频率与对应活跃等级。"""
        # 基础活跃时段（早/午/晚型）
        chronotype = random.choice(["early", "day", "night"])  # 早/日/夜
        frequency: List[int] = []

        for hour in range(24):
            base = 0
            if chronotype == "early" and 6 <= hour <= 10:
                base = random.choice([2, 3, 5, 7, 10, 14])
            elif chronotype == "day" and 11 <= hour <= 17:
                base = random.choice([2, 3, 5, 7, 10, 18])
            elif chronotype == "night" and 19 <= hour <= 23:
                base = random.choice([2, 3, 5, 7, 10, 20])
            else:
                base = random.choice([0, 0, 0, 1, 2, 3])
            frequency.append(base)

        def to_level(v: int) -> str:
            if v <= 1:
                return "off_line"
            if v == 2:
                return "normal"
            if 3 <= v <= 4:
                return "busy"
            return "active"

        levels = [to_level(v) for v in frequency]
        return frequency, levels

    def _prompt_for_profile(self, seed_hint: Optional[str] = None) -> str:
        keywords = seed_hint or random.choice(
            [
                "tech enthusiast and web developer",
                "fitness coach and wellness advocate",
                "film producer and educator",
                "travel blogger and foodie",
                "music lover and software engineer",
                "social media strategist and consultant",
            ]
        )
        prompt = f"""
Generate a realistic social media user profile with these fields:
- name: person's display name
- username: short handle without spaces
- description: short bio (1 sentence)
- user_char: concise persona summary

Persona hint: {keywords}

Return strictly JSON with keys: name, username, description, user_char. No extra text.
"""
        return prompt

    async def _gen_core_profile(self) -> Dict[str, str]:
        try:
            self._init_model()
            assert self.agent is not None
            response = await self.agent.astep(
                BaseMessage.make_user_message("User", self._prompt_for_profile())
            )
            if response.msgs:
                content = response.msgs[0].content.strip()
                # 清理潜在的<think>等标签
                import re as _re

                content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
                data = json.loads(content)
                # 最小字段校验
                for k in ["name", "username", "description", "user_char"]:
                    if k not in data or not isinstance(data[k], str):
                        raise ValueError("invalid llm fields")
                return {
                    "name": data["name"].strip()[:80],
                    "username": data["username"].strip().replace(" ", "")[:32],
                    "description": data["description"].strip()[:280],
                    "user_char": data["user_char"].strip()[:280],
                }
        except Exception as e:
            logger.warning(f"LLM生成用户档案失败，将使用回退策略: {e}")

        # 回退策略：基于模板随机生成
        first = random.choice(["Alex", "Jamie", "Taylor", "Casey", "Jordan", "Morgan"]) 
        last = random.choice(["Lee", "Kim", "Patel", "Garcia", "Smith", "Brown"]) 
        handle = f"{first.lower()}_{last.lower()}{random.randint(1, 9999)}"
        desc = random.choice(
            [
                "Tech geek, coffee addict, and part-time traveler.",
                "Sharing thoughts on fitness, productivity, and life hacks.",
                "Movie buff and educator, exploring stories that matter.",
                "Food lover on a journey to find the best local eats.",
                "Software engineer by day, music enthusiast by night.",
            ]
        )
        return {
            "name": f"{first} {last}",
            "username": handle,
            "description": desc,
            "user_char": desc,
        }

    async def generate_users(self, num_users: int = 10, seed: Optional[int] = None) -> List[GeneratedUser]:
        if seed is not None:
            random.seed(seed)

        # 先生成核心字段
        profiles: List[Dict[str, str]] = []
        for _ in range(num_users):
            profile = await self._gen_core_profile()
            profiles.append(profile)

        # 保证 username 唯一
        seen: Dict[str, int] = {}
        for p in profiles:
            base = p["username"].lower()
            if base not in seen:
                seen[base] = 0
                continue
            seen[base] += 1
            p["username"] = f"{base}{seen[base]}"

        # 生成其他字段与图结构
        users: List[GeneratedUser] = []
        user_ids = self._generate_unique_user_ids(num_users)
        created_list = [self._random_created_at() for _ in range(num_users)]

        # 初步占位，稍后补充社交图字段
        for idx in range(num_users):
            freq, levels = self._generate_activity_pattern()
            followers = random.randint(0, max(0, int(num_users * 2)))
            following = random.randint(0, max(1, int(num_users * 2)))

            users.append(
                GeneratedUser(
                    user_id=user_ids[idx],
                    name=profiles[idx]["name"],
                    username=profiles[idx]["username"],
                    description=profiles[idx]["description"],
                    user_char=profiles[idx]["user_char"],
                    created_at=created_list[idx],
                    followers_count=followers,
                    following_count=following,
                    activity_level_frequency=freq,
                    activity_level=levels,
                    following_list=[],
                    following_agentid_list=[],
                    followers_list=[],
                    previous_tweets=[],
                    tweets_id=[],
                )
            )

        # 构造社交图（关注/被关注）
        self._wire_social_graph(users)

        return users

    @staticmethod
    def _generate_unique_user_ids(n: int) -> List[int]:
        # 模拟Twitter风格的较大整数ID
        ids: set[int] = set()
        while len(ids) < n:
            candidate = random.randint(10_000_000, 9_999_999_999)
            ids.add(candidate)
        return list(ids)

    @staticmethod
    def _wire_social_graph(users: List[GeneratedUser]) -> None:
        # 让每个用户随机关注若干其他用户
        n = len(users)
        user_id_list = [u.user_id for u in users]

        for i, u in enumerate(users):
            possible_indices = [idx for idx in range(n) if idx != i]
            follow_k = min(u.following_count, len(possible_indices))
            chosen_indices = random.sample(possible_indices, k=follow_k) if follow_k > 0 else []

            u.following_agentid_list = chosen_indices  # 以行号索引表示
            u.following_list = [users[idx].user_id for idx in chosen_indices]

        # 反向构造 followers_list
        id_to_index = {u.user_id: idx for idx, u in enumerate(users)}
        for follower_idx, follower in enumerate(users):
            for followed_id in follower.following_list:
                idx = id_to_index[followed_id]
                users[idx].followers_list.append(follower.user_id)

        # 根据真实链接更新 followers_count/following_count
        for u in users:
            u.followers_count = len(u.followers_list)
            u.following_count = len(u.following_list)

    @staticmethod
    def _to_csv_row(idx: int, u: GeneratedUser) -> Dict[str, Any]:
        return {
            "Unnamed: 0": idx,
            "user_id": u.user_id,
            "name": u.name,
            "username": u.username,
            "description": u.description,
            "created_at": u.created_at,
            "followers_count": u.followers_count,
            "following_count": u.following_count,
            "following_list": json.dumps(u.following_list, ensure_ascii=False),
            "following_agentid_list": json.dumps(u.following_agentid_list, ensure_ascii=False),
            "previous_tweets": json.dumps(u.previous_tweets, ensure_ascii=False),
            "tweets_id": json.dumps(u.tweets_id, ensure_ascii=False),
            "activity_level_frequency": json.dumps(u.activity_level_frequency, ensure_ascii=False),
            "activity_level": json.dumps(u.activity_level, ensure_ascii=False),
            "user_char": u.user_char,
            "followers_list": json.dumps(u.followers_list, ensure_ascii=False),
        }

    def save_users_to_csv(self, users: List[GeneratedUser], output_path: str) -> str:
        try:
            records = [self._to_csv_row(i, u) for i, u in enumerate(users)]
            df = pd.DataFrame(records)

            # 确保列顺序与参考文件一致
            columns = [
                "Unnamed: 0",
                "user_id",
                "name",
                "username",
                "description",
                "created_at",
                "followers_count",
                "following_count",
                "following_list",
                "following_agentid_list",
                "previous_tweets",
                "tweets_id",
                "activity_level_frequency",
                "activity_level",
                "user_char",
                "followers_list",
            ]

            # 对缺失列做容错
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            df = df[columns]

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"✓ 用户CSV已保存: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ 保存用户CSV失败: {e}")
            raise


async def _main_async(num_users: int, output_path: str, seed: Optional[int]) -> None:
    generator = UserGenerator()
    users = await generator.generate_users(num_users=num_users, seed=seed)
    generator.save_users_to_csv(users, output_path)


def _resolve_default_output() -> str:
    # 默认输出放到 data/raw/users_info.csv
    try:
        default_path = os.path.join(config.paths.data_dir, "raw", "users_info_new.csv")
        return os.path.abspath(default_path)
    except Exception:
        return os.path.abspath("./data/raw/users_info_new.csv")


if __name__ == "__main__":
    """
    运行方法示例：

    1) 使用默认LLM与默认输出路径，生成10个用户：
       python -m cim.core.user_generator --num-users 10

    2) 指定输出路径与随机种子：
       python -m cim.core.user_generator --num-users 50 --output ./data/raw/users_info_50.csv --seed 42
    """
    import asyncio

    parser = argparse.ArgumentParser(description="Generate user CSV in CIM format")
    parser.add_argument("--num-users", type=int, default=20, help="要生成的用户数量")
    parser.add_argument(
        "--output",
        type=str,
        default=_resolve_default_output(),
        help="输出CSV路径（默认: data/raw/users_info_new.csv）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")

    args = parser.parse_args()

    try:
        asyncio.run(_main_async(args.num_users, args.output, args.seed))
    except KeyboardInterrupt:
        logger.warning("中断执行")

