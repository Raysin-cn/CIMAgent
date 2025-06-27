import os
import sys
sys.path.append(os.getcwd())

import asyncio
from cim import OasisPostInjector, config
import json
import sqlite3

# 可配置参数
USER_ID = 1  # 你要分析的用户ID
STEPS = 10    # 推进的步数
USERS_CSV = config.config.post_generation.users_file
POSTS_JSON = config.config.get_file_path("processed", "generated_posts.json")
PROFILE_OUTPUT = config.config.get_file_path("processed", "oasis_user_profiles.csv")
DB_PATH = "./TEMP/test/simu.db"
save_dir = "./TEMP/test/"

async def main():
    print("单用户仿真推进脚本 - CIMAgent")
    print("=" * 60)
    # 1. 初始化注入器和数据
    injector = OasisPostInjector(db_path=DB_PATH)
    injector.load_users_data(USERS_CSV)
    injector.load_generated_posts(POSTS_JSON)
    profile_path = injector.create_user_profile_csv(PROFILE_OUTPUT)

    # 2. 运行模拟并推进若干步
    env = await injector.run_simulation_with_posts(
        profile_path=profile_path,
        posts=injector.generated_posts,
        num_steps=STEPS
    )

    # 3. 保存该用户的agent记忆（如有）
    agent = env.agent_graph.get_agent(USER_ID)
    save_dir = "./TEMP/test"
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(agent, "save_memory"):
        memory_path = os.path.join(save_dir, f"user_{USER_ID}_memory.json")
        agent.save_memory(memory_path)
        print(f"\n[用户Agent记忆已保存] {memory_path}")
    else:
        print(f"\n[警告] agent对象不支持save_memory方法")

    print("\n仿真推进完成！如需数据抽取请运行 extract_user_data.py 脚本。")



def extract_user_data():
    print("单用户数据抽取脚本 - CIMAgent")
    print("=" * 60)
    conn = sqlite3.connect(DB_PATH)
    db_cursor = conn.cursor()

    # 1. 用户profile
    db_cursor.execute("SELECT * FROM user WHERE user_id = ?", (USER_ID,))
    user_profile = db_cursor.fetchone()
    print("\n[用户Profile]")
    print(user_profile)
    with open(os.path.join(save_dir, f"user_{USER_ID}_profile.json"), "w", encoding="utf-8") as f:
        json.dump(user_profile, f, ensure_ascii=False, indent=2)

    # 2. 用户所有帖子
    db_cursor.execute("SELECT * FROM post WHERE user_id = ?", (USER_ID,))
    user_posts = db_cursor.fetchall()
    print("\n[用户所有帖子]")
    for post in user_posts:
        print(post)
    with open(os.path.join(save_dir, f"user_{USER_ID}_posts.json"), "w", encoding="utf-8") as f:
        json.dump(user_posts, f, ensure_ascii=False, indent=2)

    # 3. 用户所有行为轨迹
    db_cursor.execute("SELECT * FROM trace WHERE user_id = ?", (USER_ID,))
    user_traces = db_cursor.fetchall()
    print("\n[用户所有行为轨迹]")
    for trace in user_traces:
        print(trace)
    with open(os.path.join(save_dir, f"user_{USER_ID}_traces.json"), "w", encoding="utf-8") as f:
        json.dump(user_traces, f, ensure_ascii=False, indent=2)

    print("\n数据抽取完成！所有数据已保存到 ./TEMP/test 目录。")
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
    extract_user_data()
