from tqdm import tqdm
import asyncio
import os
import pandas as pd
from string import Template
import json
import logging
import argparse

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import OpenAIMessage
import sqlite3

import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='社交媒体态度分析工具')
    parser.add_argument('--topic_index', type=int, default=4,
                      help='要分析的主题索引')
    parser.add_argument('--topics_path', type=str, default='data/CIM_experiments/topics.json',
                      help='主题文件路径')
    parser.add_argument('--db_dir', type=str, default='experiments/topic_4_Qwen3-14B_True_20250516_115712/backups',
                      help='数据库文件目录')
    parser.add_argument('--start_index', type=int, default=0,
                      help='起始数据库文件索引')
    parser.add_argument('--end_index', type=int, default=10,
                      help='结束数据库文件索引')
    parser.add_argument('--output_dir', type=str, default='experiments/topic_4_Qwen3-14B_True_20250516_115712',
                      help='输出目录')
    return parser.parse_args()

attitude_analysis_template = Template(
    "You are a social media analysis expert. You are given a post and you need to analyze the attitude of the post based on the topic. "
    "Here is the topic of the post: $topic\n"
    "Here is the information of the post you are interested in:\n"
    "Post Information:\n"
    "Content: $post_content\n"
    "You only answer the stance of the post, not any other information."
    "You should answer in the following format. The stance should be a single word (pro, con, neutral), don't include any other information: \n"
    "Stance: <post_stance>"
)

vllm_model = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    # model_type="/home/models/Qwen3-8B",
    model_type='/data/model/Qwen3-14B',
    url="http://0.0.0.0:12345/v1",
    model_config_dict={"max_tokens": 16000}
)

async def attitude_ONE(user_post:str, topic:str):
    try:
        # 通过llm_model分析posts_list中每个post的attitude
        post_msg = attitude_analysis_template.substitute(topic=topic, post_content=user_post)
        # 将文本消息转换为OpenAI格式的消息列表
        messages = [{"role": "user", "content": post_msg}]
        # 直接使用消息列表,不需要转换为OpenAIMessage
        post_attitude = await vllm_model.arun(messages)
        return post_attitude
    except Exception as e:
        logging.error(f"Error in attitude_ONE: {str(e)}")
        return None

async def main():
    try:
        args = parse_args()
        logging.info("Starting attitude analysis...")
        
        topic = json.load(open(args.topics_path))[f"topic_{args.topic_index}"]['title']
        logging.info(f"Analyzing topic: {topic}")

        db_file_list = sorted(os.listdir(args.db_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        db_file_list = db_file_list[args.start_index:args.end_index]
        logging.info(f"Found {len(db_file_list)} database files")

        # 读取用户数据
        db_file_start = db_file_list[0]
        conn = sqlite3.connect(os.path.join(args.db_dir, db_file_start))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user")
        users_data = cursor.fetchall()
        users_df = pd.DataFrame(users_data, columns=[i[0] for i in cursor.description])
        users_nums = len(users_df)
        logging.info(f"Found {users_nums} users")
        conn.close()

        users_attitude_list = []
        total_tasks = 0

        # 使用tqdm显示数据库文件处理进度
        for db_file in tqdm(db_file_list, desc="Processing database files"):
            db_path = os.path.join(args.db_dir, db_file)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM post")
            posts_data = cursor.fetchall()
            posts_df = pd.DataFrame(posts_data, columns=[i[0] for i in cursor.description])
            
            for user_id in users_df['user_id']:
                user_posts = posts_df[posts_df['user_id'] == user_id]
                if not user_posts.empty:
                    last_post_content = user_posts.iloc[-1]['content']
                    users_attitude_list.append(attitude_ONE(last_post_content, topic))
                    total_tasks += 1
            conn.close()

        logging.info(f"Created {total_tasks} analysis tasks")

        users_attitude_list_results = []
        # 使用tqdm显示任务完成进度
        with tqdm(total=total_tasks, desc="Analyzing posts") as pbar:
            for i in range(0, total_tasks, users_nums):  # 每次处理users_nums个任务
                batch = users_attitude_list[i:i+users_nums]
                results = await asyncio.gather(*batch)
                pbar.update(len(batch))
                users_attitude_list_results.extend(results)

        msg_results = []
        for response_result in users_attitude_list_results:
            msg_result = response_result.choices[0].message.content
            stance_match = re.search(r'Stance:\s*(\w+)', msg_result)
            if stance_match:
                stance = stance_match.group(1).lower()
            else:
                stance = "unknown"
            
            msg_results.append(stance)

        users_attitude_per_step = []
        for step_index in range(len(db_file_list)):
            users_attitude_per_step.append(msg_results[step_index*users_nums:(step_index+1)*users_nums])

        
        logging.info("Analysis completed")
        for step_index in range(len(db_file_list)):
            print(f"Step {step_index+1}:")
            print(users_attitude_per_step[step_index])
            print("\n")

        # 将结果保存到csv文件
        result_df = pd.DataFrame(users_attitude_per_step)
        result_df = result_df.T
        result_df.columns = [str(i) for i in range(len(result_df.columns))]
        result_df.index.name = 'user_id'
        output_path = os.path.join(args.output_dir, 'attitude_analysis_results.csv')
        result_df.to_csv(output_path, index=True)
        logging.info(f"Results saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

