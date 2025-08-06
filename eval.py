import argparse
import asyncio
import pandas as pd
from cim.core.stance_detector import StanceDetector
from datetime import datetime
import json

global_semaphore = asyncio.Semaphore(5)  # 全局最大并发5

def parse_args():
    parser = argparse.ArgumentParser(description="分析数据库中所有用户的立场随时间变化情况")
    parser.add_argument('--db_path', type=str, required=True, help='数据库文件路径')
    parser.add_argument('--output', type=str, default='stance_time_series.json', help='输出json文件路径')
    parser.add_argument('--csv', type=str, default='stance_time_series.csv', help='输出csv文件路径')
    parser.add_argument('--topic', type=str, default=None, help='检测主题（可选）')
    parser.add_argument('--max_users', type=int, default=None, help='最多分析多少个用户（可选）')
    return parser.parse_args()


def get_user_posts_by_time(detector, user_id):
    """
    获取用户所有原创帖，按时间升序排列
    """
    conn = detector._get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT post_id, content, created_at FROM post 
        WHERE user_id = ? AND original_post_id IS NULL
        ORDER BY created_at ASC
    """
    cursor.execute(query, (user_id,))
    posts = cursor.fetchall()
    conn.close()
    post_list = [
        {'post_id': p[0], 'content': p[1], 'created_at': p[2]} for p in posts
    ]
    return post_list


async def analyze_user_stance_time_series(detector, user_id, topic=None, max_concurrent=5):
    user_info = detector.get_user_info(user_id)
    if not user_info:
        return None
    posts = get_user_posts_by_time(detector, user_id)
    if not posts:
        return None
    time_series = [None] * len(posts)
    semaphore = asyncio.Semaphore(max_concurrent)
    async def stance_task(idx, post):
        async with global_semaphore:
            result = await detector.detect_stance_for_text(post['content'], topic=topic)
            time_series[idx] = {
                'post_id': post['post_id'],
                'created_at': post['created_at'],
                'stance': result['pred'],
                'confidence': result['confidence'],
                'text': post['content'],
                'reasonings': result['reasonings'],
            }
    tasks = [stance_task(idx, post) for idx, post in enumerate(posts)]
    await asyncio.gather(*tasks)
    return {
        'user_id': user_id,
        'user_name': user_info.get('user_name', ''),
        'name': user_info.get('name', ''),
        'time_series': time_series
    }


async def main():
    args = parse_args()
    detector = StanceDetector(db_path=args.db_path)
    users = detector.get_all_users_with_posts()
    if args.max_users:
        users = users[:args.max_users]
    all_results = []
    batch_size = 3  # 可根据实际情况调整batch大小
    for i in range(1, len(users), batch_size):
        batch_users = users[i:i+batch_size]
        print(f"处理用户batch: {i} ~ {i+len(batch_users)-1}")
        user_tasks = [
            asyncio.create_task(analyze_user_stance_time_series(detector, user_id, topic=args.topic))
            for user_id in batch_users
        ]
        user_results = await asyncio.gather(*user_tasks)
        for result in user_results:
            if result:
                all_results.append(result)
    # 保存为json
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    # 保存为csv（每行一个用户一条帖子的立场）
    rows = []
    for user in all_results:
        for item in user['time_series']:
            rows.append({
                'user_id': user['user_id'],
                'user_name': user['user_name'],
                'name': user['name'],
                'post_id': item['post_id'],
                'created_at': item['created_at'],
                'stance': item['stance'],
                'confidence': item['confidence'],
                'text': item['text'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False, encoding='utf-8')
    print(f"分析完成，结果已保存到 {args.output} 和 {args.csv}")

if __name__ == '__main__':
    asyncio.run(main())
