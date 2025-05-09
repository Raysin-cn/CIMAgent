# CIMagent - Twitter社交网络模拟系统

CIMagent 是一个基于大语言模型的 Twitter 社交网络模拟系统，用于研究和分析社交网络中的用户行为和信息传播。

## 项目特点

- 基于大语言模型的用户行为模拟
- 支持多种社交网络行为（发帖、点赞、评论、转发等）
- 可配置的种子用户选择算法
- 自动数据备份和实验参数记录
- 灵活的环境配置和参数设置

## 安装要求

- Python 3.x
- VLLM
- pandas
- 其他依赖项（建议使用 requirements.txt 安装）

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/your-username/CIMagent.git
cd CIMagent
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据文件：
   - 用户信息文件 (CSV格式)
   - 话题数据文件 (CSV格式)

4. 运行模拟：
```bash
python main.py --model_path /path/to/model --db_path /path/to/database
```

## 参数说明

### 数据相关参数
- `--db_path`: 数据库文件保存路径（默认: data/CIM_experiments/twitter_simulation.db"）
- `--topic_file`: 话题数据文件路径（默认: "data/CIM_experiments/posts/posts_topic_3.csv"）
- `--users_file`: 用户数据文件路径（默认: "data/CIM_experiments/users_info.csv"）

### 模型相关参数
- `--model_path`: 模型路径（默认: "/path/to/models(eg.Qwen3-14B)"）
- `--model_url`: 模型服务URL（默认: "http://localhost:$port/v1"）
- `--max_tokens_1`: 第一个模型的最大token数（默认: 12000）
- `--max_tokens_2`: 第二个模型的最大token数（默认: 1000）

### 模拟相关参数
- `--total_steps`: 总模拟步数（默认: 72）
- `--backup_interval`: 数据库备份间隔步数（默认: 1）
- `--seed_rate`: 种子用户比例（默认: 0.1）
- `--seed_algo`: 种子用户选择算法（默认: "Random"）

## 数据备份

系统会自动在指定的备份目录中创建以下文件：
- 数据库备份文件：`twitter_simulation_{step}.db`
- 实验配置文件：`experiment_config.txt`

## 支持的用户行为

- 刷新（REFRESH）
- 搜索用户（SEARCH_USER）
- 搜索帖子（SEARCH_POSTS）
- 发帖（CREATE_POST）
- 点赞/取消点赞（LIKE_POST/UNLIKE_POST）
- 不喜欢/取消不喜欢（DISLIKE_POST/UNDO_DISLIKE_POST）
- 评论相关操作（CREATE_COMMENT, LIKE_COMMENT, UNLIKE_COMMENT等）
- 关注/取消关注（FOLLOW/UNFOLLOW）
- 静音/取消静音（MUTE/UNMUTE）
- 转发/引用（REPOST/QUOTE_POST）

## 注意事项

1. 确保模型服务已经正确启动并可访问
2. 数据文件格式需符合系统要求
3. 建议定期检查备份文件的完整性
4. 可以通过调整参数来优化模拟效果

## 许可证

Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. 