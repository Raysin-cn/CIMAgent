# CIMAgent 帖子生成器

这个模块提供了基于用户档案和话题数据生成个性化社交媒体帖子的功能。

## 功能特点

- 🎯 **个性化生成**: 基于用户档案（描述、活跃度、粉丝数等）生成符合用户特征的帖子
- 🗣️ **话题相关**: 支持指定话题，生成相关内容
- 🔄 **多样性**: 通过调整温度参数控制生成内容的多样性
- 📊 **批量生成**: 支持批量生成多个用户的帖子
- 💾 **多格式输出**: 支持JSON和CSV格式保存

## 文件结构

```
cim/
├── generate_public_posts.py    # 主要生成器类
└── README.md                   # 说明文档

scripts/
└── generate_posts_example.py   # 使用示例脚本

data/
├── users_info.csv              # 用户数据
├── topics.json                 # 话题数据
└── generated/                  # 生成的帖子输出目录
```

## 快速开始

### 1. 基本使用

```python
import asyncio
from cim.generate_public_posts import PostGenerator

async def main():
    # 初始化生成器
    generator = PostGenerator(temperature=0.9)
    
    # 加载数据
    users = generator.load_users_data("data/users_info.csv")
    topics = generator.load_topics_data("data/topics.json")
    
    # 生成帖子
    posts = await generator.generate_multiple_posts(
        user_ids=list(users.keys())[:5],  # 前5个用户
        topic_id="topic_1",               # 第一个话题
        num_posts=3                       # 生成3条帖子
    )
    
    # 保存结果
    generator.save_posts_to_json(posts, "output.json")

asyncio.run(main())
```

### 2. 使用命令行脚本

```bash
# 生成5条关于topic_1的帖子，使用10个用户
python scripts/generate_posts_example.py --topic topic_1 --num_posts 5 --users 10

# 生成10条关于topic_2的帖子，使用20个用户，温度设为0.8
python scripts/generate_posts_example.py --topic topic_2 --num_posts 10 --users 20 --temperature 0.8
```

## 参数说明

### PostGenerator 参数

- `temperature` (float): 模型温度参数，控制生成多样性
  - 0.0-0.3: 确定性高，内容相似
  - 0.5-0.7: 平衡的多样性
  - 0.8-1.0: 高多样性，创意丰富

### 命令行参数

- `--topic`: 话题ID (如 "topic_1", "topic_2")
- `--num_posts`: 要生成的帖子数量
- `--users`: 使用的用户数量
- `--temperature`: 模型温度参数

## 数据格式

### 用户数据 (users_info.csv)

包含以下字段：
- `user_id`: 用户ID
- `name`: 真实姓名
- `username`: 用户名
- `description`: 个人描述
- `followers_count`: 粉丝数
- `following_count`: 关注数
- `user_char`: 用户特征
- `activity_level`: 活跃度
- `activity_level_frequency`: 活动频率

### 话题数据 (topics.json)

```json
{
    "topic_1": {
        "title": "话题标题",
        "description": "话题描述",
        "keywords": ["关键词1", "关键词2"],
        "related_topics": ["相关话题1", "相关话题2"]
    }
}
```

### 输出格式

生成的帖子包含以下字段：
- `user_id`: 用户ID
- `username`: 用户名
- `content`: 帖子内容
- `topic`: 话题标题
- `timestamp`: 生成时间

## 可用话题

当前支持的话题：

1. **topic_1**: 中美贸易关税对经济转型的影响
2. **topic_2**: LLM代理模拟人类社交媒体行为的可信度
3. **topic_3**: AI艺术在传统艺术比赛中的准入问题
4. **topic_4**: 社交媒体平台政治广告禁令
5. **topic_5**: 学校用AI学习工具替代传统教科书
6. **topic_6**: 远程工作成为默认就业模式

## 注意事项

1. **API限制**: 生成过程中会添加延迟以避免API限制
2. **数据质量**: 确保用户数据和话题数据格式正确
3. **模型选择**: 当前使用GPT-4o-mini模型，可根据需要调整
4. **输出目录**: 生成的帖子会保存到 `data/generated/` 目录

## 错误处理

- 如果用户ID不存在，会跳过该用户
- 如果话题ID不存在，会显示可用的话题列表
- 网络错误或API错误会记录到控制台

## 扩展功能

可以通过以下方式扩展功能：

1. **添加新话题**: 在 `topics.json` 中添加新的话题数据
2. **自定义提示词**: 修改 `_create_user_prompt` 方法
3. **支持其他模型**: 修改 `ModelFactory.create` 的参数
4. **添加更多输出格式**: 扩展保存方法 