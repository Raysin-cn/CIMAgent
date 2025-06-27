# CIM (Content Influence Modeling) 模块

CIM模块提供了社交媒体内容影响建模的完整解决方案，包括立场检测、帖子生成、数据管理和可视化分析等功能。

## 功能特点

- 🎯 **立场检测**: 基于LLM的社交媒体帖子立场识别和分析
- 🗣️ **帖子生成**: 基于用户档案和话题的个性化帖子生成
- 🔄 **数据注入**: 将生成的帖子注入到Oasis社交网络模拟框架
- 📊 **数据管理**: 数据库备份、恢复、验证和整理
- 📈 **可视化分析**: 立场分布、演化趋势和统计分析图表
- ⚙️ **配置管理**: 统一的配置系统，支持环境变量和灵活参数调整

## 模块结构

```
cim/
├── __init__.py                 # 模块初始化，导出主要类和函数
├── config.py                   # 配置管理模块
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── stance_detector.py      # 立场检测器
│   ├── post_generator.py       # 帖子生成器
│   └── data_injector.py        # 数据注入器
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── data_manager.py         # 数据管理器
│   └── visualizer.py           # 可视化工具
└── README.md                   # 说明文档
```

## 快速开始

### 1. 基本导入

```python
from cim import StanceDetector, PostGenerator, OasisPostInjector, DataManager, StanceVisualizer
from cim.config import config
```

### 2. 立场检测

```python
import asyncio

async def detect_stance():
    # 初始化检测器
    detector = StanceDetector()
    
    # 检测单个文本
    result = await detector.detect_stance_for_text(
        "这个政策对经济发展有积极影响", 
        topic="经济政策"
    )
    print(f"立场: {result['stance']}, 置信度: {result['confidence']}")
    
    # 检测所有用户的立场
    results = await detector.detect_stance_for_all_users(
        topic="中美贸易关税",
        post_limit=3
    )
    
    # 保存结果
    detector.save_stance_results(results, "stance_results.json")
    
    # 生成摘要
    summary = detector.generate_stance_summary(results)
    print(f"总用户数: {summary['total_users']}")

asyncio.run(detect_stance())
```

### 3. 帖子生成

```python
import asyncio

async def generate_posts():
    # 初始化生成器
    generator = PostGenerator()
    
    # 加载数据
    users = generator.load_users_data("data/users_info.csv")
    topics = generator.load_topics_data("data/topics.json")
    
    # 生成帖子
    posts = await generator.generate_multiple_posts(
        user_ids=list(users.keys())[:5],
        topic_id="topic_1",
        num_posts=2
    )
    
    # 保存结果
    generator.save_posts_to_json(posts, "generated_posts.json")
    generator.save_posts_to_csv(posts, "generated_posts.csv")

asyncio.run(generate_posts())
```

### 4. 数据注入和模拟

```python
import asyncio

async def run_simulation():
    # 初始化注入器
    injector = OasisPostInjector()
    
    # 加载数据
    injector.load_users_data("data/users_info.csv")
    injector.load_generated_posts("data/generated_posts.json")
    
    # 创建用户档案
    profile_path = injector.create_user_profile_csv("oasis_profiles.csv")
    
    # 运行模拟
    env = await injector.run_simulation_with_posts(
        profile_path=profile_pa