# CIMAgent Main.py 使用说明

## 概述

重构后的 `main.py` 使用新的CIM模块配置，提供统一的配置管理和功能接口，支持Oasis社交网络模拟与匿名帖子注入。

## 主要改进

### 1. 统一配置管理
- 使用CIM模块的配置系统
- 支持环境变量配置
- 自动路径管理

### 2. 增强的数据管理
- 数据库备份和恢复
- 数据文件验证
- 临时文件清理
- 数据摘要报告

### 3. 完善的错误处理
- 统一的日志系统
- 详细的错误信息
- 调试模式支持

### 4. 新增功能参数
- `--backup`: 运行前备份数据库
- `--cleanup`: 运行后清理临时文件
- `--debug`: 启用调试模式

## 使用方法

### 基本用法

```bash
# 使用默认配置运行
python main.py

# 指定自定义参数
python main.py --users_csv data/raw/users.csv --posts_json data/processed/posts.json --steps 30

# 启用数据备份和清理
python main.py --backup --cleanup

# 调试模式
python main.py --debug
```

### 参数说明

#### 数据文件参数
- `--users_csv`: 用户数据CSV文件路径
  - 默认: `data/users_info.csv`
- `--posts_json`: 生成的帖子JSON文件路径
  - 默认: `data/processed/generated_posts.json`
- `--profile_output`: Oasis用户档案输出路径
  - 默认: `data/processed/oasis_user_profiles.csv`

#### 数据库参数
- `--db_path`: 模拟数据库路径
  - 默认: `./data/twitter_simulation.db`

#### 模拟参数
- `--steps`: 模拟步数（代理互动步数）
  - 默认: 20

#### 数据管理参数
- `--backup`: 在运行前备份数据库
- `--cleanup`: 运行后清理临时文件

#### 调试参数
- `--debug`: 启用调试模式

## 配置信息

程序启动时会显示当前配置信息：

```
CIMAgent 配置信息:
========================================
数据目录: /path/to/data
数据库路径: /path/to/data/twitter_simulation.db
模型平台: VLLM
模型类型: /data/model/Qwen3-14B
日志级别: INFO
调试模式: False
========================================
```

## 运行流程

1. **显示配置信息** - 显示当前CIM配置
2. **数据备份** (可选) - 备份现有数据库
3. **初始化注入器** - 创建帖子注入器实例
4. **验证数据文件** - 检查输入文件的有效性
5. **加载数据** - 加载用户数据和生成的帖子
6. **创建用户档案** - 生成Oasis格式的用户档案
7. **运行模拟** - 执行社交网络模拟并注入匿名帖子
8. **数据清理** (可选) - 清理临时文件
9. **生成报告** - 创建数据摘要报告

## 输出文件

### 主要输出
- **数据库文件**: 包含模拟结果的SQLite数据库
- **用户档案**: Oasis格式的用户档案CSV文件
- **数据摘要报告**: JSON格式的运行摘要

### 日志文件
- 程序运行日志保存在 `data/logs/` 目录

### 备份文件
- 数据库备份保存在 `data/backup/` 目录

## 错误处理

### 常见错误

1. **文件不存在错误**
   ```
   ❌ 文件不存在: 用户数据文件不存在或无效: data/users_info.csv
   ```
   - 检查文件路径是否正确
   - 确保文件存在且可读

2. **数据库连接错误**
   ```
   ❌ 数据库连接失败: [Errno 2] No such file or directory
   ```
   - 检查数据库路径
   - 确保目录存在

3. **模型初始化错误**
   ```
   ❌ 模型初始化失败: Connection refused
   ```
   - 检查模型服务器是否运行
   - 验证模型配置参数

### 调试模式

启用调试模式可以获取更详细的错误信息：

```bash
python main.py --debug
```

调试模式会：
- 显示详细的日志信息
- 输出完整的错误堆栈
- 启用更详细的调试输出

## 环境变量配置

可以通过环境变量覆盖默认配置：

```bash
# 设置模型配置
export CIM_MODEL_PLATFORM=VLLM
export CIM_MODEL_TYPE=/data/model/Qwen3-14B
export CIM_MODEL_URL=http://localhost:12345/v1

# 设置数据库路径
export CIM_DB_PATH=./data/my_simulation.db

# 启用调试模式
export CIM_DEBUG=true

# 运行程序
python main.py
```

## 示例脚本

### 完整运行示例

```bash
#!/bin/bash

# 设置环境变量
export CIM_DEBUG=true

# 运行模拟
python main.py \
    --users_csv data/raw/users_info.csv \
    --posts_json data/processed/generated_posts.json \
    --steps 30 \
    --backup \
    --cleanup \
    --debug

echo "模拟完成！"
```

### 批量处理示例

```bash
#!/bin/bash

# 处理多个数据集
for dataset in dataset1 dataset2 dataset3; do
    echo "处理数据集: $dataset"
    
    python main.py \
        --users_csv "data/raw/${dataset}_users.csv" \
        --posts_json "data/processed/${dataset}_posts.json" \
        --db_path "data/output/${dataset}_simulation.db" \
        --backup \
        --cleanup
    
    echo "数据集 $dataset 处理完成"
done
```

## 注意事项

1. **数据文件格式**: 确保输入文件符合预期格式
2. **模型服务**: 确保模型服务器正在运行
3. **磁盘空间**: 确保有足够的磁盘空间存储结果
4. **网络连接**: 如果使用远程模型，确保网络连接稳定
5. **权限**: 确保程序有读写数据目录的权限

## 故障排除

### 性能优化
- 调整 `--steps` 参数控制模拟步数
- 使用 `--cleanup` 清理临时文件
- 定期清理备份文件

### 内存管理
- 监控内存使用情况
- 适当减少并发处理数量
- 及时清理不需要的数据

### 日志管理
- 定期清理日志文件
- 使用适当的日志级别
- 监控错误日志 