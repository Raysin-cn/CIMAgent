# 使用curl验证大模型是否可以调用
echo "正在验证大模型连接..."
curl -X POST http://localhost:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/model/Qwen3-14B",
    "messages": [
      {
        "role": "user",
        "content": "你好，请简单介绍一下自己"
      }
    ],
    "max_tokens": 100
  }'

# 检查curl命令的退出状态
if [ $? -eq 0 ]; then
    echo "✅ 大模型连接验证成功"
else
    echo "❌ 大模型连接验证失败"
fi
