#!/bin/bash
# 完整的启动和验证脚本

echo "=== 启动脚本 ==="
echo ""

# 1. 检查环境变量
echo "1. 检查环境变量..."
ENV_OK=true

if [ -z "$TSL_USERNAME" ]; then
    echo "❌ TSL_USERNAME 未设置"
    ENV_OK=false
else
    echo "✓ TSL_USERNAME: ${TSL_USERNAME:0:3}***"
fi

if [ -z "$TSL_PASSWORD" ]; then
    echo "❌ TSL_PASSWORD 未设置"
    ENV_OK=false
else
    echo "✓ TSL_PASSWORD: ***"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY 未设置"
    ENV_OK=false
else
    echo "✓ GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}***"
fi

if [ "$ENV_OK" = false ]; then
    echo ""
    echo "请先设置环境变量："
    echo "  export TSL_USERNAME='你的用户名'"
    echo "  export TSL_PASSWORD='你的密码'"
    echo "  export GEMINI_API_KEY='你的API密钥'"
    echo ""
    echo "或者创建 env.sh 文件并 source 它："
    echo "  cp env.sh.template env.sh"
    echo "  # 编辑 env.sh 填入凭证"
    echo "  source env.sh"
    exit 1
fi

echo ""
echo "2. 测试 TinySoft 连接..."
python test_tsl_connection.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ TinySoft 连接测试失败，请检查凭证"
    exit 1
fi

echo ""
echo "3. 检查后端是否已在运行..."
if curl -s http://localhost:8000/api/jobs > /dev/null 2>&1; then
    echo "⚠️  后端已在运行，将先停止..."
    pkill -f "uvicorn api_server:app" || true
    sleep 2
fi

echo ""
echo "4. 启动后端..."
echo "   URL: http://localhost:8000"
echo "   按 Ctrl+C 停止"
echo ""

# 启动 uvicorn（前台运行，方便查看日志）
python -m uvicorn api_server:app --reload --port 8000
