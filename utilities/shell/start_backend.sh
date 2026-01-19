#!/bin/bash
# 启动后端服务的脚本
# 使用方法: ./start_backend.sh

echo "=== Backend Startup ==="
echo ""

# 检查环境变量
check_env() {
    local var_name=$1
    local var_value=${!var_name}
    if [ -z "$var_value" ]; then
        echo "❌ $var_name not set"
        return 1
    else
        echo "✓ $var_name is set"
        return 0
    fi
}

echo "Checking environment variables..."
check_env "TSL_USERNAME" || { echo "Please set: export TSL_USERNAME='your_username'"; exit 1; }
check_env "TSL_PASSWORD" || { echo "Please set: export TSL_PASSWORD='your_password'"; exit 1; }
check_env "GEMINI_API_KEY" || { echo "Please set: export GEMINI_API_KEY='your_api_key'"; exit 1; }

echo ""
echo "TSL_SERVER: ${TSL_SERVER:-tsl.tinysoft.com.cn}"
echo "TSL_PORT: ${TSL_PORT:-443}"
echo ""

# 可选：设置今天作为交易日期
# export INIT_DATE=$(date +%Y-%m-%d)
# export END_DATE=$(date +%Y-%m-%d)


echo "Starting backend on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# 尝试激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 启动 uvicorn
if command -v python3 &> /dev/null; then
    python3 -m uvicorn api_server:app --reload --port 8000
elif command -v python &> /dev/null; then
    python -m uvicorn api_server:app --reload --port 8000
else
    echo "❌ Python not found. Please install Python 3."
    exit 1
fi
