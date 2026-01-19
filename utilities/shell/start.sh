#!/bin/bash
# 启动主程序的脚本（自动使用虚拟环境）

# 确保在项目根目录
cd "$(dirname "$0")"

echo "🚀 Starting..."

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在，请先创建："
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# 使用虚拟环境的Python运行
echo "📦 使用虚拟环境: .venv"
.venv/bin/python3 main.py "$@"

echo ""
echo "✅ Completed!"
