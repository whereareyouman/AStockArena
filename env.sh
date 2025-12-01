# AI-Trader 环境变量配置
# 使用方法: source env.sh （或复制到 ~/.zshrc）

# ============ TinySoft 配置（必填）============
export TSL_USERNAME="liuzeyu"
export TSL_PASSWORD="liuzeyu"
export TSL_SERVER="tsl.tinysoft.com.cn"
export TSL_PORT="443"

# ============ Gemini API Key（必填）============
export GEMINI_API_KEY='AIzaSyAkJoamC6Fc-yJVY5wAyk6k4hCLtfCXvAU'

# ============ 可选：日期范围配置 ============
# 如果设置，会覆盖 configs/default_config.json 中的日期
# export INIT_DATE='2025-11-13'
# export END_DATE='2025-11-13'

# ============ 可选：每小时自动决策 ============
# export ENABLE_HOURLY_TRADING=true
# export TRADING_CONFIG_PATH='configs/default_config.json'

# ============ 可选：OpenAI 配置 ============
# export OPENAI_API_KEY='your_openai_api_key'
# export OPENAI_API_BASE='https://api.openai.com/v1'

# ============ 可选：DashScope / Qwen ============
# export QWEN_API_KEY='your_dashscope_api_key'
# export DASHSCOPE_API_KEY='your_dashscope_api_key'
# export QWEN_API_BASE='https://dashscope.aliyuncs.com/compatible-mode/v1'
export QWEN_API_KEY='sk-ec68e918b17d4bb7be996ab91b5daf73'

# ============ 多模型运行模式 ============
export PARALLEL_RUN=true

echo "✓ Environment variables loaded"
echo "  TSL_USERNAME: ${TSL_USERNAME:0:3}***"
echo "  GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}***"
echo "  QWEN_API_KEY: ${QWEN_API_KEY:0:10}***"
