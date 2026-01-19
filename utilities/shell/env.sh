# ============ TinySoft 配置（必填）============
export TSL_USERNAME=
export TSL_PASSWORD=
export TSL_SERVER=tsl.tinysoft.com.cn
export TSL_PORT=443

# ============ Gemini API Key（必填）============
# 用于 gemini-3-pro-preview 等 Gemini 模型
# 获取方式: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY=

# ============ 可选：日期范围配置 ============
# 如果设置，会覆盖 settings/default_config.json 中的日期
# export INIT_DATE=2025-11-13
# export END_DATE=2025-11-13

export INIT_DATE=2026-01-12
export END_DATE=2026-02-12

# ============ 可选：每小时自动决策 ============
# export ENABLE_HOURLY_TRADING=true
# export TRADING_CONFIG_PATH=settings/default_config.json

# ============ OpenAI 兼容 API Key（必填）============
# 用于 claude-opus-4-5, gpt-5.2, deepseek-reasoner 等 OpenAI 兼容模型
# 注意：如果使用代理服务（如 api.openai-proxy.org），请填写代理服务的 API key
# 获取方式: 从您的代理服务提供商获取
export OPENAI_API_KEY=
export OPENAI_API_BASE=https://api.openai.com/v1

# ============ DashScope / Qwen API Key（必填，如果使用 qwen3-max）============
# 用于 qwen3-max 模型
# 获取方式: https://dashscope.console.aliyun.com/apiKey
# 注意：也可以使用 DASHSCOPE_API_KEY 环境变量
export QWEN_API_KEY=
export USE_LOCAL_DATA_ONLY=false

# ============ 多模型运行模式 ============
export PARALLEL_RUN=true

export REALTIME_MODE=wait

echo "✓ Environment variables loaded"
echo "  TSL_USERNAME: ${TSL_USERNAME:0:3}***"
echo "  GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}***"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}***"
echo "  QWEN_API_KEY: ${QWEN_API_KEY:0:10}***"
