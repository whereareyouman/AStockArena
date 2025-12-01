# AI-Trader 使用手册

> 说明整体环境、运行流程、共享预抓、备份以及 LLM API 现状，便于快速上手多模型 A 股日内交易 Agent。

---

## 1. 项目概览

- **定位**：服务科创板 10 支白名单股票（`SH688008/688111/688009/688981/688256/688271/688047/688617/688303/688180`）的多模型 A 股日内交易 Agent。
- **决策节奏**：每天固定 `10:30 / 11:30 / 14:00` 三次；10:30 起步可确保 TinySoft 有完整小时线（避免 09:30 虚拟数据）。
- **核心特色**
  - SharedPrefetch：所有模型共用一次性行情/新闻/指标快照，避免并发抓数与 token 暴涨。
  - 多进程隔离：`PARALLEL_RUN=true` 时父进程按模型派生子进程，日志写入 `logs/jobs/<signature-uuid>.log`。
  - 回测/实时统一：`main.py` 读取 `configs/default_config.json` 或环境变量区间，既可多日重放也可当天实盘。
- **重要目录**
  - `main.py`（统一入口） · `agent/`（BaseAgent 与模型逻辑） · `agent/shared_prefetch.py`（共享快照）  
  - `api_server.py`（FastAPI 后端） · `Tradingsimulation/`（Vite 前端） · `scripts/`（工具、备份、检测）

---

## 2. 环境与快速启动

### 2.1 创建并填充 env

```bash
cp env.sh.template env.sh
```

编辑 `env.sh`（推荐 VSCode / nano）：
- **必填凭证**
  - `TSL_USERNAME` / `TSL_PASSWORD`（TinySoft）
  - `GEMINI_API_KEY`
  - `OPENAI_API_KEY` + `OPENAI_API_BASE=https://api.openai-proxy.org/v1`（必须含 `/v1`）
- **Qwen3-235B（DashScope OpenAI 模式）**
  - 需单独配置 `QWEN_API_KEY` 或 `DASHSCOPE_API_KEY`，并推荐使用 **Python 3.12+**（DashScope SDK 对低版本支持较差）。
  - 没有额外代理服务时，可把 `OPENAI_BASE_URL` 留空，BaseAgent 会默认访问 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
  - 如需单独跑该模型，请新建 `runtime_env_qwen3-235b.json` 并在 `env.sh` 导出专属凭证。
- **可选**
  - `ANTHROPIC_API_KEY` / `ANTHROPIC_API_BASE=https://api.openai-proxy.org/anthropic`
  - `ENABLE_HOURLY_TRADING`、`INIT_DATE/END_DATE` 等
- `PARALLEL_RUN=true` 已预置，使 `python3 main.py` 自动采用多进程。

加载变量：
```bash
source env.sh
```

### 2.2 Python / Node 依赖

```bash
# Python 3.12 推荐
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 前端依赖
cd Tradingsimulation
npm install
```

### 2.3 连接自检

```bash
# TinySoft
python test_tsl_connection.py
# 期望输出 Login successful + Query successful

# 后端（另开终端）
./start_backend.sh   # 或 python -m uvicorn api_server:app --reload --port 8000

# PnL API
curl -s 'http://localhost:8000/api/live/pnl-series?signature=gemini-2.5-flash&days=5&valuation=equity' | python -m json.tool
```

### 2.4 启动顺序

```bash
# 共享快照热身（可选，确保 TinySoft/AKShare 正常）
TODAY_DATE=2025-11-17 CURRENT_TIME="2025-11-17 10:30:00" \
  ./.venv/bin/python scripts/test_prefetch_all.py

# 后端 API
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# 主流程（自动读取 configs/default_config.json 并按 PARALLEL_RUN 启动多模型）
python3 main.py

# 前端
cd Tradingsimulation && npm run dev
```

访问 `http://localhost:5173`，点击 “Start Trading” 将触发 `/api/job/start`，由后端启动 `main.py` 子进程。
后端若直接访问 `http://127.0.0.1:8000/`，现在会返回 “服务就绪” 提示（此前默认 404）；所有实际 API 仍在 `/api/*`。

### 2.5 多模型运行模式

- **同进程（调试）**
  ```bash
  PARALLEL_RUN=false python3 main.py configs/default_config.json
  ```
- **多进程（默认推荐）**
  ```bash
  python3 main.py configs/default_config.json   # env 中已设 PARALLEL_RUN=true
  ```
  父进程会为每个启用模型设置 `ONLY_SIGNATURE` 与 `RUNTIME_ENV_PATH=runtime_env_<signature>.json`。
- **仅跑某模型**
  ```bash
  ONLY_SIGNATURE="gpt-5.1" python3 main.py configs/default_config.json
  ```

### 2.6 共享预抓快照自检

- 三个决策点只抓一次行情（TinySoft）、新闻（AKShare）、指标（内部计算），写入 `data/agent_data/shared/snapshots/<date>/<time>_<signature>.json`。
- 摘要日志：`data/agent_data/shared/logs/<date>/<time>.jsonl`，包含每只股票收盘价、涨跌幅、新闻数量、指标状态。
- 推荐在拉多模型前执行 `scripts/test_prefetch_all.py`（见 2.4）完成“热身”。

### 2.7 数据备份

- 任意 `python main.py ...` 或 `POST /api/run-trading` 前都会自动运行 `scripts/backup_data.py`，产出 `backups/<timestamp>/backup.tar.gz + manifest.json`。
- 并行安全：若目录重名会自动重试并附加微秒后缀。
- 常用参数：
  - 跳过：`SKIP_AUTO_BACKUP=true python main.py ...`
  - 保留数量：`AUTO_BACKUP_RETAIN=10 python main.py ...`
  - 手动：`python scripts/backup_data.py --retain 10`
  - API：`POST http://localhost:8000/api/backup?retain=10`

### 2.8 触发 LLM 决策

1. **前端按钮**：`/dashboard` → “开始决策”。
2. **API**
   ```bash
   curl -X POST 'http://localhost:8000/api/run-trading' \
     -H 'Content-Type: application/json' \
     -d '{"config_path":"configs/default_config.json"}'
   curl "http://localhost:8000/api/job/<job_id>"
   ```
3. **自动定时**：`env.sh` 中开启 `ENABLE_HOURLY_TRADING` 并设置当日 `INIT_DATE/END_DATE`，重启后端即可。
4. **强制单日重放**
   ```bash
   FORCE_REPLAY=true python3 main.py
   ```
   或在配置中写 `agent_config.force_replay=true`，会清空 `data/agent_data/<signature>/` 后按 10:30/11:30/14:00 重新执行。

### 2.9 查看结果

- 持仓：`tail -f data/agent_data/<signature>/position/position.jsonl`
- 决策日志：`tail -f data/agent_data/<signature>/logs/decision.log`
- 多进程 STDOUT：`tail -f logs/jobs/<signature-uuid>.log`
- 前端仪表：收益曲线（equity PnL）、持仓、决策历史、Live News

### 2.10 常见命令 & 排障

- 停止所有回测：`pkill -f "python main.py"`（匹配 `python3` 命令行）
- TinySoft relogin：`python scripts/tsl_logout.py`
- 确认 OpenAI 变量：`printenv | grep OPENAI`
- 若 Equity PnL 仍显示 cash：重启后端后再次请求 `api/live/pnl-series?...valuation=equity`

### 2.11 估值优先级 & 前端缓存

- `/api/live/model-stats`、`/api/live/current-positions` 统一按照 **共享快照小时线 → TinySoft 实时 → 均价兜底** 的顺序估值，并在响应中返回 `valuation_source` 字段，便于前端或排障时提示真实来源。
- 若三个阶段都不可用（例如 TinySoft 未配置），`valuation_source=fallback`，前端会给出“成本估值”提示。
- 前端通过 `ModelDataProvider` 缓存最近 ~45 秒的 `/api/live/*` 数据，切换“AI 竞技场/首页总览”不会重复发起全量请求；强制刷新或等待下一次轮询即可更新。

---

## 3. 当前进展与待办

- ✅ 多模型并行：Claude Sonnet 4.5、GPT-5.1、DeepSeek Reasoner、Gemini 2.5 Flash 已启用。
- ✅ 共享预抓：`SharedPrefetchCoordinator.ensure_snapshot` 运行稳定，失败时自动回退到单 Agent 抓数并在日志标注。
- ✅ 技术指标限窗：`get_technical_indicators` 只读取最近 40 根小时线（≈10 天），将单轮 token 限制在安全范围。
- ✅ 备份鲁棒性：`scripts/backup_data.py` 加入重试/等待与微秒后缀，解决多进程冲突。
- ✅ 收益估值：后端 API 统一按照 “共享快照（小时）→ TinySoft 实时 → 均价兜底” 三段式估值，并通过 `valuation_source` 提示前端真实来源。
- ✅ 前端缓存：`ModelDataProvider` 会缓存最近 45 秒的 `/api/live/*` 响应，切换“AI 竞技场 / 仪表盘”不再触发全量刷新。
- ⏳ **Qwen3-235B（DashScope）**：需配置 `QWEN_API_KEY`/`DASHSCOPE_API_KEY`，并生成 `runtime_env_qwen3-235b.json`；默认保持 `enabled:false`，待密钥就绪后再开启。
- ⏳ **前后端联调回归**：需重新跑一次端到端冒烟，确认 `/api/run-trading` → 子进程 → 前端三视图均正常。

---

## 4. 数据与模型管线

- **DataManager**
  - `get_hourly_stock_data(symbol, current_time, lookback_hours=6)`：唯一行情入口，返回最新 summary + mini candles。
  - `get_technical_indicators(symbol, today_date)`：先裁剪历史到 40 根，再统一计算 SMA / MACD / RSI / BBands 并写入 `data/ai_stock_data.json`。
- **SharedPrefetch (`agent/shared_prefetch.py`)**
  - `_FileLock` 保障同一 `(date, time, symbols_signature)` 只有一个进程构建。
  - 快照路径 `data/agent_data/shared/snapshots/<date>/<time>_<signature>.json`，日志 `.../logs/<date>/<time>.jsonl`。
  - 若 SharedPrefetch 不可用，BaseAgent 会 fallback 成本地抓取，同时记录警告。
- **BaseAgent**
  - 风控：20% 单票上限、100 股整数倍、T+1。
  - 决策日志：`Shared snapshot [...]` 行标记所用快照 ID、路径与创建者。
  - 默认股票池在 `agent/base_agent/base_agent.py`、`prompts/agent_prompt.py`、`data/data_pipeline.py`、`Tradingsimulation/src/utils/stockData.ts` 等处保持一致。
- **模型配置 (`configs/default_config.json`)**
  - `models[]` 数组可单独指定 `openai_base_url/api_key/safety_settings`。
  - 新模型需同步创建 `runtime_env_<signature>.json` 并在前端 `stockData.ts` 标注。

---

## 5. 运营与排障

- **备份级别**：出厂即启用；可通过 `SKIP_AUTO_BACKUP=true` 临时关闭。`manifest.json` 提供 SHA-256 校验，可用 `shasum -a 256` 验证。
- **日志定位**
  - 共享快照：`data/agent_data/shared/logs`
  - 模型决策：`data/agent_data/<signature>/logs/decision.log`
  - 子进程 STDOUT：`logs/jobs/<signature-uuid>.log`
  - Uvicorn：终端输出或 `start_backend.sh` 的日志
- **常见问题**
  - TinySoft 登录冲突：同账号只能单会话，需 `python scripts/tsl_logout.py` 释放后重试。
  - LLM 工具报 “请提供 6 位 A 股代码”：通常是模型输出 `SH688008` 被拒，非系统 bug，可在 prompt 中强调。
  - Equity PnL 未更新：确保 `api/live/pnl-series?valuation=equity` 被调用，或重启后端加载最新 env。

---

## 6. LLM API 现状与扩展路径

> 摘自 `README_LLM_API_ISSUE.md`，并结合当前实现进行更新。

1. **当前状态**
   - `api_server.py` 现已提供 `/api/llm/ping` 与 `/api/llm/ask`，内部会复用 `BaseAgent` 的模型配置并缓存会话。
   - `/api/run-trading` 依旧负责触发多模型子进程；`/api/llm/ask` 只做对话/策略问答，不执行交易。
2. **调用与排查步骤**
   1. `GET /api/llm/ping` → 查看可用 `signature`、当前缓存数量。
   2. `POST /api/llm/ask` → 若首次调用会自动初始化该 `signature` 的 `BaseAgent` 并建立会话。
   3. `curl 127.0.0.1:8000/api/jobs` → 仍可用来确认后端服务/子进程健康。
   4. 检查 `OPENAI_API_KEY`、`GEMINI_API_KEY`、`PARALLEL_RUN` 等环境变量是否在 `uvicorn` 进程内生效。
   5. 若请求卡住，确认 `logs/jobs/*.log` 里模型是否报错、或 `logs/uvicorn.log` 是否有 Traceback。
   6. 前端若 404，请核对域名、反向代理及 CORS 列表。
3. **请求示例**
   ```bash
   curl -X POST http://127.0.0.1:8000/api/llm/ask \
     -H "Content-Type: application/json" \
     -d '{
           "signature": "gpt-5.1",
           "prompt": "请总结当前持仓风险。",
           "system_prompt": "你是盘后分析助手，只输出中文要点。",
           "reset": false
         }'
   ```
   - 可选字段：`config_path`（默认 `configs/default_config.json`）、`system_prompt`（覆盖默认指令）、`reset`（重置上下文后再回答）。
   - 返回字段包括 `response`、`model`、`history_length`、`usage` 等，方便前端展示与计费。
4. **实现细节速记**
   - 全局缓存 `LLM_SESSIONS: dict[str, LLMSession]`，每个会话持有 `BaseAgent`、LangChain `Chat*` 模型以及消息历史。
   - 会话首次创建时会自动执行 `agent.initialize()`，并将默认 system prompt 写入历史；后续请求仅追加消息。
   - `reset=true` 会保留 Agent，但清空历史，便于发起新的话题。
5. **安全建议**
   - 建议在 Nginx / API 网关层追加 Token、IP 白名单与限流。
   - 对 `prompt`、`response` 做长度限制和审计，避免模型被滥用为通用对话接口。
   - 如需长连线，可把 `/api/llm/ask` 包装到队列/任务系统中，防止阻塞主事件循环。
6. **执行清单示例**
   ```
   [ ] SSH 登录 & 激活 venv
   [ ] uvicorn 监听 8000
   [ ] curl /api/llm/ping 成功
   [ ] curl /api/llm/ask 成功返回
   [ ] 前端改调 /api/llm/ask
   [ ] 验证 CORS / Nginx
   [ ] 补充限流与鉴权
   [ ] 保持 /api/run-trading 流程能同时使用
   ```

---

## 7. 后续可做的工作提示

- **接入 Qwen3-235B**
  - 在 `configs/default_config.json` 中启用 `qwen3-235b`，如需自定义代理可改写 `openai_base_url`。
  - 创建 `runtime_env_qwen3-235b.json` 并在 `env.sh` 设置 `QWEN_API_KEY` / `DASHSCOPE_API_KEY`。
  - 确认运行节点使用 Python 3.12+，再用 `ONLY_SIGNATURE="qwen3-235b" python3 main.py ...` 做一轮验证。
- **前后端联调冒烟**
  - 同时启动 `uvicorn api_server:app`、`python3 main.py`、`npm run dev`。
  - 在前端点击 “Start Trading”，观察 `/api/job/start` 是否拉起子进程并写日志；核实行情表、模型卡片、Live News 正常刷新。
- **LLM API 扩展**
  - 若需增强鉴权、拆分多租户或新增更细的路由，可在现有 `/api/llm/ask` 基础上按照第 6 节的建议继续扩展。
- **文档同步**
  - 当配置或流程有所调整时，请同时更新本 README、`QUICKSTART.md`、`LLM_DECISION_CONTROL.md`。

---

## 8. 参考资料

- `QUICKSTART.md`：更细的环境/脚本说明
- `LLM_DECISION_CONTROL.md`：决策触发流程、T+1 管控
- `DATA_STORAGE.md`：持仓、快照、日志格式
- `scripts/test_prefetch_all.py`：共享快照热身
- `scripts/backup_data.py`：备份实现与参数
- `test_tsl_connection.py`：TinySoft 连通性

如遇文档未覆盖的问题，可在项目 Issue 中记录或与维护者沟通。

