# AI-Trader 快速启动指南

## 1. 环境配置

### 1.1 创建环境变量文件
```bash
cp env.sh.template env.sh
```

### 1.2 编辑 env.sh，填入你的凭证
```bash
# 编辑文件
nano env.sh  # 或使用 vim/VSCode

# 必须填写：
# - TSL_USERNAME / TSL_PASSWORD (TinySoft 账号)
# - GEMINI_API_KEY (Google Gemini API Key)
```

### 1.3 加载环境变量
```bash
source env.sh
```

## 2. 测试连接

### 2.1 测试 TinySoft 连接
```bash
python test_tsl_connection.py
```
应该看到：
```
✓ Login successful!
✓ Query successful! Got X rows
```

### 2.2 测试后端 API
启动后端（在另一个终端）：
```bash
./start_backend.sh
```

测试 equity PnL endpoint：
```bash
curl -s 'http://localhost:8000/api/live/pnl-series?signature=gemini-2.5-flash&days=5&valuation=equity' | python -m json.tool
```

应该看到 `"valuation_used": "equity"` 而不是 `"cash"`。

## 3. 启动服务

### 3.1 后端（FastAPI）
```bash
# 方式1：使用启动脚本（推荐）
./start_backend.sh

# 方式2：直接运行
python -m uvicorn api_server:app --reload --port 8000
```

### 3.2 前端（Vite）
```bash
cd Tradingsimulation
npm run dev
```

访问: http://localhost:5173

### 3.3 多模型并发运行
- 同进程并发（简单）  
  在 `configs/default_config.json` 中将多个模型 `enabled: true`，然后：
  ```bash
  python3 main.py configs/default_config.json
  ```
  系统会在同一进程内同时运行所有已启用模型（asyncio.gather），并分别输出到 `data/agent_data/<signature>/{log,position}`。

- 多进程派发（推荐，隔离更强）  
  ```bash
  PARALLEL_RUN=true python3 main.py configs/default_config.json
  ```
  父进程会为每个模型启动子进程，并为其设置：
  - `ONLY_SIGNATURE=<signature>`（子进程仅跑该模型）
  - `RUNTIME_ENV_PATH=runtime_env_<signature>.json`（子进程独立运行态文件）
  日志查看：`logs/jobs/<signature-uuid>.log`。

- 只运行单一模型（过滤）：
  ```bash
  ONLY_SIGNATURE="gpt-5.1" python3 main.py configs/default_config.json
  ```

### 3.4 共享预抓快照自检
- 系统会把 10:30 / 11:30 / 14:00 三个决策点的行情、指标、新闻合并成共享快照，存储在 `data/agent_data/shared/snapshots/<date>/<time>_*.json`，并在 `data/agent_data/shared/logs/<date>/<time>.jsonl` 记录摘要。
- 建议在多模型实盘/回测前先手动“热身”一次，确保 TinySoft / AKShare 数据齐全：
  ```bash
  TODAY_DATE=2025-11-17 CURRENT_TIME="2025-11-17 10:30:00" \
    ./.venv/bin/python scripts/test_prefetch_all.py
  ```
  输出会列出每只白名单股票的新闻条数、是否拿到最新价格、是否成功计算指标，并提示快照是否在本次重新生成。

### 3.5 数据备份
- 每次 `python main.py ...` 或 `POST /api/run-trading` 前，都会自动运行 `scripts/backup_data.py`，生成 `backups/<timestamp>/backup.tar.gz` + `manifest.json`。
- 跳过自动备份：`SKIP_AUTO_BACKUP=true python main.py ...`
- 自定义保留数量：`AUTO_BACKUP_RETAIN=10 python main.py ...`
- 手动 CLI 备份：
  ```bash
  python scripts/backup_data.py --retain 10
  ```
- 后端 API 手动备份：`POST http://localhost:8000/api/backup?retain=10`

## 4. 触发 LLM 决策

有三种方式：

### 方式1：通过前端按钮（最简单）
1. 打开网页 http://localhost:5173
2. 进入"仪表盘"页面
3. 点击"开始决策"按钮
4. 查看实时日志和状态

### 方式2：通过 API 触发一次
```bash
# 运行完整日期范围（configs/default_config.json 中指定）
curl -X POST 'http://localhost:8000/api/run-trading' \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"configs/default_config.json"}'

# 查看任务状态
curl "http://localhost:8000/api/job/<job_id>"
```

### 方式3：启用每小时自动决策
```bash
# 在 env.sh 中添加：
export ENABLE_HOURLY_TRADING=true
export INIT_DATE=$(date +%Y-%m-%d)  # 只跑今天
export END_DATE=$(date +%Y-%m-%d)

# 重启后端
./start_backend.sh
```

### 方式4：强制重置并重跑单日
```bash
FORCE_REPLAY=true python main.py
```
- 或在 `configs/default_config.json` 的 `agent_config.force_replay` 中写 `true`。
- 系统会清空 `data/agent_data/<signature>/`，重新写入初始持仓与日志，再按 `date_range` 指定的日期跑 09:30 / 11:30 / 14:00 三段。

## 5. 查看决策结果

### 5.1 持仓记录
```bash
tail -f data/agent_data/gemini-2.5-flash/position/position.jsonl
```

### 5.2 日志
```bash
tail -f data/agent_data/gemini-2.5-flash/log/<DATE>/session_*.jsonl
```

### 5.3 前端仪表盘
- 实时收益曲线（Gemini 使用真实 equity PnL）
- 实时持仓
- 决策历史
- `/api/live/*` 会返回 `valuation_source`，前端会显示“小时快照 / 实时 / 均价”标签；`ModelDataProvider` 默认缓存最近 45 秒的数据，切换“AI 竞技场 / 总览”不再重复请求。

## 6. 故障排查

### TinySoft 连接失败
```bash
# 检查环境变量
echo $TSL_USERNAME
echo $TSL_PASSWORD

# 测试连接
python test_tsl_connection.py

# 如果提示 "Relogin refused"，运行：
python scripts/tsl_logout.py
```

### Equity PnL 仍显示 cash
```bash
# 确认后端已重启并加载了环境变量
pkill -f "uvicorn api_server"
./start_backend.sh

# 测试
curl 'http://localhost:8000/api/live/pnl-series?signature=gemini-2.5-flash&valuation=equity'
```
- 如果接口返回 `valuation_source=fallback`，说明目前只能依据均价估值，优先检查 `data/ai_stock_data.json` 是否最新或 TinySoft 是否登录成功。

### Job not found
- 确保后端正在运行
- Job ID 可能已过期（只在内存中保存）
- 重新触发一次决策

## 7. 配置说明

### configs/default_config.json
```json
{
  "date_range": {
    "init_date": "2025-10-30",  // 起始日期
    "end_date": "2025-11-08"     // 结束日期
  },
  "models": [
    {
      "name": "gemini-2.5-flash",
      "enabled": true,            // 启用 Gemini
      ...
    }
  ],
  "agent_config": {
    "initial_cash": 1000000.0    // 初始资金
  }
}
```

### 环境变量优先级
环境变量 > configs/default_config.json

例如：
```bash
export INIT_DATE="2025-11-13"
export END_DATE="2025-11-13"
# 将覆盖 config 中的 date_range
```

## 8. 开发提示

### 修改决策逻辑
编辑 `agent/base_agent/base_agent.py`:
- `run_trading_session()`: 单次决策逻辑
- `run_intraday_trading()`: 一天3次决策的时间点（09:30 / 11:30 / 14:00）
- `buy_stock()` / `sell_stock()`: 交易执行
- `get_hourly_stock_data()`: **唯一的行情入口**，一次返回最新摘要（最新价、上一根收盘、最近 N 根收盘）以及最近几根小时 K 线（默认 6 根）。如需更多柱子可调大 `lookback_hours`，但请避免频繁调用。
- `get_technical_indicators()`: 使用同一份小时线数据重新计算 SMA / MACD / RSI / BBands；`get_stock_price_data()` 已废弃。

### 修改 UI
- 仪表盘: `Tradingsimulation/src/components/sci/DashboardView.tsx`
- 决策控制: `Tradingsimulation/src/components/sci/TradingControl.tsx`

### 添加新模型
1. 在 `configs/default_config.json` 中添加模型配置
2. 在 `Tradingsimulation/src/utils/stockData.ts` 中添加 UI 配置
3. 重启服务

## 9. 架构概览

```
用户点击"开始决策" 
  → POST /api/run-trading
  → 后端启动 subprocess: python main.py
  → BaseAgent.initialize() (加载 Gemini + DataManager)
  → BaseAgent.run_date_range()
    → 对每个交易日：
      → run_intraday_trading() (3次决策：开盘 / 午间 / 午后)
        → run_trading_session()
          → LLM 调用工具获取数据（TinySoft实时+历史）
          → LLM 做出决策（buy/sell/no_trade）
          → 记录到 position.jsonl
  → 前端轮询 GET /api/job/{id} 查看状态
  → 前端从 GET /api/live/pnl-series 获取真实收益
```

## 10. 注意事项

1. **TinySoft 并发限制**：同一账号同时只能有一个活跃会话
2. **交易时间**：A股交易时间 9:30-15:00（午休 11:30-13:00）
3. **数据延迟**：TinySoft 数据可能有1-2分钟延迟
4. **API 配额**：Gemini API 有每日请求限制
5. **持仓文件**：不要手动编辑 `position.jsonl`，格式错误会导致程序崩溃

## 支持

- 检查日志: `logs/jobs/*.log`
- 查看持仓: `data/agent_data/<SIGNATURE>/position/position.jsonl`
- 测试脚本: `python test_tsl_connection.py`
