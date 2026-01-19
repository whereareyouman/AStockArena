# 📊 Data Storage Documentation

> **AStock Arena** - Multi-Model Trading System Data Management Guide

---

## 📁 Data Storage Locations

### 1. Portfolio Position Data

**Location:** `data_flow/agent_data/{model_name}/position/position.jsonl`

**Format:** JSONL (one JSON object per line)

**Description:** Complete portfolio snapshots after each trading decision

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique sequential identifier (auto-incremented) |
| `date` | string | Trading date (YYYY-MM-DD) |
| `decision_time` | string | Decision timestamp (normalized format) |
| `decision_count` | integer | Sequential decision number |
| `positions` | object | Holdings object containing `CASH` (float) and stock positions with `shares`, `avg_price`, `purchase_date` |
| `this_action` | object | Current decision action (`action`, `symbol`, `amount`) |

**Note:** 
- Cash balance is stored in `positions.CASH`, not as a top-level `cash` field
- Total equity is calculated as `positions.CASH + sum(holdings market value)`, not stored directly

**Example Paths:**
```
data_flow/agent_data/gemini-2.5-flash/position/position.jsonl
data_flow/agent_data/deepseek-reasoner/position/position.jsonl
data_flow/agent_data/claude-haiku-4-5/position/position.jsonl
```

**Use Cases:**
- ✅ Calculate cumulative returns
- ✅ Compute Sharpe ratio
- ✅ Calculate maximum drawdown
- ✅ Generate equity curve charts
- ✅ Display recent decision stream

---

### 2. Decision Logs

**Location:** `data_flow/agent_data/{model_name}/log/{date}/{sanitized_time}.jsonl`

**Format:** JSONL (one JSON object per line)

**Description:** Complete LLM decision-making process and reasoning

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Decision timestamp (ISO 8601) |
| `signature` | string | Model identifier |
| `decision_time` | string | Trading decision time |
| `decision_count` | integer | Sequential number |
| `new_messages` | array | Conversation messages (system/user/assistant) |

**File Naming:**
- The filename format is `{sanitized_time}.jsonl` where `sanitized_time` is `decision_time` with colons and spaces replaced by hyphens and underscores
- Example: `10-30-00.jsonl` (not `2026-01-12_10-30-00.jsonl` since the date is already in the directory path)

**Example Paths:**
```
data_flow/agent_data/gemini-2.5-flash/log/2026-01-12/10-30-00.jsonl
data_flow/agent_data/gpt-5.1/log/2026-01-12/14-00-00.jsonl
```

**Use Cases:**
- 🔍 Analyze AI decision quality
- 📝 Audit trading records
- 🐛 Debug model behavior
- 🔄 Trace decision reasoning

---

### 3. News Data

**Location:** `data_flow/news.csv`

**Format:** CSV (Comma-Separated Values)

**Description:** Aggregated stock market news from AKShare

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `title` | string | News headline |
| `content` | string | Full article or summary |
| `publish_time` | datetime | Publication timestamp |
| `source` | string | News source (East Money, Sina Finance, etc.) |
| `url` | string | Original article URL |
| `symbol` | string | Related stock symbol |
| `query` | string | Search query used to fetch this news |
| `search_time` | datetime | Timestamp when the news was fetched |

**Data Source:** 
- AKShare Python library (`akshare.stock_news_em()`)
- Daily updates via `data_flow/data_pipeline.py`

**Use Cases:**
- 📰 Frontend news panel display
- 🤖 LLM decision-making context
- 📈 Sentiment analysis input

---

## 🔄 Operation Modes Comparison

### Backtest Mode (Default)

**Configuration:** Set date range in `settings/default_config.json` (or `configs/default_config.json`)

```json
{
  "init_date": "2025-10-30",
  "end_date": "2025-11-08",
  "backtest": true
}
```

**Data Sources:**
- 📊 TinySoft historical data (pyTSL)
- 📰 Saved historical news (news.csv)

**Execution Flow:**
```
1. Start from init_date → Process each trading day sequentially
2. Retrieve historical market data for the day
3. LLM makes decisions based on historical information
4. Simulate trade execution and update positions
5. Record to position.jsonl
6. Continue to next trading day until end_date
```

**Advantages:**
- ⚡ Fast strategy validation
- 🔁 Reproducible testing
- 🔌 No real-time market connection required
- 🤝 Parallel testing of multiple AI models

**Status Check:**
```bash
python main.py
# Output: "No trading days to process"
# Meaning: Backtest completed for 2025-10-30 to 2025-11-08
```

---

### Live Trading Mode

**Configuration Method 1:** Override dates via environment variables

```bash
export INIT_DATE=$(date +%Y-%m-%d)
export END_DATE=$(date +%Y-%m-%d)
python main.py
```

**Configuration Method 2:** Modify `settings/default_config.json` (or `configs/default_config.json`)

```json
{
  "init_date": "2026-01-14",  // Today's date
  "end_date": "2026-01-14",
  "backtest": false
}
```

**Data Sources:**
- 📡 TinySoft live market feed (pyTSL)
- 📰 Real-time news updates (AKShare)

**Execution Flow:**
```
1. Check if today is a trading day (skip weekends/holidays)
2. Trigger decisions at fixed daily time points:
   • 10:30 - First hourly data available
   • 11:30 - Midday (last hour of morning session)
   • 14:00 - Afternoon core period
3. Fetch real-time quotes and latest news
4. LLM makes decisions
5. Simulate execution (no actual broker API connection)
6. Record to position.jsonl
```

**Note:** Trading is typically triggered manually via the web interface "Start Trading" button. Automatic scheduling is optional and can be configured separately if needed.

---

## ⚡ Quick Mode Switch

### Run Today's Live Trading:
```bash
cd /path/to/project
export INIT_DATE=$(date +%Y-%m-%d)
export END_DATE=$(date +%Y-%m-%d)
source env.sh  # Load TinySoft credentials
python main.py
```

### Backtest Specific Date Range:
```bash
export INIT_DATE="2025-11-01"
export END_DATE="2025-11-08"
python main.py
```

### Start via Frontend:
```
1. Navigate to http://localhost:5173
2. Click "Start Trading" button
3. Backend launches main.py to process today's trading
4. View "Decision Stream" on the right side for real-time updates
```

---

## 🧹 Data Cleanup Recommendations

### Reset Single Model Data:
```bash
rm -rf data_flow/trading_summary_each_agent/gemini-2.5-flash/position/position.jsonl
rm -rf data_flow/trading_summary_each_agent/gemini-2.5-flash/log/*
```

### Clean with Backup:
```bash
cp data_flow/trading_summary_each_agent/gemini-2.5-flash/position/position.jsonl \
   data_flow/trading_summary_each_agent/gemini-2.5-flash/position/position_backup_$(date +%Y%m%d).jsonl
> data_flow/trading_summary_each_agent/gemini-2.5-flash/position/position.jsonl
```

### Clean Old News (Keep Last 30 Days):
```python
# utilities/clean_news_csv.py
import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('data_flow/news.csv')
cutoff = datetime.now() - timedelta(days=30)
df = df[pd.to_datetime(df['publish_time']) > cutoff]
df.to_csv('data_flow/news.csv', index=False)
```

---

## ❓ Frequently Asked Questions (FAQ)

### Q: Can I delete position.jsonl?
**A:** Yes, but you will lose historical return data. Frontend charts will have no data to display. **Backup first** is recommended.

### Q: How to check the number of records?
**A:** 
```bash
# On Linux/Mac
wc -l data_flow/trading_summary_each_agent/gemini-2.5-flash/position/position.jsonl

```

### Q: Why does backend say "No trading days to process"?
**A:** The configured date range has been fully processed. To continue trading:
1. Modify `end_date` to a future date, **or**
2. Set environment variables `INIT_DATE`/`END_DATE` to today

### Q: Does live mode actually place real orders?
**A:** **No.** Currently only records simulated trades. To connect with real brokers, you need:
1. Integrate broker API (e.g., Huatai, East Money)
2. Add risk control checks
3. Implement order management system

### Q: How much disk space does data consume?
**A:** Example metrics:
- `position.jsonl`: ~50KB (37 records)
- Single day log: ~120KB (3 decisions × full conversations)
- `news.csv`: ~10MB (3 months of news)

---

## 🗂️ Related Files Index

| File | Purpose |
|------|---------|
| `data_flow/data_pipeline.py` | Data pipeline orchestration |
| `settings/default_config.json` | Main configuration file (primary location) |
| `configs/default_config.json` | Main configuration file (alternative location) |
| `utilities/shell/env.sh.template` | Environment variable template |
| `api_server.py` | Backend API (see `/api/live/*` endpoints) |
| `analysis/visualize.py` | PnL visualization generator |
| `experiments/visualize.py` | Alternative visualization workspace |

---

**Last Updated:** January 14, 2026  
**Version:** 1.0.0
