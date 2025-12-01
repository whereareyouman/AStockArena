# 数据存储说明 (Data Storage Documentation)

## 数据保存位置 (Data Storage Locations)

### 1. 交易头寸数据 (Portfolio Position Data)
**位置**: `data/agent_data/{model_name}/position/position.jsonl`

**格式**: JSONL (每行一个JSON对象)

**内容**: 每次交易决策后的完整投资组合快照
- `date`: 交易日期 (YYYY-MM-DD)
- `cash`: 当前现金余额
- `equity`: 总权益 (现金 + 持仓市值)
- `positions`: 持仓列表 (股票代码、股数、成本等)
- `decision`: 当日决策动作
- `timestamp`: 决策时间戳

**示例路径**:
```
data/agent_data/gemini-2.5-flash/position/position.jsonl
data/agent_data/deepseek-chat-v3.1/position/position.jsonl
```

**用途**:
- 计算累计收益率 (total return %)
- 计算夏普比率 (Sharpe ratio)
- 计算最大回撤 (max drawdown)
- 生成权益曲线图表
- 显示最近决策流

---

### 2. 决策日志 (Decision Logs)
**位置**: `data/agent_data/{model_name}/log/{date}/session_{timestamp}.jsonl`

**格式**: JSONL (每行一个JSON对象)

**内容**: LLM的完整决策过程和推理
- `timestamp`: 决策时间
- `prompt`: 发送给LLM的输入提示
- `response`: LLM的原始响应
- `reasoning`: 决策理由
- `action`: 交易动作 (buy/sell/hold)
- `symbol`: 股票代码
- `amount`: 交易金额

**示例路径**:
```
data/agent_data/gemini-2.5-flash/log/2025-10-02/session_093012.jsonl
data/agent_data/gemini-2.5-flash/log/2025-11-07/session_140015.jsonl
```

**用途**:
- 分析AI决策质量
- 审计交易记录
- 调试模型行为
- 回溯决策原因

---

### 3. 新闻数据 (News Data)
**位置**: `data/news.csv`

**格式**: CSV (逗号分隔值)

**内容**: AKShare聚合的股市新闻
- `title`: 新闻标题
- `content`: 新闻内容 (完整或摘要)
- `publish_time`: 发布时间
- `symbol`: 相关股票代码
- `source`: 新闻来源 (东方财富、新浪财经等)
- `url`: 原文链接

**数据来源**: 
- AKShare Python库 (`akshare.stock_news_em()`)
- 每日更新 (通过`data_pipeline.py`)

**用途**:
- 前端新闻面板显示
- LLM决策时的参考信息
- 情绪分析输入

---

## 运行模式对比 (Backtest vs Live Mode)

### 回测模式 (Backtest Mode) - 当前默认
**配置方式**: 在 `configs/default_config.json` 中设置日期范围

```json
{
  "init_date": "2025-10-30",
  "end_date": "2025-11-08",
  "backtest": true
}
```

**数据来源**:
- TinySoft历史数据 (pyTSL)
- 已保存的历史新闻 (news.csv)

**执行流程**:
1. 从`init_date`开始逐个交易日处理
2. 获取当日历史行情数据
3. LLM基于历史信息做决策
4. 模拟执行交易并更新持仓
5. 记录到position.jsonl
6. 继续下一个交易日直到`end_date`

**优点**:
- 快速验证策略
- 可重复测试
- 无需实时市场连接
- 可并行测试多个AI模型

**当前状态识别**:
```bash
python main.py
# 输出: "No trading days to process"
# 说明: 已完成2025-10-30到2025-11-08的回测
```

---

### 实盘模式 (Live Trading Mode)
**配置方式**: 通过环境变量覆盖日期为今天

```bash
export INIT_DATE=$(date +%Y-%m-%d)
export END_DATE=$(date +%Y-%m-%d)
python main.py
```

**或者修改 `configs/default_config.json`**:
```json
{
  "init_date": "2025-11-09",  // 今天日期
  "end_date": "2025-11-09",
  "backtest": false
}
```

**数据来源**:
- TinySoft实时行情 (pyTSL live feed)
- 实时新闻更新 (AKShare)

**执行流程**:
1. 检测今天是否为交易日 (跳过周末/节假日)
2. 在每日固定时点触发决策:
   - 09:30 (开盘)
   - 11:30 (午间)
   - 14:00 (午后)
3. 获取实时行情和最新新闻
4. LLM做出决策
5. **模拟执行** (当前未连接券商API)
6. 记录到position.jsonl

**定时执行**:
- 使用cron/systemd定时任务
- 或web界面"Start Trading"按钮触发

**示例cron配置** (每小时检查):
```cron
0 9-15 * * 1-5 cd /path/to/AI-Trader && /path/to/venv/bin/python main.py
```

---

## 快速切换模式 (Quick Mode Switch)

### 运行今日实盘:
```bash
cd /Users/fangdoudou/Desktop/urop25-26/AI-Trader_11_8
export INIT_DATE=$(date +%Y-%m-%d)
export END_DATE=$(date +%Y-%m-%d)
source env.sh  # 加载TinySoft凭据
python main.py
```

### 回测特定日期范围:
```bash
export INIT_DATE="2025-11-01"
export END_DATE="2025-11-08"
python main.py
```

### 通过前端启动:
1. 访问 http://localhost:5173
2. 点击"Start Trading"按钮
3. 后端启动 `main.py` 处理今日交易
4. 查看右侧"决策流"实时更新

---

## 数据清理建议 (Data Cleanup)

### 重置单个模型数据:
```bash
rm -rf data/agent_data/gemini-2.5-flash/position/position.jsonl
rm -rf data/agent_data/gemini-2.5-flash/log/*
```

### 保留备份后清理:
```bash
cp data/agent_data/gemini-2.5-flash/position/position.jsonl \
   data/agent_data/gemini-2.5-flash/position/position_backup_$(date +%Y%m%d).jsonl
> data/agent_data/gemini-2.5-flash/position/position.jsonl
```

### 清理旧新闻 (保留最近30天):
```python
# scripts/clean_news_csv.py
import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('data/news.csv')
cutoff = datetime.now() - timedelta(days=30)
df = df[pd.to_datetime(df['publish_time']) > cutoff]
df.to_csv('data/news.csv', index=False)
```

---

## 常见问题 (FAQ)

**Q: position.jsonl可以删除吗？**  
A: 可以，但会丢失历史收益数据。前端图表将无数据显示。建议先备份。

**Q: 如何查看当前有多少条记录？**  
A: `wc -l data/agent_data/gemini-2.5-flash/position/position.jsonl`

**Q: 为什么后端说"No trading days to process"？**  
A: 配置的日期范围已处理完。要继续交易：
   1. 修改`end_date`到未来日期，或
   2. 设置环境变量`INIT_DATE`/`END_DATE`为今天

**Q: 实盘模式会真的下单吗？**  
A: 不会。当前只记录模拟交易。要对接真实券商需要：
   1. 集成券商API (如华泰、东方财富)
   2. 添加风控检查
   3. 实现订单管理系统

**Q: 数据占用多少空间？**  
A: 示例：
   - position.jsonl: ~50KB (37条记录)
   - 单日log: ~120KB (3次决策 × 完整对话)
   - news.csv: ~10MB (3个月新闻)

---

## 相关文件索引

- 数据管道: `data/data_pipeline.py`
- 配置文件: `configs/default_config.json`
- 环境模板: `env.sh.template`
- 后端API: `api_server.py` (查看`/api/live/*`端点)
- 位置迁移脚本: `scripts/migrate_positions.py`
