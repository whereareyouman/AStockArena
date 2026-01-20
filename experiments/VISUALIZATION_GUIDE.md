# AStock Arena 可视化指南

## 概述

该可视化脚本支持两套模型版本的性能对比：

- **Lite 版本（低配）**：Claude Haiku 4.5, DeepSeek Chat, GPT-5.1, Qwen3-235b, Gemini 2.5 Flash
- **Pro 版本（升级）**：Claude Opus 4.5, DeepSeek Reasoner, GPT-5.2, Qwen3-Max, Gemini 3 Pro Preview

## 生成的图表

### 1. PnL 周报图（两个版本）
- `pnl_weekly_unrealized.png` - 未实现收益（浮动权益，使用市场价格）
- `pnl_weekly_realized.png` - 已实现收益（成本价）

### 2. ETF 表现图
- `etf_performance.png` - 10只股票等权重ETF的表现
  - 左：ETF 价格变化
  - 右：收益率变化

### 3. ETF vs 模型对比图
- `etf_vs_models_comparison.png` - 将AI模型与指数基准对比
  - 蓝色线：等权重ETF (10只股票)
  - 橙色虚线：科创50指数
  - 彩色实线：各AI模型

### 4. 模型版本对比图 ⭐ **新增**
- `model_version_comparison.png` - 5个子图，对比同系列模型的 Lite vs Pro 版本
  - 灰色线：Lite 版本性能
  - 彩色线：Pro 版本性能
  - 展示升级版模型相比低配版的改进情况

### 5. 其他分析图
- `stock_attention.png` - 股票关注度（多少个模型持有）
- `model_attention_by_date.png` - 模型日期关注度
- `performance_summary.md` - 性能统计总结

## 使用方法

### 方式 1：使用默认配置（Lite 版本）
```bash
cd /path/to/AStockArena
python3 experiments/visualize.py
```

### 方式 2：切换到 Pro 版本
```bash
export MODEL_VERSION=pro
python3 experiments/visualize.py
```

或者一行命令：
```bash
MODEL_VERSION=pro python3 experiments/visualize.py
```

### 方式 3：回到 Lite 版本
```bash
export MODEL_VERSION=lite
python3 experiments/visualize.py
```

## 关键特性

### ✅ 双模型版本支持
- 轻松切换 Lite/Pro 版本进行对比分析
- 所有图表自动使用选定的模型版本

### ✅ Unrealized PnL（浮动权益）
- 与指数基准进行公平对比
- 反映模型的真实市场表现和择时能力

### ✅ 科创50指数基准
- 使用 akshare 获取真实的市场指数数据
- 与ETF和模型性能并排比较

### ✅ 模型版本对比
- 新增 `model_version_comparison.png` 显示：
  - Claude Haiku 4.5 vs Claude Opus 4.5
  - DeepSeek Chat vs DeepSeek Reasoner
  - GPT-5.1 vs GPT-5.2
  - Qwen3-235b vs Qwen3-Max
  - Gemini 2.5 Flash vs Gemini 3 Pro Preview

## 数据来源

1. **模型性能数据**：`data_flow/pnl_snapshots/` 中的 PnL 快照文件
2. **股票价格数据**：`data_flow/ai_stock_data.json`
3. **指数数据**：通过 akshare 实时获取科创50数据

## 输出目录

所有可视化文件保存在：
```
experiments/visualizations/
```

## 模型映射表

| Lite 版本 | Pro 版本 |
|----------|---------|
| Claude Haiku 4.5 | Claude Opus 4.5 |
| DeepSeek Chat | DeepSeek Reasoner |
| GPT-5.1 | GPT-5.2 |
| Qwen3-235b | Qwen3-Max |
| Gemini 2.5 Flash | Gemini 3 Pro Preview |

## 可视化样式

- **时间轴**：统一的决策时点（10:30, 11:30, 14:00）
- **曲线**：样条插值（k=3）以平滑数据
- **颜色**：
  - 模型：保持一致的配色方案
  - ETF：蓝色 (#2E86AB)
  - 科创50：橙色虚线 (#FF6B35)

## 常见问题

**Q: 为什么 Lite 版本性能比 Pro 版本好？**
A: 这可能取决于特定的市场条件和时间段。通过对比图，你可以发现在不同时期各模型的相对表现。

**Q: 如何生成特定模型的对比？**
A: 修改 `MODELS_LITE` 或 `MODELS_PRO` 字典中的模型列表，然后重新运行脚本。

**Q: 图表中的时间点是什么含义？**
A: 每天有3个决策时点（10:30, 11:30, 14:00），对应模型做出交易决策的时刻。
