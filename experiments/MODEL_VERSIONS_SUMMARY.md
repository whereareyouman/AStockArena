# 📊 可视化功能实现总结

## 🎯 完成功能清单

### 1. ✅ 双模型版本配置
```python
MODELS_LITE = {
    "claude-haiku-4-5": Claude Haiku 4.5,
    "deepseek-chat": DeepSeek Chat,
    "gpt-5.1": GPT-5.1,
    "qwen3-235b": Qwen3-235b,
    "gemini-2.5-flash": Gemini 2.5 Flash
}

MODELS_PRO = {
    "claude-opus-4-5": Claude Opus 4.5,
    "deepseek-reasoner": DeepSeek Reasoner,
    "gpt-5.2": GPT-5.2,
    "qwen3-max": Qwen3-Max,
    "gemini-3-pro-preview": Gemini 3 Pro Preview
}
```

**特点**：
- 每套5个模型
- 不同的配色方案保持一致性
- 通过 `MODEL_VERSION` 环境变量切换

### 2. ✅ 灵活的版本切换

```bash
# 方式1：使用 Lite（默认）
python3 experiments/visualize.py

# 方式2：切换到 Pro
export MODEL_VERSION=pro
python3 experiments/visualize.py

# 方式3：一行命令
MODEL_VERSION=pro python3 experiments/visualize.py
```

**输出**：
```
📌 Using LITE models: ['claude-haiku-4-5', 'deepseek-chat', 'gpt-5.1', 'qwen3-235b', 'gemini-2.5-flash']
```

### 3. ✅ 新增模型版本对比图

**文件**：`model_version_comparison.png`

**结构**：5个子图（5×1 布局）
- 子图 1：Claude Haiku vs Claude Opus
- 子图 2：DeepSeek Chat vs DeepSeek Reasoner  
- 子图 3：GPT-5.1 vs GPT-5.2
- 子图 4：Qwen3-235b vs Qwen3-Max
- 子图 5：Gemini 2.5 Flash vs Gemini 3 Pro

**特点**：
- 灰色线显示 Lite 版本
- 彩色线显示 Pro 版本（与主图颜色保持一致）
- 样条曲线插值，同样的 3 个决策时点
- 清晰地展示升级效果

### 4. ✅ 自动化数据提取

新增函数：`extract_unrealized_pnl_by_models(model_dict)`
- 接收任意模型字典
- 提取对应的 Unrealized PnL 数据
- 用于对比两个版本

### 5. ✅ 改进的主函数流程

```
主函数流程：
┌─────────────────────────────────────────┐
│ 1. 提取两套模型版本的数据（Lite + Pro）│
├─────────────────────────────────────────┤
│ 2. 根据 MODEL_VERSION 选择活跃模型     │
├─────────────────────────────────────────┤
│ 3. 生成所有现有图表（使用活跃模型）    │
├─────────────────────────────────────────┤
│ 4. 生成模型版本对比图（Lite vs Pro）   │
├─────────────────────────────────────────┤
│ 5. 生成性能统计摘要                    │
└─────────────────────────────────────────┘
```

## 📈 生成的图表清单

### 主要图表（模型选择）
| 图表 | 文件名 | 用途 |
|------|--------|------|
| PnL 周报 - Unrealized | `pnl_weekly_unrealized.png` | 浮动权益对比 |
| PnL 周报 - Realized | `pnl_weekly_realized.png` | 成本价权益对比 |
| ETF 表现 | `etf_performance.png` | 10只股票ETF走势 |
| ETF vs 模型 | `etf_vs_models_comparison.png` | AI模型 vs 基准 |

### 分析图表（固定）
| 图表 | 文件名 | 用途 |
|------|--------|------|
| **版本对比** ⭐ | `model_version_comparison.png` | Lite vs Pro 对比 |
| 股票关注度 | `stock_attention.png` | 模型关注的股票 |
| 日期关注度 | `model_attention_by_date.png` | 各日期的选股数量 |
| 性能统计 | `performance_summary.md` | 数字化性能指标 |

## 🔄 版本对比示例

运行以下命令对比两套模型：

```bash
# Step 1: 生成 Lite 版本所有图表
python3 experiments/visualize.py
# 输出：pnl_weekly_unrealized.png (Lite models)

# Step 2: 生成 Pro 版本所有图表
MODEL_VERSION=pro python3 experiments/visualize.py
# 输出：pnl_weekly_unrealized.png (Pro models)

# Step 3: 对比模型版本
# 查看 model_version_comparison.png
# 对比 Claude Haiku vs Claude Opus, DeepSeek Chat vs DeepSeek Reasoner 等
```

## 💡 使用建议

### 场景 1：评估低配版模型
```bash
python3 experiments/visualize.py
# 查看所有 Lite 模型的性能
```

### 场景 2：测试高端模型
```bash
MODEL_VERSION=pro python3 experiments/visualize.py
# 查看所有 Pro 模型的性能
```

### 场景 3：对比升级效果
```bash
# 同时生成两个版本后，查看 model_version_comparison.png
# 对于每个模型系列，明确看到升级版vs低配版的改进
```

### 场景 4：选择最佳模型
```bash
# 查看 ETF vs Models 图表中的性能排名
# 结合成本考虑，选择 Lite 或 Pro 版本
```

## 🔧 技术细节

### 模型映射
```python
MODEL_PAIRS = {
    "claude-haiku-4-5": "claude-opus-4-5",
    "deepseek-chat": "deepseek-reasoner",
    "gpt-5.1": "gpt-5.2",
    "qwen3-235b": "qwen3-max",
    "gemini-2.5-flash": "gemini-3-pro-preview"
}
```

### 环境变量
```bash
MODEL_VERSION=lite    # 使用 Lite 版本（默认）
MODEL_VERSION=pro     # 使用 Pro 版本
```

### 数据流
```
PnL 快照文件 (15 个模型)
    ↓
MODELS_LITE (提取 5 个)
MODELS_PRO (提取 5 个)
    ↓
选择 MODEL_VERSION
    ↓
生成所有图表 (使用选定版本)
生成对比图表 (Lite vs Pro)
```

## 📝 修改的文件

### `/Users/fangdoudou/Desktop/AStockArena/experiments/visualize.py`

**主要修改**：
1. 添加 `MODELS_LITE` 和 `MODELS_PRO` 字典
2. 添加 `MODEL_PAIRS` 映射
3. 添加 `MODEL_VERSION` 环境变量支持
4. 添加 `extract_unrealized_pnl_by_models()` 函数
5. 添加 `plot_model_version_comparison()` 函数
6. 更新 `main()` 函数以支持两套模型版本

**新增功能**：
- 灵活的模型版本切换
- 自动化的两套模型数据提取
- 5×1 子图的版本对比可视化

## 📚 相关文档

- [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) - 详细使用指南
- [run_visualizations.sh](./run_visualizations.sh) - 快速启动脚本

## ✨ 总结

✅ 创建了两套完整的模型配置（Lite 和 Pro）
✅ 支持灵活的版本切换
✅ 生成了新的模型版本对比图表
✅ 保持了所有现有图表的功能
✅ 改进了用户体验和可视化分析

现在你可以：
1. 对比低配版和升级版模型的性能
2. 轻松切换不同的模型版本进行分析
3. 通过对比图表明确看到升级的效果
4. 为选择模型版本提供数据支持
