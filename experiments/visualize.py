#!/usr/bin/env python3
"""
PnL Visualization Script for AStock Arena
Generates three comparison charts: intraday, daily, and weekly PnL across all models.
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np
from scipy.interpolate import make_interp_spline


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


ROOT_DIR = Path(__file__).parent
PROJECT_ROOT = ROOT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# PnL snapshots directory
PNL_SNAPSHOTS_DIR = PROJECT_ROOT / "data_flow" / "pnl_snapshots"

# Legacy data directories (for backward compatibility)
DATA_DATES = [f"data_{day}_1_2026" for day in range(12, 17)]


MODELS = {
    "claude-opus-4-5": {
        "color": "#8B5CF6", 
        "label": "Claude Opus 4.5"
    },
    "deepseek-reasoner": {
        "color": "#F59E0B", 
        "label": "DeepSeek Reasoner"
    },
    "gpt-5.2": {
        "color": "#3B82F6", 
        "label": "GPT-5.2"
    },
    "qwen3-max": {
        "color": "#EF4444", 
        "label": "Qwen3-Max"
    },
    "gemini-3-pro-preview": {
        "color": "#10B981", 
        "label": "Gemini 3 Pro Preview"
    }
}

INITIAL_CAPITAL = 1000000.0


def read_position_data(model_signature: str, data_dir: str = None) -> List[Dict]:
    """读取模型的持仓数据"""
    # 优先使用新的目录结构
    position_file = PROJECT_ROOT / "data_flow" / "trading_summary_each_agent" / model_signature / "position" / "position.jsonl"
    
    # 如果新路径不存在，尝试旧路径（向后兼容）
    if not position_file.exists() and data_dir:
        position_file = ROOT_DIR / data_dir / "agent_data" / model_signature / "position" / "position.jsonl"
    
    if not position_file.exists():
        return []
    
    positions = []
    with open(position_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                positions.append(json.loads(line))
    
    return positions


def get_price_at_time(symbol: str, decision_time: str, date_str: str = None) -> float:
    """根据决策时点获取股票市场价格（从 ai_stock_data.json）"""
    stock_data_path = PROJECT_ROOT / "data_flow" / "ai_stock_data.json"
    if not stock_data_path.exists():
        return 0.0
    
    try:
        with open(stock_data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # 尝试多个可能的键名
        stock_entry = None
        symbol_upper = symbol.upper()
        for key in [symbol_upper, symbol_upper.replace("SH", "").replace("SZ", ""), f"SH{symbol_upper}", f"SZ{symbol_upper}"]:
            if key in full_data:
                stock_entry = full_data[key]
                break
        
        if not stock_entry or not isinstance(stock_entry, dict):
            return 0.0
        
        # 优先使用小时线行情，如果没有则使用日线行情
        hourly_data = stock_entry.get("小时线行情") or []
        daily_data = stock_entry.get("日线行情") or []
        
        data_list = hourly_data if hourly_data else daily_data
        if not data_list:
            return 0.0
        
        # 根据 decision_time 查找 <= 目标时间的最新价格
        target_time = decision_time or (date_str + " 15:00:00" if date_str else None)
        if not target_time:
            last_item = data_list[-1]
            return float(last_item.get("close") or last_item.get("buy1") or 0)
        
        # 倒序遍历，找第一个 <= target_time 的记录
        best_match = None
        for item in reversed(data_list):
            item_time = item.get("time") or item.get("date") or ""
            if item_time and item_time <= target_time:
                best_match = item
                break
        
        if best_match:
            return float(best_match.get("close") or best_match.get("buy1") or 0)
        
        # 如果找不到，返回第一条记录的价格（最早的价格）
        first_item = data_list[0]
        return float(first_item.get("close") or first_item.get("buy1") or 0)
        
    except Exception:
        return 0.0


def calculate_equity_with_cost_price(position_data: Dict) -> float:
    """计算账户权益（使用成本价）= 现金 + 持仓成本（avg_price * shares）"""
    cash = float(position_data.get('positions', {}).get('CASH', 0))
    equity = cash
    
    positions = position_data.get('positions', {})
    for symbol, info in positions.items():
        if symbol != 'CASH' and isinstance(info, dict):
            shares = float(info.get('shares', 0))
            avg_price = float(info.get('avg_price', 0))
            if shares > 0 and avg_price > 0:
                # 使用成本价计算
                equity += shares * avg_price
    
    return equity


def calculate_equity_with_market_price(position_data: Dict, decision_time: str, date_str: str = None) -> float:
    """计算账户权益（使用市场价格）= 现金 + 持仓市值（决策时点的市场价格）"""
    cash = float(position_data.get('positions', {}).get('CASH', 0))
    equity = cash
    
    positions = position_data.get('positions', {})
    for symbol, info in positions.items():
        if symbol != 'CASH' and isinstance(info, dict):
            shares = float(info.get('shares', 0))
            if shares > 0:
                # 使用决策时点的市场价格
                current_price = get_price_at_time(symbol, decision_time, date_str)
                equity += shares * current_price
    
    return equity


def read_pnl_snapshot(model_signature: str) -> List[Dict]:
    """读取模型的 PnL 快照文件"""
    # 文件名格式: pnl_{signature}.json
    pnl_file = PNL_SNAPSHOTS_DIR / f"pnl_{model_signature}.json"
    
    if not pnl_file.exists():
        return []
    
    try:
        with open(pnl_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Warning: Failed to read PnL snapshot for {model_signature}: {e}")
        return []


def extract_unrealized_pnl() -> Dict[str, List[Tuple[datetime, float, float]]]:
    """
    提取 Unrealized PnL（浮动盈亏）
    从 position.jsonl 计算，使用市场价格（decision_time 的价格）
    返回: {model_signature: [(datetime, equity, return_pct), ...]}
    
    Unrealized PnL = 现金 + 持仓市值（shares * 当前市场价格）
    这是浮动权益，会随市场价格变化而波动。
    """
    all_pnl_data = {}
    
    for model_sig in MODELS.keys():
        pnl_series = []
        
        # 从 position.jsonl 计算（使用市场价格）
        # 优先使用新目录结构，如果不存在则尝试旧目录结构
        positions = read_position_data(model_sig)
        
        # 如果新目录没有数据，尝试旧目录结构（向后兼容）
        if not positions:
            for data_dir in DATA_DATES:
                positions = read_position_data(model_sig, data_dir)
                if positions:
                    break
        
        for pos in positions:
            # 跳过seed记录
            if pos.get('seed', False):
                continue
            
            decision_time_str = pos.get('decision_time', '')
            date_str = pos.get('date', '')
            if not decision_time_str:
                continue
            
            try:
                dt = datetime.strptime(decision_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            
            # 使用市场价格（决策时点的价格）计算权益
            equity = calculate_equity_with_market_price(pos, decision_time_str, date_str)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            pnl_series.append((dt, equity, return_pct))
        
        if pnl_series:
            all_pnl_data[model_sig] = sorted(pnl_series, key=lambda x: x[0])
            print(f"✓ Loaded {len(pnl_series)} unrealized PnL points for {model_sig} (using market price)")
    
    return all_pnl_data


def extract_realized_pnl() -> Dict[str, List[Tuple[datetime, float, float]]]:
    """
    提取 Realized PnL（基于成本价的权益）
    从 position.jsonl 计算，使用成本价（avg_price）
    返回: {model_signature: [(datetime, equity, return_pct), ...]}
    
    Realized PnL = 现金 + 持仓成本（shares * avg_price）
    这是已实现的权益，不会随市场价格浮动。
    """
    all_pnl_data = {}
    
    for model_sig in MODELS.keys():
        pnl_series = []
        
        # 从 position.jsonl 计算（使用成本价）
        # 优先使用新目录结构，如果不存在则尝试旧目录结构
        positions = read_position_data(model_sig)
        
        # 如果新目录没有数据，尝试旧目录结构（向后兼容）
        if not positions:
            for data_dir in DATA_DATES:
                positions = read_position_data(model_sig, data_dir)
                if positions:
                    break
        
        for pos in positions:
            # 跳过seed记录
            if pos.get('seed', False):
                continue
            
            decision_time_str = pos.get('decision_time', '')
            if not decision_time_str:
                continue
            
            try:
                dt = datetime.strptime(decision_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            
            # 使用成本价计算权益（不会随市场波动）
            equity = calculate_equity_with_cost_price(pos)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            pnl_series.append((dt, equity, return_pct))
        
        if pnl_series:
            all_pnl_data[model_sig] = sorted(pnl_series, key=lambda x: x[0])
            print(f"✓ Loaded {len(pnl_series)} realized PnL points for {model_sig} (using cost price)")
    
    return all_pnl_data


def extract_stock_attention() -> Dict[datetime, Dict[str, int]]:
    """
    提取股票关注度数据（每个时间点，每支股票被多少个模型持有）
    返回: {datetime: {stock_symbol: num_models_holding, ...}, ...}
    """
    # 使用嵌套字典存储：{datetime: {stock_symbol: set(models_holding)}}
    attention_data_sets = {}
    
    for model_sig in MODELS.keys():
        # 遍历所有日期目录
        for data_dir in DATA_DATES:
            positions = read_position_data(model_sig, data_dir)
            
            for pos in positions:
                # 跳过seed记录
                if pos.get('seed', False):
                    continue
                
                decision_time_str = pos.get('decision_time', '')
                if not decision_time_str:
                    continue
                
                try:
                    dt = datetime.strptime(decision_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                
                # 初始化时间点的数据结构
                if dt not in attention_data_sets:
                    attention_data_sets[dt] = {}
                
                # 收集该模型在此时间点持有的股票
                positions_dict = pos.get('positions', {})
                for symbol, info in positions_dict.items():
                    if symbol != 'CASH' and isinstance(info, dict):
                        shares = info.get('shares', 0)
                        if shares > 0:
                            # 初始化股票的模型集合
                            if symbol not in attention_data_sets[dt]:
                                attention_data_sets[dt][symbol] = set()
                            # 将模型添加到集合中（自动去重）
                            attention_data_sets[dt][symbol].add(model_sig)
    
    # 转换为最终格式：计数而非集合
    attention_data = {}
    for dt, stocks in attention_data_sets.items():
        attention_data[dt] = {symbol: len(models) for symbol, models in stocks.items()}
    
    # 按时间排序
    sorted_attention = {dt: attention_data[dt] for dt in sorted(attention_data.keys())}
    return sorted_attention


def extract_model_attention_by_date() -> Dict[str, Dict[str, int]]:
    """
    提取每个模型在每个日期的股票持有数（取每天各时间点的平均值）
    返回: {model_sig: {date_str: avg_num_stocks_held, ...}, ...}
    """
    model_attention = {}
    
    for model_sig in MODELS.keys():
        model_attention[model_sig] = {}
        
        # 遍历所有日期目录
        for data_dir in DATA_DATES:
            positions = read_position_data(model_sig, data_dir)
            
            # 提取日期
            date_str = data_dir.replace('data_', '').replace('_2026', '')
            
            # 存储该日期每个时间点的持仓股票数
            stocks_count_per_time = []
            
            for pos in positions:
                # 跳过seed记录
                if pos.get('seed', False):
                    continue
                
                # 统计该时间点持有的股票数
                stocks_at_this_time = 0
                positions_dict = pos.get('positions', {})
                for symbol, info in positions_dict.items():
                    if symbol != 'CASH' and isinstance(info, dict):
                        shares = info.get('shares', 0)
                        if shares > 0:  # 只统计持仓数大于0的股票
                            stocks_at_this_time += 1
                
                stocks_count_per_time.append(stocks_at_this_time)
            
            # 计算平均值（四舍五入到整数）
            if stocks_count_per_time:
                avg_stocks = round(sum(stocks_count_per_time) / len(stocks_count_per_time))
                model_attention[model_sig][date_str] = avg_stocks
            else:
                model_attention[model_sig][date_str] = 0
    
    return model_attention


def plot_model_attention_by_date(model_attention: Dict[str, Dict[str, int]]):
    """绘制每个模型在不同日期的关注度（持有股票数）"""
    if not model_attention:
        print("⚠ No model attention data available")
        return
    
    # 获取所有日期
    all_dates = set()
    for date_dict in model_attention.values():
        all_dates.update(date_dict.keys())
    all_dates = sorted(list(all_dates), 
                       key=lambda x: datetime.strptime(x, '%d_%m'))  # 按日期排序
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 柱状图参数
    x = np.arange(len(all_dates))
    width = 0.15
    
    # 为每个模型绘制柱状图
    for idx, model_sig in enumerate(sorted(MODELS.keys())):
        stocks_count = [model_attention.get(model_sig, {}).get(date, 0) 
                        for date in all_dates]
        
        ax.bar(x + idx * width, stocks_count,
               width=width,
               label=MODELS[model_sig]["label"],
               color=MODELS[model_sig]["color"],
               alpha=0.8,
               edgecolor='black',
               linewidth=0.5)
    
    # 设置x轴标签
    date_labels = [d.replace('_', '-') for d in all_dates]
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(date_labels, fontsize=12, fontweight='bold')
    
    # 格式化
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Stocks Held', fontsize=14, fontweight='bold')
    ax.set_title('Model Stock Attention by Date (Jan 12-16, 2026)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    
    # 设置y轴（最多10支股票）
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    
    # 图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_attention_by_date.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_stock_attention(attention_data: Dict[datetime, Dict[str, int]]):
    """绘制股票关注度堆积面积图"""
    if not attention_data:
        print("⚠ No stock attention data available")
        return
    
    # 排序时间
    times = sorted(attention_data.keys())
    
    # 获取所有股票符号
    all_stocks = set()
    for stock_dict in attention_data.values():
        all_stocks.update(stock_dict.keys())
    all_stocks = sorted(list(all_stocks))
    
    # 构建数据矩阵
    attention_matrix = []
    for stock in all_stocks:
        stock_attention = [attention_data[t].get(stock, 0) for t in times]
        attention_matrix.append(stock_attention)
    
    # 生成颜色列表（为10支股票分配颜色）
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_stocks)))
    
    # 创建堆积面积图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 时间轴索引
    x_indices = np.arange(len(times))
    
    # 绘制堆积面积
    ax.stackplot(x_indices, attention_matrix,
                labels=all_stocks,
                colors=colors,
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5)
    
    # 设置x轴标签
    x_labels = [t.strftime('%m-%d\n%H:%M') for t in times]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # 格式化
    ax.set_xlabel('Date & Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Models Holding', fontsize=14, fontweight='bold')
    ax.set_title('Stock Attention Over Time (Jan 12-16, 2026)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    
    # 计算实际数据的最大值（每个时间点的总和）
    max_total = 0
    for t in times:
        total = sum(attention_data[t].values())
        max_total = max(max_total, total)
    
    # 动态设置y轴范围：使用实际最大值 + 1
    y_max = max_total + 1
    ax.set_ylim(0, y_max)
    ax.set_yticks(range(0, y_max + 1))
    
    # 图例（放在右侧）
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
             ncol=1, framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "stock_attention.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_weekly_pnl(pnl_data: Dict[str, List[Tuple[datetime, float, float]]], 
                     title: str, output_filename: str):
    """绘制 PnL 对比图（通用函数）"""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 收集所有时间点用于创建统一的x轴标签
    all_times = set()
    for data_points in pnl_data.values():
        for dt, _, _ in data_points:
            all_times.add(dt)
    
    # 按时间排序并创建索引映射
    sorted_times = sorted(all_times)
    time_to_index = {dt: idx for idx, dt in enumerate(sorted_times)}
    
    for model_sig, data_points in pnl_data.items():
        if not data_points:
            continue
        
        # 使用等间距的索引作为x坐标
        x_indices = np.array([time_to_index[dt] for dt, _, _ in data_points])
        returns = np.array([return_pct for _, _, return_pct in data_points])
        
        # 去除重复的x坐标（保留第一个）
        unique_indices = []
        unique_returns = []
        seen = set()
        for x, y in zip(x_indices, returns):
            if x not in seen:
                unique_indices.append(x)
                unique_returns.append(y)
                seen.add(x)
        
        x_indices = np.array(unique_indices)
        returns = np.array(unique_returns)
        
        # 使用样条插值创建平滑曲线
        if len(x_indices) > 3:  # 样条插值至少需要4个点
            x_smooth = np.linspace(x_indices.min(), x_indices.max(), 300)
            spl = make_interp_spline(x_indices, returns, k=3)
            returns_smooth = spl(x_smooth)
            ax.plot(x_smooth, returns_smooth,
                   color=MODELS[model_sig]["color"],
                   linewidth=3,
                   linestyle='-',
                   alpha=0.9,
                   label=MODELS[model_sig]["label"])
        else:
            ax.plot(x_indices, returns,
                   color=MODELS[model_sig]["color"],
                   linewidth=3,
                   linestyle='-',
                   alpha=0.9,
                   label=MODELS[model_sig]["label"])
        
        # 添加数据点标记
        ax.scatter(x_indices, returns,
                  color=MODELS[model_sig]["color"],
                  s=50,
                  alpha=0.6,
                  zorder=5)
    
    # 添加基准线和图例
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even', zorder=1)
    ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='gray')
    
    # 格式化
    ax.set_xlabel('Date & Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置x轴和y轴格式
    x_positions = list(range(len(sorted_times)))
    x_labels = [dt.strftime('%m-%d\n%H:%M') for dt in sorted_times]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_weekly_pnl_unrealized(pnl_data: Dict[str, List[Tuple[datetime, float, float]]]):
    """绘制 Unrealized PnL 对比图（使用市场价格，浮动盈亏）"""
    plot_weekly_pnl(pnl_data, 
                     'Weekly Unrealized PnL Comparison (Market Price, Jan 12-16, 2026)',
                     'pnl_weekly_unrealized.png')


def plot_weekly_pnl_realized(pnl_data: Dict[str, List[Tuple[datetime, float, float]]]):
    """绘制 Realized PnL 对比图（使用成本价，已实现权益）"""
    plot_weekly_pnl(pnl_data,
                     'Weekly Realized PnL Comparison (Cost Price, Jan 12-16, 2026)',
                     'pnl_weekly_realized.png')


def generate_summary_stats(pnl_data: Dict[str, List[Tuple[datetime, float, float]]]) -> str:
    """生成统计摘要"""
    summary_lines = ["## 📊 Performance Summary\n"]
    summary_lines.append("| Model | Latest Return | Max Return | Min Return | Volatility |")
    summary_lines.append("|-------|---------------|------------|------------|------------|")
    
    for model_sig, data_points in sorted(pnl_data.items()):
        if not data_points:
            continue
        
        returns = [r for _, _, r in data_points]
        latest_return = returns[-1] if returns else 0
        max_return = max(returns) if returns else 0
        min_return = min(returns) if returns else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        summary_lines.append(
            f"| {MODELS[model_sig]['label']} | "
            f"{latest_return:.2f}% | "
            f"{max_return:.2f}% | "
            f"{min_return:.2f}% | "
            f"{volatility:.2f}% |"
        )
    
    return "\n".join(summary_lines)


def main():
    """主函数"""
    print("=" * 60)
    print("AStock Arena - PnL Visualization Generator")
    print("=" * 60)
    print()
    
    # 1. 提取 Unrealized PnL 数据（使用市场价格，浮动盈亏）
    print("📥 Extracting Unrealized PnL data from position files (market price)...")
    unrealized_pnl_data = extract_unrealized_pnl()
    
    if not unrealized_pnl_data:
        print("⚠ Warning: No unrealized PnL data found.")
    else:
        print(f"\n✓ Loaded unrealized PnL data for {len(unrealized_pnl_data)} models\n")
    
    # 2. 提取 Realized PnL 数据（使用成本价，已实现权益）
    print("📥 Extracting Realized PnL data from position files (cost price)...")
    realized_pnl_data = extract_realized_pnl()
    
    if not realized_pnl_data:
        print("⚠ Warning: No realized PnL data found.")
    else:
    
        print(f"\n✓ Loaded realized PnL data for {len(realized_pnl_data)} models\n")
    
    # 3. 生成图表
    if unrealized_pnl_data:
        print("📈 Generating weekly Unrealized PnL chart (market price)...")
        plot_weekly_pnl_unrealized(unrealized_pnl_data)
    
    if realized_pnl_data:
        print("📈 Generating weekly Realized PnL chart (cost price)...")
        plot_weekly_pnl_realized(realized_pnl_data)
    
    print("📊 Generating stock attention chart...")
    attention_data = extract_stock_attention()
    plot_stock_attention(attention_data)
    
    print("📊 Generating model attention by date chart...")
    model_attention = extract_model_attention_by_date()
    plot_model_attention_by_date(model_attention)
    
    # 4. 生成统计摘要（优先使用 Realized PnL，如果没有则使用 Unrealized PnL）
    print("\n📊 Generating summary statistics...")
    pnl_for_summary = realized_pnl_data if realized_pnl_data else unrealized_pnl_data
    if not pnl_for_summary:
        print("⚠ Warning: No PnL data available for summary statistics.")
    else:
        summary = generate_summary_stats(pnl_for_summary)
        summary_file = OUTPUT_DIR / "performance_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"✓ Saved: {summary_file}")
    
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
