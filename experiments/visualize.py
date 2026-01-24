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
import pandas as pd
import os

# 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    # 尝试从项目根目录加载 .env 文件
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env file from {env_path}")
    else:
        # 如果项目根目录没有，尝试从当前目录加载
        load_dotenv()
except ImportError:
    # 如果没有安装 python-dotenv，跳过
    pass
except Exception as e:
    print(f"⚠️ Warning: Failed to load .env file: {e}")


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


ROOT_DIR = Path(__file__).parent
PROJECT_ROOT = ROOT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Date range filter (inclusive)
DATE_FILTER_START = datetime.strptime("2026-01-12 00:00:00", "%Y-%m-%d %H:%M:%S")
DATE_FILTER_END = datetime.strptime("2026-01-23 23:59:59", "%Y-%m-%d %H:%M:%S")


def in_date_range(dt: datetime) -> bool:
    """Return True if dt is within configured date window (inclusive)."""
    return DATE_FILTER_START <= dt <= DATE_FILTER_END

# PnL snapshots directory
PNL_SNAPSHOTS_DIR = PROJECT_ROOT / "data_flow" / "pnl_snapshots"

# Legacy data directories (for backward compatibility)
DATA_DATES = [f"data_{day}_1_2026" for day in range(12, 17)]

# 10只股票（等权重ETF组成）
ETF_STOCKS = [
    "SH688008",  # 澜起科技
    "SH688111",  # 金山办公
    "SH688009",  # 中国通号
    "SH688981",  # 中芯国际
    "SH688256",  # 寒武纪
    "SH688271",  # 联影医疗
    "SH688047",  # 龙芯中科
    "SH688617",  # 惠泰医疗
    "SH688303",  # 大全能源
    "SH688180",  # 君实生物
]


# 低配版模型（Lite版本）
MODELS_LITE = {
    "claude-haiku-4-5": {
        "color": "#8B5CF6", 
        "label": "Claude Haiku 4.5"
    },
    "deepseek-chat": {
        "color": "#F59E0B", 
        "label": "DeepSeek Chat"
    },
    "gpt-5.1": {
        "color": "#3B82F6", 
        "label": "GPT-5.1"
    },
    "qwen3-235b": {
        "color": "#EF4444", 
        "label": "Qwen3-235b"
    },
    "gemini-2.5-flash": {
        "color": "#10B981", 
        "label": "Gemini 2.5 Flash"
    }
}

# 升级版模型（Pro版本）
MODELS_PRO = {
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

# 模型版本对比映射
MODEL_PAIRS = {
    "claude-haiku-4-5": "claude-opus-4-5",
    "deepseek-chat": "deepseek-reasoner",
    "gpt-5.1": "gpt-5.2",
    "qwen3-235b": "qwen3-max",
    "gemini-2.5-flash": "gemini-3-pro-preview"
}

# 默认选择 Lite 版本，可通过环境变量或参数修改
import os
MODEL_VERSION = os.getenv("MODEL_VERSION", "lite").lower()
MODELS = MODELS_LITE if MODEL_VERSION == "lite" else MODELS_PRO

print(f"📌 Using {MODEL_VERSION.upper()} models: {list(MODELS.keys())}")

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

            if not in_date_range(dt):
                continue
            
            # 使用市场价格（决策时点的价格）计算权益
            equity = calculate_equity_with_market_price(pos, decision_time_str, date_str)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            pnl_series.append((dt, equity, return_pct))
        
        if pnl_series:
            all_pnl_data[model_sig] = sorted(pnl_series, key=lambda x: x[0])
            print(f"✓ Loaded {len(pnl_series)} unrealized PnL points for {model_sig} (using market price)")
    
    return all_pnl_data


def extract_unrealized_pnl_by_models(model_dict: Dict) -> Dict[str, List[Tuple[datetime, float, float]]]:
    """
    提取指定模型字典中的 Unrealized PnL（浮动盈亏）
    从 position.jsonl 计算，使用市场价格（decision_time 的价格）
    返回: {model_signature: [(datetime, equity, return_pct), ...]}
    """
    all_pnl_data = {}
    
    for model_sig in model_dict.keys():
        pnl_series = []
        
        # 从 position.jsonl 计算（使用市场价格）
        positions = read_position_data(model_sig)
        
        # 如果新目录没有数据，尝试旧目录结构（向后兼容）
        if not positions:
            for data_dir in DATA_DATES:
                positions = read_position_data(model_sig, data_dir)
                if positions:
                    break
        
        for pos in positions:
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

            if not in_date_range(dt):
                continue
            
            equity = calculate_equity_with_market_price(pos, decision_time_str, date_str)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            pnl_series.append((dt, equity, return_pct))
        
        if pnl_series:
            all_pnl_data[model_sig] = sorted(pnl_series, key=lambda x: x[0])
    
    return all_pnl_data


def extract_realized_pnl_by_models(model_dict: Dict) -> Dict[str, List[Tuple[datetime, float, float]]]:
    """
    提取指定模型字典中的 Realized PnL（基于成本价的权益）
    从 position.jsonl 计算，使用成本价（avg_price）
    返回: {model_signature: [(datetime, equity, return_pct), ...]}
    """
    all_pnl_data = {}
    
    for model_sig in model_dict.keys():
        pnl_series = []
        
        # 从 position.jsonl 计算（使用成本价）
        positions = read_position_data(model_sig)
        
        # 如果新目录没有数据，尝试旧目录结构（向后兼容）
        if not positions:
            for data_dir in DATA_DATES:
                positions = read_position_data(model_sig, data_dir)
                if positions:
                    break
        
        for pos in positions:
            if pos.get('seed', False):
                continue
            
            decision_time_str = pos.get('decision_time', '')
            if not decision_time_str:
                continue
            
            try:
                dt = datetime.strptime(decision_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            if not in_date_range(dt):
                continue
            
            equity = calculate_equity_with_cost_price(pos)
            return_pct = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            pnl_series.append((dt, equity, return_pct))
        
        if pnl_series:
            all_pnl_data[model_sig] = sorted(pnl_series, key=lambda x: x[0])
    
    return all_pnl_data


def extract_realized_pnl() -> Dict[str, List[Tuple[datetime, float, float]]]:
    """
    提取 Realized PnL（基于成本价的权益）
    从 position.jsonl 计算，使用成本价（avg_price）
    返回: {model_signature: [(datetime, equity, return_pct), ...]}
    
    Realized PnL = 现金 + 持仓成本（shares * avg_price）
    这是已实现的权益，不会随市场价格浮动。
    """
    return extract_realized_pnl_by_models(MODELS)


def extract_stock_attention(model_dict: Dict = None) -> Dict[datetime, Dict[str, int]]:
    """
    提取股票关注度数据（每个时间点，每支股票被多少个模型持有）
    返回: {datetime: {stock_symbol: num_models_holding, ...}, ...}
    可通过 model_dict 指定模型子集（默认全量 MODELS）。
    """
    if model_dict is None:
        model_dict = MODELS

    # 使用嵌套字典存储：{datetime: {stock_symbol: set(models_holding)}}
    attention_data_sets = {}
    
    for model_sig in model_dict.keys():
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
                
                if not in_date_range(dt):
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


def extract_model_attention_by_date(model_dict: Dict = None) -> Dict[str, Dict[str, int]]:
    """
    提取每个模型在每个日期的股票持有数（取每天各时间点的平均值）
    返回: {model_sig: {date_str: avg_num_stocks_held, ...}, ...}
    可通过 model_dict 指定模型子集（默认全量 MODELS）。
    """
    if model_dict is None:
        model_dict = MODELS

    model_attention = {}
    
    for model_sig in model_dict.keys():
        model_attention[model_sig] = {}
        
        positions = read_position_data(model_sig)
        if not positions:
            for data_dir in DATA_DATES:
                positions = read_position_data(model_sig, data_dir)
                if positions:
                    break
        
        date_to_stocks_count = {}  # {date_str: [stocks_count_per_time, ...]}
        
        for pos in positions:
            if pos.get('seed', False):
                continue
            
            date_str = pos.get('date', '')
            if not date_str:
                decision_time_str = pos.get('decision_time', '')
                if decision_time_str:
                    try:
                        dt = datetime.strptime(decision_time_str, "%Y-%m-%d %H:%M:%S")
                        date_str = dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                else:
                    continue
            
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if not in_date_range(dt):
                    continue
                date_key = dt.strftime('%d_%m')
            except ValueError:
                continue
            
            stocks_at_this_time = 0
            positions_dict = pos.get('positions', {})
            for symbol, info in positions_dict.items():
                if symbol != 'CASH' and isinstance(info, dict):
                    shares = info.get('shares', 0)
                    if shares > 0:
                        stocks_at_this_time += 1
            
            if date_key not in date_to_stocks_count:
                date_to_stocks_count[date_key] = []
            date_to_stocks_count[date_key].append(stocks_at_this_time)
        
        for date_key, stocks_count_per_time in date_to_stocks_count.items():
            if stocks_count_per_time:
                avg_stocks = round(sum(stocks_count_per_time) / len(stocks_count_per_time))
                model_attention[model_sig][date_key] = avg_stocks
            else:
                model_attention[model_sig][date_key] = 0
    
    return model_attention


def plot_model_attention_by_date(model_attention: Dict[str, Dict[str, int]], output_filename: str = "model_attention_by_date.png", title_prefix: str = "Model Stock Attention by Date", models_config: Dict = None):
    """绘制每个模型在不同日期的关注度（持有股票数）"""
    if not model_attention:
        print("⚠ No model attention data available")
        return
    
    # 如果没有指定 models_config，使用全局 MODELS
    if models_config is None:
        models_config = MODELS
    
    # 获取所有日期
    all_dates = set()
    for date_dict in model_attention.values():
        all_dates.update(date_dict.keys())
    all_dates = sorted(list(all_dates), 
                       key=lambda x: datetime.strptime(x, '%d_%m'))  # 按日期排序
    
    # 动态生成标题：根据实际日期范围
    if all_dates:
        first_date = datetime.strptime(all_dates[0], '%d_%m')
        last_date = datetime.strptime(all_dates[-1], '%d_%m')
        if first_date.month == last_date.month:
            title_date_range = f"Jan {first_date.day}-{last_date.day}, 2026"
        else:
            title_date_range = f"{first_date.strftime('%b %d')}-{last_date.strftime('%b %d')}, 2026"
    else:
        title_date_range = "Jan 12-16, 2026"  # 默认值
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 柱状图参数
    x = np.arange(len(all_dates))
    width = 0.15
    
    # 为每个模型绘制柱状图
    model_sigs = sorted(model_attention.keys())
    for idx, model_sig in enumerate(model_sigs):
        stocks_count = [model_attention.get(model_sig, {}).get(date, 0) 
                        for date in all_dates]
        label = models_config.get(model_sig, {}).get("label", model_sig)
        color = models_config.get(model_sig, {}).get("color", "#666666")
        
        ax.bar(x + idx * width, stocks_count,
               width=width,
               label=label,
               color=color,
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
    ax.set_title(f'{title_prefix} ({title_date_range})', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    
    # 动态设置y轴范围
    max_stocks = 0
    for date_dict in model_attention.values():
        max_stocks = max(max_stocks, max(date_dict.values(), default=0))
    y_max = max(6, max_stocks + 1)
    ax.set_ylim(0, y_max)
    ax.set_yticks(range(0, y_max + 1))
    
    # 图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_stock_attention(attention_data: Dict[datetime, Dict[str, int]], output_filename: str = "stock_attention.png", title_prefix: str = "Stock Attention Over Time"):
    """绘制股票关注度堆积面积图"""
    if not attention_data:
        print("⚠ No stock attention data available")
        return
    
    # 排序时间
    times = sorted(attention_data.keys())
    
    # 动态生成标题：根据实际日期范围
    if times:
        first_date = times[0]
        last_date = times[-1]
        if first_date.month == last_date.month:
            title_date_range = f"{first_date.strftime('%b %d')}-{last_date.day}, {first_date.year}"
        else:
            title_date_range = f"{first_date.strftime('%b %d')}-{last_date.strftime('%b %d')}, {first_date.year}"
    else:
        title_date_range = "Jan 12-16, 2026"  # 默认值
    
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
    ax.set_title(f'{title_prefix} ({title_date_range})', 
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
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_weekly_pnl(pnl_data: Dict[str, List[Tuple[datetime, float, float]]], 
                     title: str, output_filename: str, model_dict: Dict = None):
    """绘制 PnL 对比图（通用函数）"""
    if model_dict is None:
        model_dict = MODELS
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 收集所有时间点用于创建统一的x轴标签
    all_times = set()
    for data_points in pnl_data.values():
        for dt, _, _ in data_points:
            all_times.add(dt)
    
    # 按时间排序并创建索引映射
    sorted_times = sorted(all_times)
    time_to_index = {dt: idx for idx, dt in enumerate(sorted_times)}
    
    # 动态生成日期范围用于标题
    if sorted_times:
        first_date = sorted_times[0]
        last_date = sorted_times[-1]
        if first_date.month == last_date.month:
            date_range = f"{first_date.strftime('%b %d')}-{last_date.day}, {first_date.year}"
        else:
            date_range = f"{first_date.strftime('%b %d')}-{last_date.strftime('%b %d')}, {first_date.year}"
        # 替换标题中的 "Date Range" 占位符
        title = title.replace('Date Range', date_range)
    
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
        
        # 获取模型配置信息
        model_info = model_dict.get(model_sig, model_dict.get(model_sig, MODELS.get(model_sig, {"color": "#999999", "label": model_sig})))
        
        # 使用样条插值创建平滑曲线
        if len(x_indices) > 3:  # 样条插值至少需要4个点
            x_smooth = np.linspace(x_indices.min(), x_indices.max(), 300)
            spl = make_interp_spline(x_indices, returns, k=3)
            returns_smooth = spl(x_smooth)
            ax.plot(x_smooth, returns_smooth,
                   color=model_info["color"],
                   linewidth=3,
                   linestyle='-',
                   alpha=0.9,
                   label=model_info["label"])
        else:
            ax.plot(x_indices, returns,
                   color=model_info["color"],
                   linewidth=3,
                   linestyle='-',
                   alpha=0.9,
                   label=model_info["label"])
        
        # 添加数据点标记
        ax.scatter(x_indices, returns,
                  color=model_info["color"],
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


def plot_weekly_pnl_unrealized(pnl_data: Dict[str, List[Tuple[datetime, float, float]]], model_dict: Dict = None, version: str = ""):
    """绘制 Unrealized PnL 对比图（使用市场价格，浮动盈亏）"""
    if model_dict is None:
        model_dict = MODELS
    
    version_suffix = f"_{version}" if version else ""
    filename = f'pnl_weekly_unrealized{version_suffix}.png'
    
    plot_weekly_pnl(pnl_data, 
                     'Weekly Unrealized PnL Comparison (Market Price, Date Range)',
                     filename,
                     model_dict)


def plot_weekly_pnl_realized(pnl_data: Dict[str, List[Tuple[datetime, float, float]]], model_dict: Dict = None, version: str = ""):
    """绘制 Realized PnL 对比图（使用成本价，已实现权益）"""
    if model_dict is None:
        model_dict = MODELS
    
    version_suffix = f"_{version}" if version else ""
    filename = f'pnl_weekly_realized{version_suffix}.png'
    
    plot_weekly_pnl(pnl_data,
                     'Weekly Realized PnL Comparison (Cost Price, Date Range)',
                     filename,
                     model_dict)


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


def calculate_etf_price_series() -> List[Tuple[datetime, float]]:
    """
    计算10只股票的等权重ETF价格序列
    从 ai_stock_data.json 中获取每只股票的价格数据（仅 2026-01-12 到 2026-01-20），等权重求平均
    每天保留3个决策时点：10:30, 11:30, 14:00
    返回: [(timestamp, etf_price), ...]
    """
    stock_data_path = PROJECT_ROOT / "data_flow" / "ai_stock_data.json"
    if not stock_data_path.exists():
        print("⚠️ Warning: ai_stock_data.json not found, skipping ETF calculation")
        return []
    
    try:
        with open(stock_data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Failed to read ai_stock_data.json: {e}")
        return []
    
    # 日期范围限制
    start_date = DATE_FILTER_START
    end_date = DATE_FILTER_END
    
    # 决策时点（和前面两个图保持一致）
    decision_times = ["10:30:00", "11:30:00", "14:00:00"]
    
    # 为每只股票收集历史价格数据
    stock_prices = {symbol: {} for symbol in ETF_STOCKS}
    
    for symbol in ETF_STOCKS:
        if symbol not in full_data:
            print(f"⚠️ Warning: {symbol} not found in ai_stock_data.json")
            continue
        
        stock_data = full_data[symbol]
        if not isinstance(stock_data, dict):
            continue
        
        # 优先使用小时线行情，其次是日线行情
        price_data = stock_data.get('小时线行情') or stock_data.get('日线行情') or []
        
        for candle in price_data:
            # 时间字段可能是 'date' 或 'time'
            timestamp = candle.get('date') or candle.get('time')
            close_price = float(candle.get('close', 0))
            
            if timestamp and close_price > 0:
                try:
                    dt = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
                    # 只保留 2026-01-12 到 2026-01-20 的数据
                    if start_date <= dt <= end_date:
                        # 只保留决策时点的数据
                        time_str = dt.strftime('%H:%M:%S')
                        if time_str in decision_times:
                            stock_prices[symbol][timestamp] = close_price
                except:
                    continue
    
    # 合并所有时间戳
    all_timestamps = set()
    for prices_dict in stock_prices.values():
        all_timestamps.update(prices_dict.keys())
    
    if not all_timestamps:
        print("⚠️ Warning: No price data found for ETF stocks in date range 2026-01-12 to 2026-01-20")
        return []
    
    # 排序时间戳并计算每个时刻的等权重ETF价格
    sorted_timestamps = sorted(all_timestamps)
    etf_series = []
    
    for ts in sorted_timestamps:
        prices = []
        for symbol in ETF_STOCKS:
            if ts in stock_prices[symbol]:
                prices.append(stock_prices[symbol][ts])
        
        if prices:  # 只要有数据就计算平均
            avg_price = np.mean(prices)
            # 解析时间戳
            try:
                dt = datetime.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')
            except:
                continue
            
            etf_series.append((dt, avg_price))
    
    print(f"✓ ETF series: {len(etf_series)} data points from {etf_series[0][0]} to {etf_series[-1][0]}")
    
    return etf_series


def calculate_etf_return_series(etf_series: List[Tuple[datetime, float]]) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    计算ETF的收益率序列
    以 2026-01-12 第一个价格作为基准价格（初始价格）
    返回: {
        'etf': [(datetime, etf_price), ...],
        'return_pct': [(datetime, return_pct), ...]
    }
    """
    if not etf_series:
        return {}
    
    # 按时间排序
    etf_series = sorted(etf_series, key=lambda x: x[0])
    
    # 计算收益率（相对 2026-01-12 的第一个价格作为初始价格）
    initial_price = etf_series[0][1]
    
    if initial_price <= 0:
        print(f"⚠️ Warning: Invalid initial price {initial_price}")
        return {}
    
    return_pct_series = []
    
    for dt, price in etf_series:
        if price > 0:
            # 收益率 = (当前价格 - 初始价格) / 初始价格 * 100
            return_pct = ((price - initial_price) / initial_price) * 100
        else:
            return_pct = 0
        return_pct_series.append((dt, return_pct))
    
    print(f"✓ ETF return series: initial_price={initial_price:.2f}, final_price={etf_series[-1][1]:.2f}, final_return={return_pct_series[-1][1]:.2f}%")
    
    return {
        'etf': etf_series,
        'return_pct': return_pct_series
    }


def plot_etf_performance(etf_data: Dict):
    """
    绘制10只股票等权重ETF的表现图表
    包含: ETF价格、收益率（使用样条曲线，每天3个决策点）
    """
    if not etf_data or 'return_pct' not in etf_data:
        print("⚠️ Warning: No ETF data to plot")
        return
    
    etf_series = etf_data.get('etf', [])
    return_series = etf_data.get('return_pct', [])
    
    if not etf_series or not return_series:
        return
    
    # 按时间排序
    return_series = sorted(return_series, key=lambda x: x[0])
    etf_series = sorted(etf_series, key=lambda x: x[0])
    
    # 收集所有时间点
    all_times = set([dt for dt, _ in return_series])
    sorted_times = sorted(all_times)
    time_to_index = {dt: idx for idx, dt in enumerate(sorted_times)}
    
    # 提取数据（保留所有决策时点）
    x_indices = [time_to_index[dt] for dt, _ in return_series]
    returns = [ret for _, ret in return_series]
    prices = [price for _, price in etf_series]
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Equal-Weight ETF Performance (10 Stocks)', fontsize=16, fontweight='bold', y=0.995)
    
    # 图1: ETF价格走势（使用样条曲线）
    ax1 = axes[0]
    x_indices_arr = np.array(x_indices)
    prices_arr = np.array(prices)
    
    if len(x_indices) > 3:
        x_smooth = np.linspace(min(x_indices), max(x_indices), 300)
        spl = make_interp_spline(x_indices_arr, prices_arr, k=3)
        prices_smooth = spl(x_smooth)
        ax1.plot(x_smooth, prices_smooth, linewidth=2.5, color='#2E86AB', label='ETF Value')
    else:
        ax1.plot(x_indices, prices, 'o-', linewidth=2.5, color='#2E86AB', label='ETF Value', markersize=4)
    
    ax1.scatter(x_indices, prices, color='#2E86AB', s=30, alpha=0.6, zorder=5)
    ax1.set_ylabel('Value (¥)', fontsize=11)
    ax1.set_title('Equal-Weight ETF Value Trend', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # 图2: 收益率走势（使用样条曲线）
    ax2 = axes[1]
    returns_arr = np.array(returns)
    
    if len(x_indices) > 3:
        x_smooth = np.linspace(min(x_indices), max(x_indices), 300)
        spl = make_interp_spline(x_indices_arr, returns_arr, k=3)
        returns_smooth = spl(x_smooth)
        ax2.plot(x_smooth, returns_smooth, linewidth=2.5, color='#4CAF50', label='Return')
    else:
        ax2.plot(x_indices, returns, 'o-', linewidth=2.5, color='#4CAF50', label='Return', markersize=4)
    
    ax2.scatter(x_indices, returns, color='#4CAF50', s=30, alpha=0.6, zorder=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Date & Time', fontsize=11)
    ax2.set_ylabel('Return (%)', fontsize=11)
    ax2.set_title('Equal-Weight ETF Return Trend', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.legend(fontsize=10)
    
    # 设置x轴标签（显示日期和时间，和前两个图一样）
    x_labels = [dt.strftime('%m-%d\n%H:%M') for dt in sorted_times]
    x_positions = list(range(len(sorted_times)))
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([])  # 上图不显示标签
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = OUTPUT_DIR / "etf_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def fetch_star50_benchmark_series() -> List[Tuple[datetime, float]]:
    """
    从 nav_history.json 读取 Star50 benchmark 数据
    返回: [(datetime, return_pct), ...]
    """
    nav_history_path = PROJECT_ROOT / "data_flow" / "star50_benchmark" / "nav_history.json"
    
    if not nav_history_path.exists():
        print(f"⚠️ Warning: nav_history.json not found at {nav_history_path}")
        return []
    
    try:
        with open(nav_history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nav_history = data.get('nav_history', [])
        if not nav_history:
            print("⚠️ Warning: nav_history is empty")
            return []
        
        # 决策时点（每天3个时间点）
        decision_times = ["10:30:00", "11:30:00", "14:00:00"]
        
        star50_series = []
        current_date = None
        time_index_in_day = 0
        
        for entry in nav_history:
            date_str = entry.get('date', '')
            pnl_pct = entry.get('pnl_pct', 0.0)
            
            if not date_str:
                continue
            
            try:
                # 解析日期 (格式: "20260112")
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                
                # 如果是新的一天，重置时间索引
                if current_date != date_str:
                    current_date = date_str
                    time_index_in_day = 0
                
                # 根据当天的时间索引确定具体时间
                if time_index_in_day < len(decision_times):
                    time_str = decision_times[time_index_in_day]
                    time_parts = time_str.split(':')
                    dt = date_obj.replace(
                        hour=int(time_parts[0]),
                        minute=int(time_parts[1]),
                        second=int(time_parts[2])
                    )
                    
                    # 检查日期是否在范围内
                    if in_date_range(dt):
                        star50_series.append((dt, float(pnl_pct)))
                    
                    time_index_in_day += 1
                    
            except (ValueError, KeyError) as e:
                print(f"⚠️ Warning: Failed to parse entry {entry}: {e}")
                continue
        
        # 按时间排序
        star50_series = sorted(star50_series, key=lambda x: x[0])
        
        if star50_series:
            print(f"✓ Star50 benchmark series: {len(star50_series)} data points from {star50_series[0][0]} to {star50_series[-1][0]}")
        
        return star50_series
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to read nav_history.json: {e}")
        import traceback
        traceback.print_exc()
        return []


def plot_etf_vs_models(etf_data: Dict, unrealized_pnl_data: Dict, star50_series: List[Tuple[datetime, float]] = None):
    """
    对比ETF、Star50 benchmark与各模型的表现（使用样条曲线，每天3个决策点）
    使用 Unrealized PnL（市场价格）来展现模型的实际投资收益，包含买入时机效果
    """
    if not etf_data or not unrealized_pnl_data:
        print("⚠️ Warning: Missing data for comparison chart")
        return
    
    return_series = etf_data.get('return_pct', [])
    if not return_series:
        return
    
    return_series = sorted(return_series, key=lambda x: x[0])
    
    # 收集所有时间点
    all_times = set()
    for dt, _ in return_series:
        all_times.add(dt)
    for pnl_list in unrealized_pnl_data.values():
        for dt, _, _ in pnl_list:
            all_times.add(dt)
    
    # Add Star50 benchmark times if available
    if star50_series:
        for dt, _ in star50_series:
            all_times.add(dt)
    
    sorted_times = sorted(all_times)
    time_to_index = {dt: idx for idx, dt in enumerate(sorted_times)}
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Star50 benchmark 数据处理（如果有的话）
    if star50_series:
        star50_x_indices = [time_to_index[dt] for dt, _ in star50_series if dt in time_to_index]
        star50_returns = [ret for dt, ret in star50_series if dt in time_to_index]
        
        if star50_x_indices:
            star50_x_indices = np.array(star50_x_indices)
            star50_returns = np.array(star50_returns)
            
            # 绘制Star50 benchmark（使用样条曲线，紫色系）
            if len(star50_x_indices) > 3:
                x_smooth = np.linspace(star50_x_indices.min(), star50_x_indices.max(), 300)
                spl = make_interp_spline(star50_x_indices, star50_returns, k=3)
                returns_smooth = spl(x_smooth)
                ax.plot(x_smooth, returns_smooth, linewidth=3.5, color='#9B59B6', 
                        label='Star50 benchmark', zorder=5, linestyle='-.')
            else:
                ax.plot(star50_x_indices, star50_returns, 'o-', linewidth=3, color='#9B59B6',
                        label='Star50 benchmark', markersize=5, zorder=5, linestyle='-.')
            
            ax.scatter(star50_x_indices, star50_returns, color='#9B59B6', s=50, alpha=0.7, zorder=6)
    
    # ETF数据处理（保留所有决策时点）
    etf_x_indices = [time_to_index[dt] for dt, _ in return_series]
    etf_returns = [ret for _, ret in return_series]
    
    etf_x_indices = np.array(etf_x_indices)
    etf_returns = np.array(etf_returns)
    
    # 绘制ETF（使用样条曲线，改用蓝色系）
    if len(etf_x_indices) > 3:
        x_smooth = np.linspace(etf_x_indices.min(), etf_x_indices.max(), 300)
        spl = make_interp_spline(etf_x_indices, etf_returns, k=3)
        returns_smooth = spl(x_smooth)
        ax.plot(x_smooth, returns_smooth, linewidth=3.5, color='#2E86AB', 
                label='Equal-Weight ETF (10 Stocks)', zorder=5)
    else:
        ax.plot(etf_x_indices, etf_returns, 'o-', linewidth=3, color='#2E86AB',
                label='Equal-Weight ETF (10 Stocks)', markersize=5, zorder=5)
    
    ax.scatter(etf_x_indices, etf_returns, color='#2E86AB', s=50, alpha=0.7, zorder=6)
    
    # 绘制各模型（使用样条曲线）
    for model_sig, pnl_list in unrealized_pnl_data.items():
        if not pnl_list:
            continue
        
        pnl_list = sorted(pnl_list, key=lambda x: x[0])
        
        # 提取数据（保留所有决策时点）
        model_x_indices = []
        model_returns = []
        
        for dt, _, ret_pct in pnl_list:
            if dt in time_to_index:
                idx = time_to_index[dt]
                model_x_indices.append(idx)
                model_returns.append(ret_pct)
        
        if not model_x_indices:
            continue
        
        model_x_indices = np.array(model_x_indices)
        model_returns = np.array(model_returns)
        
        model_info = MODELS.get(model_sig, {})
        label = model_info.get('label', model_sig)
        color = model_info.get('color', '#000000')
        
        # 使用样条曲线
        if len(model_x_indices) > 3:
            x_smooth = np.linspace(model_x_indices.min(), model_x_indices.max(), 300)
            spl = make_interp_spline(model_x_indices, model_returns, k=3)
            returns_smooth = spl(x_smooth)
            ax.plot(x_smooth, returns_smooth, linewidth=2.5, color=color, 
                    label=label, alpha=0.85, zorder=3)
        else:
            ax.plot(model_x_indices, model_returns, 'o-', linewidth=2, color=color, 
                    label=label, markersize=4, alpha=0.8, zorder=3)
        
        ax.scatter(model_x_indices, model_returns, color=color, s=30, alpha=0.5, zorder=4)
    
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title('AI Trading Models vs Equal-Weight ETF Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 设置x轴标签（和前两个图一样）
    x_labels = [dt.strftime('%m-%d\n%H:%M') for dt in sorted_times]
    x_positions = list(range(len(sorted_times)))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel('Date & Time', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = OUTPUT_DIR / "etf_vs_models_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_benchmarks_comparison(etf_data: Dict, lite_pnl_data: Dict, pro_pnl_data: Dict, star50_series: List[Tuple[datetime, float]] = None):
    """
    生成Lite和Pro两个版本的benchmark对比图
    左图：Lite模型 vs ETF vs Star50 benchmark
    右图：Pro模型 vs ETF vs Star50 benchmark
    """
    if not etf_data:
        print("⚠️ Warning: Missing ETF data for benchmark comparison")
        return
    
    return_series = etf_data.get('return_pct', [])
    if not return_series:
        return
    
    return_series = sorted(return_series, key=lambda x: x[0])
    
    # 创建双图表
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 绘制Lite版本
    ax = axes[0]
    _plot_benchmark_single(ax, return_series, lite_pnl_data, "Lite Version", MODELS_LITE, star50_series)
    
    # 绘制Pro版本
    ax = axes[1]
    _plot_benchmark_single(ax, return_series, pro_pnl_data, "Pro Version", MODELS_PRO, star50_series)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "benchmarks_lite_vs_pro.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_benchmarks_realized(etf_data: Dict, lite_realized: Dict, pro_realized: Dict, star50_series: List[Tuple[datetime, float]] = None):
    """
    生成 Lite 和 Pro 版本的基准对比图（使用 Realized PnL）
    左图：Lite realized vs ETF vs Star50 benchmark
    右图：Pro realized vs ETF vs Star50 benchmark
    """
    if not etf_data:
        print("⚠️ Warning: Missing ETF data for realized benchmark comparison")
        return

    return_series = etf_data.get('return_pct', [])
    if not return_series:
        return

    return_series = sorted(return_series, key=lambda x: x[0])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    _plot_benchmark_single(ax, return_series, lite_realized, "Lite Version (Realized)", MODELS_LITE, star50_series)

    ax = axes[1]
    _plot_benchmark_single(ax, return_series, pro_realized, "Pro Version (Realized)", MODELS_PRO, star50_series)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "benchmarks_lite_vs_pro_realized.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def _plot_benchmark_single(ax, etf_series, model_pnl_data, title_suffix, model_dict, star50_series=None):
    """
    绘制单个benchmark对比图（Lite或Pro）
    """
    # 收集所有时间点
    all_times = set()
    for dt, _ in etf_series:
        all_times.add(dt)
    for pnl_list in model_pnl_data.values():
        for dt, _, _ in pnl_list:
            all_times.add(dt)
    if star50_series:
        for dt, _ in star50_series:
            all_times.add(dt)
    
    sorted_times = sorted(all_times)
    time_to_index = {dt: idx for idx, dt in enumerate(sorted_times)}
    
    # 绘制Star50 benchmark
    if star50_series:
        star50_x_indices = [time_to_index[dt] for dt, _ in star50_series if dt in time_to_index]
        star50_returns = [ret for dt, ret in star50_series if dt in time_to_index]
        
        if star50_x_indices:
            star50_x_indices = np.array(star50_x_indices)
            star50_returns = np.array(star50_returns)
            
            if len(star50_x_indices) > 3:
                x_smooth = np.linspace(star50_x_indices.min(), star50_x_indices.max(), 300)
                spl = make_interp_spline(star50_x_indices, star50_returns, k=3)
                returns_smooth = spl(x_smooth)
                ax.plot(x_smooth, returns_smooth, linewidth=3.5, color='#9B59B6', 
                        label='Star50 benchmark', zorder=5, linestyle='-.')
            else:
                ax.plot(star50_x_indices, star50_returns, 'o-', linewidth=3, color='#9B59B6',
                        label='Star50 benchmark', markersize=5, zorder=5, linestyle='-.')
            
            ax.scatter(star50_x_indices, star50_returns, color='#9B59B6', s=50, alpha=0.7, zorder=6)
    
    # 绘制ETF
    etf_x_indices = [time_to_index[dt] for dt, _ in etf_series]
    etf_returns = [ret for _, ret in etf_series]
    
    etf_x_indices = np.array(etf_x_indices)
    etf_returns = np.array(etf_returns)
    
    if len(etf_x_indices) > 3:
        x_smooth = np.linspace(etf_x_indices.min(), etf_x_indices.max(), 300)
        spl = make_interp_spline(etf_x_indices, etf_returns, k=3)
        returns_smooth = spl(x_smooth)
        ax.plot(x_smooth, returns_smooth, linewidth=3.5, color='#2E86AB', 
                label='Equal-Weight ETF (10 Stocks)', zorder=5)
    
    ax.scatter(etf_x_indices, etf_returns, color='#2E86AB', s=50, alpha=0.7, zorder=6)
    
    # 绘制模型（使用相应版本的配置）
    for model_sig, pnl_list in model_pnl_data.items():
        if not pnl_list:
            continue
        
        pnl_list = sorted(pnl_list, key=lambda x: x[0])
        
        model_x_indices = []
        model_returns = []
        
        for dt, _, ret_pct in pnl_list:
            if dt in time_to_index:
                idx = time_to_index[dt]
                model_x_indices.append(idx)
                model_returns.append(ret_pct)
        
        if not model_x_indices:
            continue
        
        model_x_indices = np.array(model_x_indices)
        model_returns = np.array(model_returns)
        
        model_info = model_dict.get(model_sig, {})
        label = model_info.get('label', model_sig)
        color = model_info.get('color', '#000000')
        
        if len(model_x_indices) > 3:
            x_smooth = np.linspace(model_x_indices.min(), model_x_indices.max(), 300)
            spl = make_interp_spline(model_x_indices, model_returns, k=3)
            returns_smooth = spl(x_smooth)
            ax.plot(x_smooth, returns_smooth, linewidth=2.5, color=color, 
                    label=label, alpha=0.85, zorder=3)
        else:
            ax.plot(model_x_indices, model_returns, 'o-', linewidth=2, color=color, 
                    label=label, markersize=4, alpha=0.8, zorder=3)
        
        ax.scatter(model_x_indices, model_returns, color=color, s=30, alpha=0.5, zorder=4)
    
    ax.set_ylabel('Return (%)', fontsize=11)
    ax.set_title(f'{title_suffix}: Models vs Benchmarks', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 设置x轴标签
    x_labels = [dt.strftime('%m-%d\n%H:%M') for dt in sorted_times]
    x_positions = list(range(len(sorted_times)))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Date & Time', fontsize=11)


def plot_model_version_comparison(lite_pnl_data: Dict, pro_pnl_data: Dict):
    """
    对比低配版（Lite）和升级版（Pro）模型的表现
    显示同系列模型的升级效果
    """
    if not lite_pnl_data or not pro_pnl_data:
        print("⚠️ Warning: Missing data for model version comparison")
        return
    
    # 收集所有时间点
    all_times = set()
    for pnl_list in lite_pnl_data.values():
        for dt, _, _ in pnl_list:
            all_times.add(dt)
    for pnl_list in pro_pnl_data.values():
        for dt, _, _ in pnl_list:
            all_times.add(dt)
    
    sorted_times = sorted(all_times)
    time_to_index = {dt: idx for idx, dt in enumerate(sorted_times)}
    
    # 创建图表
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    
    # 对每个模型系列绘制对比
    model_names = list(MODEL_PAIRS.keys())
    
    for idx, (lite_model, ax) in enumerate(zip(model_names, axes)):
        pro_model = MODEL_PAIRS[lite_model]
        
        # 获取 Lite 版本数据
        if lite_model in lite_pnl_data:
            lite_list = lite_pnl_data[lite_model]
            lite_x_indices = []
            lite_returns = []
            for dt, _, ret_pct in lite_list:
                if dt in time_to_index:
                    lite_x_indices.append(time_to_index[dt])
                    lite_returns.append(ret_pct)
            
            if lite_x_indices:
                lite_x_indices = np.array(lite_x_indices)
                lite_returns = np.array(lite_returns)
                
                # 绘制 Lite 版本
                if len(lite_x_indices) > 3:
                    x_smooth = np.linspace(lite_x_indices.min(), lite_x_indices.max(), 300)
                    spl = make_interp_spline(lite_x_indices, lite_returns, k=3)
                    returns_smooth = spl(x_smooth)
                    ax.plot(x_smooth, returns_smooth, linewidth=2.5, color='#94A3B8', 
                            label='Lite', alpha=0.8, zorder=3)
                else:
                    ax.plot(lite_x_indices, lite_returns, 'o-', linewidth=2, color='#94A3B8',
                            label='Lite', markersize=4, alpha=0.8, zorder=3)
                
                ax.scatter(lite_x_indices, lite_returns, color='#94A3B8', s=30, alpha=0.5, zorder=4)
        
        # 获取 Pro 版本数据
        if pro_model in pro_pnl_data:
            pro_list = pro_pnl_data[pro_model]
            pro_x_indices = []
            pro_returns = []
            for dt, _, ret_pct in pro_list:
                if dt in time_to_index:
                    pro_x_indices.append(time_to_index[dt])
                    pro_returns.append(ret_pct)
            
            if pro_x_indices:
                pro_x_indices = np.array(pro_x_indices)
                pro_returns = np.array(pro_returns)
                
                # 获取模型配置中的颜色
                lite_model_info = MODELS_LITE.get(lite_model, {})
                color = lite_model_info.get('color', '#000000')
                
                # 绘制 Pro 版本
                if len(pro_x_indices) > 3:
                    x_smooth = np.linspace(pro_x_indices.min(), pro_x_indices.max(), 300)
                    spl = make_interp_spline(pro_x_indices, pro_returns, k=3)
                    returns_smooth = spl(x_smooth)
                    ax.plot(x_smooth, returns_smooth, linewidth=2.5, color=color, 
                            label='Pro', alpha=0.9, zorder=3)
                else:
                    ax.plot(pro_x_indices, pro_returns, 'o-', linewidth=2, color=color,
                            label='Pro', markersize=4, alpha=0.9, zorder=3)
                
                ax.scatter(pro_x_indices, pro_returns, color=color, s=30, alpha=0.6, zorder=4)
        
        # 设置子图属性
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylabel('Return (%)', fontsize=10)
        
        # 使用简短的标签
        lite_label = lite_model.split('-')[0].capitalize()
        ax.set_title(f'{lite_label}: Lite vs Pro', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
    
    # 设置 x 轴标签（仅在最后一个子图显示）
    x_labels = [dt.strftime('%m-%d\n%H:%M') if i % 3 == 0 else '' 
                for i, dt in enumerate(sorted_times)]
    x_positions = list(range(len(sorted_times)))
    
    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Date & Time', fontsize=9)
    
    fig.suptitle('Model Version Comparison: Lite vs Pro', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # 保存图表
    output_file = OUTPUT_DIR / "model_version_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("AStock Arena - PnL Visualization Generator")
    print("=" * 60)
    print()
    
    # 0. 提取两个版本的模型数据
    print("📥 Extracting data for both model versions...\n")
    
    # 提取 Lite 版本的 Unrealized 和 Realized PnL
    print("  📊 Lite models (Haiku, Chat, 5.1, 235b, Flash):")
    lite_unrealized_pnl = extract_unrealized_pnl_by_models(MODELS_LITE)
    lite_realized_pnl = extract_realized_pnl_by_models(MODELS_LITE)
    print(f"    ✓ Loaded {len(lite_unrealized_pnl)} Lite models (unrealized)")
    print(f"    ✓ Loaded {len(lite_realized_pnl)} Lite models (realized)\n")
    
    # 提取 Pro 版本的 Unrealized 和 Realized PnL
    print("  📊 Pro models (Opus, Reasoner, 5.2, Max, 3-Pro):")
    pro_unrealized_pnl = extract_unrealized_pnl_by_models(MODELS_PRO)
    pro_realized_pnl = extract_realized_pnl_by_models(MODELS_PRO)
    print(f"    ✓ Loaded {len(pro_unrealized_pnl)} Pro models (unrealized)")
    print(f"    ✓ Loaded {len(pro_realized_pnl)} Pro models (realized)\n")
    
    # 使用当前选定的模型版本进行后续可视化
    if MODEL_VERSION == "lite":
        unrealized_pnl_data = lite_unrealized_pnl
        realized_pnl_data = lite_realized_pnl
    else:
        unrealized_pnl_data = pro_unrealized_pnl
        realized_pnl_data = pro_realized_pnl
    
    print(f"📌 Proceeding with {MODEL_VERSION.upper()} models for main visualizations\n")
    
    # 3. 计算等权重ETF
    print("📊 Calculating equal-weight ETF (10 stocks)...")
    etf_series = calculate_etf_price_series()
    if etf_series:
        print(f"✓ ETF price series: {len(etf_series)} data points")
        etf_data = calculate_etf_return_series(etf_series)
    else:
        etf_data = {}
        print("⚠ Warning: Failed to calculate ETF")
    
    # 3.5 获取Star50 benchmark数据
    print("📊 Fetching Star50 benchmark data...")
    star50_series = fetch_star50_benchmark_series()
    if not star50_series:
        print("⚠ Warning: Failed to fetch Star50 benchmark data")
    
    # 4. 生成PnL对比图（4张：Lite Unrealized、Lite Realized、Pro Unrealized、Pro Realized）
    print("\n📈 Generating Weekly PnL Charts (4 charts total):")
    
    if lite_unrealized_pnl:
        print("  📊 Lite Unrealized PnL...")
        plot_weekly_pnl_unrealized(lite_unrealized_pnl, MODELS_LITE, "lite")
    
    if lite_realized_pnl:
        print("  📊 Lite Realized PnL...")
        plot_weekly_pnl_realized(lite_realized_pnl, MODELS_LITE, "lite")
    
    if pro_unrealized_pnl:
        print("  📊 Pro Unrealized PnL...")
        plot_weekly_pnl_unrealized(pro_unrealized_pnl, MODELS_PRO, "pro")
    
    if pro_realized_pnl:
        print("  📊 Pro Realized PnL...")
        plot_weekly_pnl_realized(pro_realized_pnl, MODELS_PRO, "pro")
    
    # 5. 生成其他图表
    if etf_data:
        print("📈 Generating ETF performance chart...")
        plot_etf_performance(etf_data)
        
        print("📈 Generating ETF vs Models comparison chart (with Star50 benchmark)...")
        plot_etf_vs_models(etf_data, unrealized_pnl_data, star50_series)
    
    print("📊 Generating stock attention charts (overall, Lite, Pro)...")
    attention_data_all = extract_stock_attention()
    plot_stock_attention(attention_data_all, "stock_attention.png", "Stock Attention Over Time (All Models)")

    attention_data_lite = extract_stock_attention(MODELS_LITE)
    plot_stock_attention(attention_data_lite, "stock_attention_lite.png", "Stock Attention Over Time (Lite)")

    attention_data_pro = extract_stock_attention(MODELS_PRO)
    plot_stock_attention(attention_data_pro, "stock_attention_pro.png", "Stock Attention Over Time (Pro)")
    
    print("📊 Generating model attention by date charts (Lite, Pro)...")
    model_attention_lite = extract_model_attention_by_date(MODELS_LITE)
    plot_model_attention_by_date(model_attention_lite, "model_attention_by_date_lite.png", "Model Stock Attention by Date (Lite)", models_config=MODELS_LITE)

    model_attention_pro = extract_model_attention_by_date(MODELS_PRO)
    plot_model_attention_by_date(model_attention_pro, "model_attention_by_date_pro.png", "Model Stock Attention by Date (Pro)", models_config=MODELS_PRO)
    
    # 4.5 生成模型版本对比图
    print("📈 Generating Model Version Comparison chart (Lite vs Pro)...")
    if lite_unrealized_pnl and pro_unrealized_pnl:
        plot_model_version_comparison(lite_unrealized_pnl, pro_unrealized_pnl)
    else:
        print("⚠ Warning: Insufficient data for model version comparison")
    
    # 4.6 生成Lite和Pro的benchmark对比图
    print("📈 Generating Benchmarks Comparison chart (Lite & Pro vs ETF & Star50 benchmark)...")
    if etf_data and lite_unrealized_pnl and pro_unrealized_pnl:
        plot_benchmarks_comparison(etf_data, lite_unrealized_pnl, pro_unrealized_pnl, star50_series)
    else:
        print("⚠ Warning: Insufficient data for benchmarks comparison")

    # 4.7 生成Lite和Pro的Realized benchmark对比图
    print("📈 Generating Realized Benchmarks Comparison chart (Lite & Pro vs ETF & Star50 benchmark)...")
    if etf_data and lite_realized_pnl and pro_realized_pnl:
        plot_benchmarks_realized(etf_data, lite_realized_pnl, pro_realized_pnl, star50_series)
    else:
        print("⚠ Warning: Insufficient data for realized benchmarks comparison")
    
    # 5. 生成统计摘要（使用当前选定版本的Realized PnL）
    print("\n📊 Generating summary statistics...")
    if not realized_pnl_data:
        print("⚠ Warning: No PnL data available for summary statistics.")
    else:
        summary = generate_summary_stats(realized_pnl_data)
        summary_file = OUTPUT_DIR / "performance_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"✓ Saved: {summary_file}")
    
    print(f"\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print(f"📊 Generated charts (16 total):")
    print(f"   • pnl_weekly_unrealized_lite.png (Lite Unrealized)")
    print(f"   • pnl_weekly_realized_lite.png (Lite Realized)")
    print(f"   • pnl_weekly_unrealized_pro.png (Pro Unrealized)")
    print(f"   • pnl_weekly_realized_pro.png (Pro Realized)")
    print(f"   • etf_performance.png")
    print(f"   • etf_vs_models_comparison.png")
    print(f"   • benchmarks_lite_vs_pro.png (Lite & Pro vs Benchmarks)")
    print(f"   • benchmarks_lite_vs_pro_realized.png (Lite & Pro vs Benchmarks, Realized)")
    print(f"   • model_version_comparison.png")
    print(f"   • stock_attention.png (All Models)")
    print(f"   • stock_attention_lite.png (Lite Models)")
    print(f"   • stock_attention_pro.png (Pro Models)")
    print(f"   • model_attention_by_date_lite.png (Lite Models)")
    print(f"   • model_attention_by_date_pro.png (Pro Models)")
    print(f"   • performance_summary.md")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"📌 Active model version for other benchmarks: {MODEL_VERSION.upper()}")
    print("=" * 60)




if __name__ == "__main__":
    main()
