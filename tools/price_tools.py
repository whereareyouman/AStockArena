import copy
import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

# 将项目根目录加入 Python 路径，便于从子目录直接运行本文件
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from tools.general_tools import get_config_value

all_nasdaq_100_symbols = [
    "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "NFLX", "PLTR", "COST", "ASML", "AMD", "CSCO", "AZN", "TMUS", "MU", "LIN",
    "PEP", "SHOP", "APP", "INTU", "AMAT", "LRCX", "PDD", "QCOM", "ARM", "INTC",
    "BKNG", "AMGN", "TXN", "ISRG", "GILD", "KLAC", "PANW", "ADBE", "HON",
    "CRWD", "CEG", "ADI", "ADP", "DASH", "CMCSA", "VRTX", "MELI", "SBUX",
    "CDNS", "ORLY", "SNPS", "MSTR", "MDLZ", "ABNB", "MRVL", "CTAS", "TRI",
    "MAR", "MNST", "CSX", "ADSK", "PYPL", "FTNT", "AEP", "WDAY", "REGN", "ROP",
    "NXPI", "DDOG", "AXON", "ROST", "IDXX", "EA", "PCAR", "FAST", "EXC", "TTWO",
    "XEL", "ZS", "PAYX", "WBD", "BKR", "CPRT", "CCEP", "FANG", "TEAM", "CHTR",
    "KDP", "MCHP", "GEHC", "VRSK", "CTSH", "CSGP", "KHC", "ODFL", "DXCM", "TTD",
    "ON", "BIIB", "LULU", "CDW", "GFS"
]


def normalize_symbol(symbol: Optional[str]) -> Optional[str]:
    """
    统一股票代码格式：
    - 已有前缀 SH/SZ 的保持不变
    - 6 位纯数字根据常见规则自动补前缀
    """
    if symbol is None:
        return None

    s = str(symbol).strip().upper()
    if not s:
        return s

    if len(s) >= 8 and s[:2] in ("SH", "SZ"):
        return s

    if len(s) == 6 and s.isdigit():
        if s.startswith(("688", "689", "600", "601", "603", "605", "730", "735")):
            return f"SH{s}"
        if s.startswith(("000", "001", "002", "003", "300", "301", "302")):
            return f"SZ{s}"
        return f"SH{s}"

    return s


def strip_exchange_prefix(symbol: Optional[str]) -> Optional[str]:
    """
    将带前缀的股票代码还原为 6 位数字，供行情接口使用。
    """
    if symbol is None:
        return None
    s = str(symbol).strip().upper()
    if len(s) >= 8 and s[:2] in ("SH", "SZ"):
        return s[2:]
    return s


def normalize_decision_time(date_str: Optional[str], decision_time: Optional[str]) -> str:
    """
    规范化 decision_time，确保与记录日期匹配，且始终包含 HH:MM:SS。
    """
    date_part = str(date_str or "").strip()
    time_part = "00:00:00"

    if decision_time:
        candidate = decision_time.strip()
        try:
            dt = datetime.fromisoformat(candidate)
            time_part = dt.strftime("%H:%M:%S")
        except ValueError:
            tokens = candidate.split()
            tail = tokens[-1] if tokens else ""
            if ":" in tail:
                segments = tail.split(":")
                if len(segments) == 2:
                    tail = f"{tail}:00"
                elif len(segments) > 3:
                    tail = "00:00:00"
                time_part = tail

    if not date_part:
        return time_part
    return f"{date_part} {time_part}"


def _normalize_position_entry(data: Any) -> Tuple[int, Optional[str], Optional[float]]:
    """
    将 position 条目转换为统一结构，返回 (shares, purchase_date, avg_price)。
    """
    if isinstance(data, dict):
        shares = int(data.get("shares", 0) or 0)
        purchase_date = data.get("purchase_date")
        if purchase_date in (None, "", "null"):
            purchase_date = None
        else:
            purchase_date = str(purchase_date)
        avg_price_raw = data.get("avg_price")
        avg_price = None
        if avg_price_raw is not None:
            try:
                avg_price = float(avg_price_raw)
            except (TypeError, ValueError):
                avg_price = None
        return shares, purchase_date, avg_price

    try:
        shares = int(float(data))
    except Exception:
        shares = 0
    return shares, None, None


def normalize_positions(positions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    深拷贝并归并持仓，移除重复/裸代码键，保证结构统一：
    {
        "CASH": float,
        "SH600519": {"shares": int, "purchase_date": Optional[str]},
        ...
    }
    """
    if not positions:
        return {"CASH": 0.0}

    normalized: Dict[str, Any] = {}
    cash = positions.get("CASH", 0.0)
    try:
        normalized["CASH"] = float(cash)
    except Exception:
        normalized["CASH"] = 0.0

    accumulator: Dict[str, Dict[str, Any]] = {}

    for raw_symbol, data in positions.items():
        if raw_symbol == "CASH":
            continue

        normalized_symbol = normalize_symbol(raw_symbol)
        shares, purchase_date, avg_price = _normalize_position_entry(data)
        if shares <= 0 or normalized_symbol is None:
            continue

        existing = accumulator.get(normalized_symbol)
        if existing:
            existing_shares = existing.get("shares", 0)
            existing["shares"] = existing_shares + shares
            if existing.get("purchase_date") is None:
                existing["purchase_date"] = purchase_date
            elif purchase_date:
                existing_date = existing.get("purchase_date")
                if existing_date is None:
                    existing["purchase_date"] = purchase_date
                else:
                    existing["purchase_date"] = min(existing_date, purchase_date)
            if avg_price is not None and shares > 0:
                existing_avg = existing.get("avg_price")
                if existing_avg is None:
                    existing["avg_price"] = avg_price
                else:
                    total_shares = existing_shares + shares
                    if total_shares > 0:
                        weighted = ((existing_avg * existing_shares) + (avg_price * shares)) / total_shares
                        existing["avg_price"] = weighted
        else:
            accumulator[normalized_symbol] = {
                "shares": shares,
                "purchase_date": purchase_date,
                "avg_price": avg_price,
            }

    normalized.update(accumulator)
    return normalized


def get_yesterday_date(today_date: str) -> str:
    """
    获取昨日日期，考虑休市日。
    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。

    Returns:
        yesterday_date: 昨日日期字符串，格式 YYYY-MM-DD。
    """
    # 计算昨日日期，考虑休市日
    today_dt = datetime.strptime(today_date, "%Y-%m-%d")
    yesterday_dt = today_dt - timedelta(days=1)
    
    # 如果昨日是周末，向前找到最近的交易日
    while yesterday_dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
        yesterday_dt -= timedelta(days=1)
    
    yesterday_date = yesterday_dt.strftime("%Y-%m-%d")
    return yesterday_date

def get_open_prices(today_date: str, symbols: List[str], merged_path: Optional[str] = None) -> Dict[str, Optional[float]]:
    """从 data/merged.jsonl 中读取指定日期与标的的开盘价。

    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD。
        symbols: 需要查询的股票代码列表。
        merged_path: 可选，自定义 merged.jsonl 路径；默认读取项目根目录下 data/merged.jsonl。

    Returns:
        {symbol_price: open_price 或 None} 的字典；若未找到对应日期或标的，则值为 None。
    """
    wanted = set(symbols)
    results: Dict[str, Optional[float]] = {}

    if merged_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        merged_file = base_dir / "data" / "merged.jsonl"
    else:
        merged_file = Path(merged_path)

    if not merged_file.exists():
        return results

    with merged_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except Exception:
                continue
            meta = doc.get("Meta Data", {}) if isinstance(doc, dict) else {}
            sym = meta.get("2. Symbol")
            if sym not in wanted:
                continue
            series = doc.get("Time Series (Daily)", {})
            if not isinstance(series, dict):
                continue
            bar = series.get(today_date)
            if isinstance(bar, dict):
                open_val = bar.get("1. buy price")
                try:
                    results[f'{sym}_price'] = float(open_val) if open_val is not None else None
                except Exception:
                    results[f'{sym}_price'] = None

    return results

def get_yesterday_open_and_close_price(today_date: str, symbols: List[str], merged_path: Optional[str] = None) -> tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """从 data/merged.jsonl 中读取指定日期与股票的昨日买入价和卖出价。

    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        symbols: 需要查询的股票代码列表。
        merged_path: 可选，自定义 merged.jsonl 路径；默认读取项目根目录下 data/merged.jsonl。

    Returns:
        (买入价字典, 卖出价字典) 的元组；若未找到对应日期或标的，则值为 None。
    """
    wanted = set(symbols)
    buy_results: Dict[str, Optional[float]] = {}
    sell_results: Dict[str, Optional[float]] = {}

    if merged_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        merged_file = base_dir / "data" / "merged.jsonl"
    else:
        merged_file = Path(merged_path)

    if not merged_file.exists():
        return buy_results, sell_results

    yesterday_date = get_yesterday_date(today_date)

    with merged_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except Exception:
                continue
            meta = doc.get("Meta Data", {}) if isinstance(doc, dict) else {}
            sym = meta.get("2. Symbol")
            if sym not in wanted:
                continue
            series = doc.get("Time Series (Daily)", {})
            if not isinstance(series, dict):
                continue
            
            # 尝试获取昨日买入价和卖出价
            bar = series.get(yesterday_date)
            if isinstance(bar, dict):
                buy_val = bar.get("1. buy price")  # 买入价字段
                sell_val = bar.get("4. sell price")  # 卖出价字段
                
                try:
                    buy_price = float(buy_val) if buy_val is not None else None
                    sell_price = float(sell_val) if sell_val is not None else None
                    buy_results[f'{sym}_price'] = buy_price
                    sell_results[f'{sym}_price'] = sell_price
                except Exception:
                    buy_results[f'{sym}_price'] = None
                    sell_results[f'{sym}_price'] = None
            else:
                # 如果昨日没有数据，尝试向前查找最近的交易日
                today_dt = datetime.strptime(today_date, "%Y-%m-%d")
                yesterday_dt = today_dt - timedelta(days=1)
                current_date = yesterday_dt
                found_data = False
                
                # 最多向前查找5个交易日
                for _ in range(5):
                    current_date -= timedelta(days=1)
                    # 跳过周末
                    while current_date.weekday() >= 5:
                        current_date -= timedelta(days=1)
                    
                    check_date = current_date.strftime("%Y-%m-%d")
                    bar = series.get(check_date)
                    if isinstance(bar, dict):
                        buy_val = bar.get("1. buy price")
                        sell_val = bar.get("4. sell price")
                        
                        try:
                            buy_price = float(buy_val) if buy_val is not None else None
                            sell_price = float(sell_val) if sell_val is not None else None
                            buy_results[f'{sym}_price'] = buy_price
                            sell_results[f'{sym}_price'] = sell_price
                            found_data = True
                            break
                        except Exception:
                            continue
                
                if not found_data:
                    buy_results[f'{sym}_price'] = None
                    sell_results[f'{sym}_price'] = None

    return buy_results, sell_results

def get_yesterday_profit(today_date: str, yesterday_buy_prices: Dict[str, Optional[float]], yesterday_sell_prices: Dict[str, Optional[float]], yesterday_init_position: Dict[str, float]) -> Dict[str, float]:
    """
    获取今日开盘时持仓的收益，收益计算方式为：(昨日收盘价格 - 昨日开盘价格)*当前持仓。
    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        yesterday_buy_prices: 昨日开盘价格字典，格式为 {symbol_price: price}
        yesterday_sell_prices: 昨日收盘价格字典，格式为 {symbol_price: price}
        yesterday_init_position: 昨日初始持仓字典，格式为 {symbol: weight}

    Returns:
        {symbol: profit} 的字典；若未找到对应日期或标的，则值为 0.0。
    """
    profit_dict = {}
    
    # 遍历所有股票代码
    for symbol in all_nasdaq_100_symbols:
        symbol_price_key = f'{symbol}_price'
        
        # 获取昨日开盘价和收盘价
        buy_price = yesterday_buy_prices.get(symbol_price_key)
        sell_price = yesterday_sell_prices.get(symbol_price_key)
        
        # 获取昨日持仓权重
        position_data = yesterday_init_position.get(symbol, {})
        if isinstance(position_data, dict):
            position_weight = position_data.get('shares', 0.0)
        else:
            # 兼容旧格式
            position_weight = float(position_data)

        # 计算收益：(收盘价 - 开盘价) * 持仓权重
        if buy_price is not None and sell_price is not None and position_weight > 0:
            profit = (sell_price - buy_price) * position_weight
            profit_dict[symbol] = round(profit, 4)  # 保留4位小数
        else:
            profit_dict[symbol] = 0.0
    
    return profit_dict

def get_today_init_position(today_date: str, modelname: str) -> Dict[str, any]:
    """
    获取今日开盘时的初始持仓（即文件中上一个交易日代表的持仓）。从../data/agent_data/{modelname}/position/position.jsonl中读取。
    如果同一日期有多条记录，选择id最大的记录作为初始持仓。
    
    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        modelname: 模型名称，用于构建文件路径。

    Returns:
        {symbol: {"shares": ..., "purchase_date": ...}} 的字典；若未找到对应日期，则返回空字典。
    """
    base_dir = Path(__file__).resolve().parents[1]
    position_file = base_dir / "data" / "agent_data" / modelname / "position" / "position.jsonl"

    if not position_file.exists():
        print(f"Position file {position_file} does not exist")
        return {}
    
    yesterday_date = get_yesterday_date(today_date)
    max_id = -1
    latest_positions = {}
  
    with position_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if doc.get("date") == yesterday_date:
                    current_id = doc.get("id", 0)
                    if current_id > max_id:
                        max_id = current_id
                        latest_positions = normalize_positions(doc.get("positions", {}))
            except Exception:
                continue
    
    return copy.deepcopy(latest_positions)

def get_latest_position(today_date: str, modelname: str) -> tuple[Dict[str, any], int, Optional[Dict[str, Any]]]:
    """
    获取最新持仓。从 ../data/agent_data/{modelname}/position/position.jsonl 中读取。
    优先选择当天 (today_date) 中 id 最大的记录；
    若当天无记录，则回退到上一个交易日，选择该日中 id 最大的记录。

    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        modelname: 模型名称，用于构建文件路径。

    Returns:
        (positions, max_id):
          - positions: {symbol: {"shares": ..., "purchase_date": ...}} 的字典；若未找到任何记录，则为空字典。
          - max_id: 选中记录的最大 id；若未找到任何记录，则为 -1。
    """
    base_dir = Path(__file__).resolve().parents[1]
    position_file = base_dir / "data" / "agent_data" / modelname / "position" / "position.jsonl"

    if not position_file.exists():
        return {}, -1, None
    
    # 先尝试读取当天记录
    max_id_today = -1
    latest_positions_today: Dict[str, any] = {}
    
    latest_record_today: Optional[Dict[str, Any]] = None
    latest_record_prev: Optional[Dict[str, Any]] = None

    with position_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if doc.get("date") == today_date:
                    current_id = doc.get("id", -1)
                    if current_id > max_id_today:
                        max_id_today = current_id
                        latest_positions_today = normalize_positions(doc.get("positions", {}))
                        latest_record_today = copy.deepcopy(doc)
            except Exception:
                continue
    
    if max_id_today >= 0:
        return copy.deepcopy(latest_positions_today), max_id_today, latest_record_today

    # 当天没有记录，则回退到上一个交易日
    prev_date = get_yesterday_date(today_date)
    max_id_prev = -1
    latest_positions_prev: Dict[str, any] = {}

    with position_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if doc.get("date") == prev_date:
                    current_id = doc.get("id", -1)
                    if current_id > max_id_prev:
                        max_id_prev = current_id
                        latest_positions_prev = normalize_positions(doc.get("positions", {}))
                        latest_record_prev = copy.deepcopy(doc)
            except Exception:
                continue

    return copy.deepcopy(latest_positions_prev), max_id_prev, latest_record_prev


def upsert_position_record(modelname: str, record: Dict[str, Any]) -> None:
    """
    在 position.jsonl 中根据 (date + decision_time) 覆盖或追加记录。
    如果 record 已包含 id，则复用；否则分配新的递增 id。
    """
    base_dir = Path(__file__).resolve().parents[1]
    position_path = base_dir / "data" / "agent_data" / modelname / "position"
    position_file = position_path / "position.jsonl"

    os.makedirs(position_path, exist_ok=True)

    target_date = record.get("date")
    target_decision_time = record.get("decision_time")

    existing_records: List[Dict[str, Any]] = []
    existing_id: Optional[int] = record.get("id")
    max_id = -1

    if position_file.exists():
        with position_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except Exception:
                    continue

                current_id = doc.get("id", -1)
                if current_id > max_id:
                    max_id = current_id

                if (
                    doc.get("date") == target_date
                    and doc.get("decision_time") == target_decision_time
                ):
                    if existing_id is None:
                        existing_id = current_id
                    continue

                normalized_doc = copy.deepcopy(doc)
                normalized_doc["positions"] = normalize_positions(normalized_doc.get("positions", {}))
                normalized_doc["decision_time"] = normalize_decision_time(
                    normalized_doc.get("date", ""), normalized_doc.get("decision_time")
                )
                action = normalized_doc.get("this_action")
                if isinstance(action, dict):
                    action_symbol = normalize_symbol(action.get("symbol"))
                    action["symbol"] = action_symbol or ""
                existing_records.append(normalized_doc)

    clean_record = copy.deepcopy(record)
    if existing_id is None:
        clean_record["id"] = max_id + 1 if max_id >= 0 else 1
    else:
        clean_record["id"] = existing_id

    clean_record["positions"] = normalize_positions(clean_record.get("positions", {}))
    clean_record["decision_time"] = normalize_decision_time(
        clean_record.get("date", ""), clean_record.get("decision_time")
    )

    action = clean_record.get("this_action")
    if isinstance(action, dict):
        action_symbol = normalize_symbol(action.get("symbol"))
        action["symbol"] = action_symbol or ""

    existing_records.append(clean_record)

    def _sort_key(item: Dict[str, Any]):
        return (
            item.get("date", ""),
            item.get("decision_time", ""),
            item.get("id", 0),
        )

    existing_records.sort(key=_sort_key)

    with position_file.open("w", encoding="utf-8") as f:
        for doc in existing_records:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

def add_no_trade_record(today_date: str, decision_time: str, decision_count: int, modelname: str):
    """
    添加不交易记录。从 ../data/agent_data/{modelname}/position/position.jsonl 中前一日最后一条持仓，并更新在今日的position.jsonl文件中。
    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        modelname: 模型名称，用于构建文件路径。

    Returns:
        None
    """
    current_position, current_action_id, latest_record = get_latest_position(today_date, modelname)
    sanitized_positions = normalize_positions(current_position)
    decision_time = normalize_decision_time(today_date, decision_time)

    save_item: Dict[str, Any] = {
        "date": today_date,
        "decision_time": decision_time,
        "decision_count": decision_count,
        "this_action": {
            "action": "no_trade",
            "symbol": "",
            "amount": 0,
        },
        "positions": sanitized_positions,
    }

    if latest_record and latest_record.get("decision_time") == decision_time:
        save_item["id"] = latest_record.get("id", current_action_id)
    else:
        save_item["id"] = current_action_id + 1

    upsert_position_record(modelname, save_item)
    return 

def get_price_limits(symbol: str, previous_close: Optional[float]) -> Dict[str, Any]:
    """
    根据股票代码推断涨跌停幅度，并结合前收盘价计算上下限。
    - 科创板、创业板默认 ±20%（代码前缀 688/689/300/301/302）
    - ST/*ST 默认 ±5%（代码或名称包含 ST）
    - 其余默认 ±10%
    """
    normalized_symbol = normalize_symbol(symbol) or symbol
    plain_symbol = strip_exchange_prefix(normalized_symbol) or normalized_symbol
    symbol_upper = (normalized_symbol or "").upper()

    category = "default"
    limit_pct = 0.10

    if plain_symbol.startswith(("688", "689", "300", "301", "302")):
        limit_pct = 0.20
        category = "star_chinext"
    elif symbol_upper.startswith(("ST", "*S")) or "ST" in symbol_upper:
        # 粗略识别 ST/*ST 股票
        limit_pct = 0.05
        category = "st"

    upper = lower = None
    if previous_close is not None:
        try:
            upper = round(float(previous_close) * (1 + limit_pct), 2)
            lower = round(float(previous_close) * (1 - limit_pct), 2)
        except Exception:
            upper = lower = None

    return {
        "symbol": symbol_upper,
        "plain_symbol": plain_symbol,
        "limit_pct": limit_pct,
        "category": category,
        "previous_close": previous_close,
        "upper": upper,
        "lower": lower,
    }

if __name__ == "__main__":
    today_date = get_config_value("TODAY_DATE")
    signature = get_config_value("SIGNATURE")
    if signature is None:
        raise ValueError("SIGNATURE environment variable is not set")
    print(today_date, signature)
    yesterday_date = get_yesterday_date(today_date)
    # print(yesterday_date)
    today_buy_price = get_open_prices(today_date, all_nasdaq_100_symbols)
    # print(today_buy_price)
    yesterday_buy_prices, yesterday_sell_prices = get_yesterday_open_and_close_price(today_date, all_nasdaq_100_symbols)
    # print(yesterday_buy_prices)
    # print(yesterday_sell_prices)
    today_init_position = get_today_init_position(today_date, signature)
    # print(today_init_position)
    latest_position, latest_action_id, latest_record = get_latest_position(today_date, signature)
    print(latest_position, latest_action_id, latest_record)
    yesterday_profit = get_yesterday_profit(today_date, yesterday_buy_prices, yesterday_sell_prices, today_init_position)
    # print(yesterday_profit)
    add_no_trade_record(today_date, signature)
