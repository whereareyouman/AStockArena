import copy
import os
import time
from dotenv import load_dotenv
load_dotenv()
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

# 将项目根目录加入 Python 路径，便于从子目录直接运行本文件
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.runtime_config import get_runtime_config_value

try:
    import fcntl  # type: ignore

    HAS_FCNTL = True
except ImportError:  # pragma: no cover
    HAS_FCNTL = False

try:
    import msvcrt  # type: ignore

    HAS_MSVCRT = True
except ImportError:  # pragma: no cover
    HAS_MSVCRT = False


class _FileLock:
    """
    简单跨平台文件锁（跨进程），用于保护 position.jsonl 的读-改-写（避免并发写入导致 tmp 竞态 / 丢更新）。
    - timeout=0 表示无限等待
    """

    def __init__(self, path: str, timeout: float = 0.0):
        self.path = path
        self.timeout = timeout
        self._handle = None

    def acquire(self, timeout: float = 0.0) -> bool:
        start = time.time()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._handle = open(self.path, "w+", encoding="utf-8")
        while True:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                if HAS_MSVCRT:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                    return True
                return True  # fallback: no real lock available
            except (BlockingIOError, OSError):
                if timeout and (time.time() - start) >= timeout:
                    return False
                time.sleep(0.2)

    def release(self) -> None:
        if not self._handle:
            return
        try:
            if HAS_FCNTL:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
            elif HAS_MSVCRT:
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __enter__(self):
        if not self.acquire(timeout=self.timeout):
            raise TimeoutError(f"Timed out acquiring lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


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


def _normalize_position_entry(data: Any) -> Tuple[int, Optional[str], Optional[float], List[Dict[str, Any]]]:
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
        lots = normalize_lots(data, fallback_shares=shares, fallback_purchase_date=purchase_date, fallback_avg_price=avg_price)
        return shares, purchase_date, avg_price, lots

    try:
        shares = int(float(data))
    except Exception:
        shares = 0
    lots = normalize_lots(None, fallback_shares=shares, fallback_purchase_date=None, fallback_avg_price=None)
    return shares, None, None, lots


def normalize_lots(
    data: Any,
    fallback_shares: int = 0,
    fallback_purchase_date: Optional[str] = None,
    fallback_avg_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Normalize lot-level holdings while preserving legacy position entries."""
    raw_lots = data.get("lots") if isinstance(data, dict) else None
    normalized: List[Dict[str, Any]] = []
    if isinstance(raw_lots, list):
        for lot in raw_lots:
            if not isinstance(lot, dict):
                continue
            try:
                shares = int(lot.get("shares", 0) or 0)
            except Exception:
                shares = 0
            if shares <= 0:
                continue
            purchase_date = lot.get("purchase_date")
            if purchase_date in (None, "", "null"):
                purchase_date = fallback_purchase_date
            else:
                purchase_date = str(purchase_date)
            price_raw = lot.get("avg_price", fallback_avg_price)
            avg_price = None
            if price_raw is not None:
                try:
                    avg_price = float(price_raw)
                except Exception:
                    avg_price = fallback_avg_price
            normalized.append({
                "shares": shares,
                "purchase_date": purchase_date,
                "avg_price": avg_price,
            })

    if not normalized and fallback_shares > 0:
        normalized.append({
            "shares": int(fallback_shares),
            "purchase_date": fallback_purchase_date,
            "avg_price": fallback_avg_price,
        })
    return normalized


def summarize_lots(lots: List[Dict[str, Any]]) -> Tuple[int, Optional[str], Optional[float]]:
    """Return total shares, earliest purchase date, and weighted average price."""
    total_shares = 0
    earliest_date: Optional[str] = None
    weighted_cost = 0.0
    priced_shares = 0
    for lot in lots:
        try:
            shares = int(lot.get("shares", 0) or 0)
        except Exception:
            shares = 0
        if shares <= 0:
            continue
        total_shares += shares
        purchase_date = lot.get("purchase_date")
        if purchase_date:
            purchase_date = str(purchase_date)
            earliest_date = purchase_date if earliest_date is None else min(earliest_date, purchase_date)
        avg_price = lot.get("avg_price")
        if avg_price is not None:
            try:
                weighted_cost += float(avg_price) * shares
                priced_shares += shares
            except Exception:
                pass
    weighted_avg = (weighted_cost / priced_shares) if priced_shares > 0 else None
    return total_shares, earliest_date, weighted_avg


def get_available_sell_shares(position_entry: Any, today_date: str) -> int:
    """Return shares eligible for T+1 selling based on lot purchase dates."""
    shares, purchase_date, avg_price, lots = _normalize_position_entry(position_entry)
    if shares <= 0:
        return 0
    available = 0
    for lot in lots:
        lot_date = lot.get("purchase_date")
        try:
            lot_shares = int(lot.get("shares", 0) or 0)
        except Exception:
            lot_shares = 0
        if lot_shares <= 0:
            continue
        if not lot_date or str(lot_date) < str(today_date):
            available += lot_shares
    return available


def remove_shares_from_lots(position_entry: Any, amount: int, today_date: str) -> List[Dict[str, Any]]:
    """Remove shares from sellable lots FIFO-style, preserving same-day locked lots."""
    _, _, _, lots = _normalize_position_entry(position_entry)
    remaining_to_sell = int(amount)
    updated: List[Dict[str, Any]] = []
    for lot in lots:
        lot_copy = copy.deepcopy(lot)
        try:
            lot_shares = int(lot_copy.get("shares", 0) or 0)
        except Exception:
            lot_shares = 0
        if lot_shares <= 0:
            continue
        lot_date = lot_copy.get("purchase_date")
        sellable = (not lot_date) or str(lot_date) < str(today_date)
        if sellable and remaining_to_sell > 0:
            sold = min(lot_shares, remaining_to_sell)
            lot_shares -= sold
            remaining_to_sell -= sold
        if lot_shares > 0:
            lot_copy["shares"] = lot_shares
            updated.append(lot_copy)
    if remaining_to_sell > 0:
        raise ValueError("可卖持仓数量不足")
    return updated


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
        # 如果持仓为空，返回初始现金（而不是0）
        initial_cash = _load_initial_cash()
        return {"CASH": initial_cash}

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
        shares, purchase_date, avg_price, lots = _normalize_position_entry(data)
        if shares <= 0 or normalized_symbol is None:
            continue

        existing = accumulator.get(normalized_symbol)
        if existing:
            existing["lots"].extend(lots)
            total_shares, earliest_date, weighted_avg = summarize_lots(existing["lots"])
            existing["shares"] = total_shares
            existing["purchase_date"] = earliest_date
            existing["avg_price"] = weighted_avg
        else:
            total_shares, earliest_date, weighted_avg = summarize_lots(lots)
            accumulator[normalized_symbol] = {
                "shares": total_shares,
                "purchase_date": earliest_date,
                "avg_price": weighted_avg,
                "lots": lots,
            }

    normalized.update(accumulator)
    return normalized


def _load_initial_cash() -> float:
    """Load initial_cash from settings/default_config.json (fallback to 1000000)."""
    try:
        base_dir = Path(__file__).resolve().parents[1]
        config_file = base_dir / "settings" / "default_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            agent_cfg = (cfg or {}).get("agent_config", {})
            return float(agent_cfg.get("initial_cash", 1000000.0))
    except Exception:
        pass
    return 1000000.0


def _load_log_path() -> Path:
    """Load canonical trading log path from settings/default_config.json."""
    base_dir = Path(__file__).resolve().parents[1]
    try:
        config_file = base_dir / "settings" / "default_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            raw_path = (cfg or {}).get("log_config", {}).get("log_path", "./data_flow/trading_summary_each_agent")
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path
            return path
    except Exception:
        pass
    return base_dir / "data_flow" / "trading_summary_each_agent"


def get_position_file_path(modelname: str, for_write: bool = False) -> Path:
    """Return canonical position path, with read-only fallback for old root-level logs."""
    base_dir = Path(__file__).resolve().parents[1]
    canonical = _load_log_path() / modelname / "position" / "position.jsonl"
    if for_write or canonical.exists():
        return canonical
    legacy = base_dir / "trading_summary_each_agent" / modelname / "position" / "position.jsonl"
    if legacy.exists():
        return legacy
    return canonical


@contextmanager
def file_transaction_lock(path: Any, suffix: str = ".txn.lock", timeout: float = 0.0):
    """Lock a full read-calculate-write transaction for an arbitrary file."""

    target = Path(path)
    os.makedirs(target.parent, exist_ok=True)
    lock_file = target.with_name(target.name + suffix)
    with _FileLock(str(lock_file), timeout=timeout):
        yield


@contextmanager
def position_transaction_lock(modelname: str, timeout: float = 0.0):
    """Lock a model's full read-calculate-write position transaction."""

    position_file = get_position_file_path(modelname, for_write=True)
    with file_transaction_lock(position_file, suffix=".txn.lock", timeout=timeout):
        yield


def calculate_previous_trading_date(today_date: str) -> str:
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

def get_initial_position_for_date(today_date: str, modelname: str) -> Dict[str, any]:
    """
    获取今日开盘时的初始持仓（即文件中上一个交易日代表的持仓）。从../data_flow/trading_summary_each_agent/{modelname}/position/position.jsonl中读取。
    如果同一日期有多条记录，选择id最大的记录作为初始持仓。
    如果找不到上一个交易日的记录，则查找文件中最后一条记录；如果文件为空或没有任何记录，返回初始现金持仓。
    
    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        modelname: 模型名称，用于构建文件路径。

    Returns:
        {symbol: {"shares": ..., "purchase_date": ...}} 的字典；若未找到任何记录，则返回初始现金持仓。
    """
    position_file = get_position_file_path(modelname)

    if not position_file.exists():
        print(f"Position file {position_file} does not exist")
        # 如果持仓文件不存在，返回初始现金持仓
        initial_cash = _load_initial_cash()
        return {"CASH": initial_cash}
    
    yesterday_date = calculate_previous_trading_date(today_date)
    max_id = -1
    latest_positions = {}
    
    # 用于查找最后一条记录的备用变量
    last_record_date = None
    last_record_id = -1
    last_record_positions = {}
  
    with position_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                doc_date = doc.get("date")
                
                # 优先查找上一个交易日的记录
                if doc_date == yesterday_date:
                    current_id = doc.get("id", 0)
                    if current_id > max_id:
                        max_id = current_id
                        latest_positions = normalize_positions(doc.get("positions", {}))
                
                # 同时记录所有记录中日期在 today_date 之前的最后一条记录（作为备用）
                if doc_date and doc_date < today_date:
                    current_id = doc.get("id", -1)
                    if last_record_date is None or doc_date > last_record_date or (doc_date == last_record_date and current_id > last_record_id):
                        last_record_date = doc_date
                        last_record_id = current_id
                        last_record_positions = normalize_positions(doc.get("positions", {}))
            except Exception:
                continue
    
    # 如果找到了上一个交易日的记录，返回它
    if latest_positions:
        return copy.deepcopy(latest_positions)
    
    # 如果找不到上一个交易日的记录，尝试使用最后一条记录
    if last_record_positions:
        return copy.deepcopy(last_record_positions)
    
    # 如果完全没有记录，返回初始现金持仓
    initial_cash = _load_initial_cash()
    return {"CASH": initial_cash}

def get_current_position(today_date: str, modelname: str) -> tuple[Dict[str, any], int, Optional[Dict[str, Any]]]:
    """
    获取最新持仓。从 ../data_flow/trading_summary_each_agent/{modelname}/position/position.jsonl 中读取。
    优先选择当天 (today_date) 中 id 最大的记录；
    若当天无记录，则回退到上一个交易日，选择该日中 id 最大的记录。

    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        modelname: 模型名称，用于构建文件路径。

    Returns:
        (positions, max_id, latest_record):
          - positions: {symbol: {"shares": ..., "purchase_date": ...}} 的字典；若未找到任何记录，则为空字典。
          - max_id: 选中记录的最大 id；若未找到任何记录，则为 -1。
          - latest_record: 最新的完整记录，如果不存在则为 None
    """
    position_file = get_position_file_path(modelname)

    if not position_file.exists():
        # 如果持仓文件不存在，返回初始现金持仓
        initial_cash = _load_initial_cash()
        return {"CASH": initial_cash}, -1, None
    
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

    # 当天没有记录，则优先回退到上一个工作日；如果该工作日没有记录，
    # 继续回退到文件中早于 today_date 的最后一条记录。
    # 这覆盖 A 股节假日/临时休市导致多个自然工作日没有 position 记录的情况。
    prev_date = calculate_previous_trading_date(today_date)
    max_id_prev = -1
    latest_positions_prev: Dict[str, any] = {}
    latest_record_any_prev: Optional[Dict[str, Any]] = None
    latest_positions_any_prev: Dict[str, any] = {}
    latest_any_key: Optional[tuple[str, str, int]] = None

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

                doc_date = doc.get("date")
                if doc_date and str(doc_date) < str(today_date):
                    try:
                        current_id = int(doc.get("id", -1))
                    except Exception:
                        current_id = -1
                    decision_time = normalize_decision_time(doc_date, doc.get("decision_time"))
                    key = (str(doc_date), decision_time, current_id)
                    if latest_any_key is None or key > latest_any_key:
                        latest_any_key = key
                        latest_positions_any_prev = normalize_positions(doc.get("positions", {}))
                        latest_record_any_prev = copy.deepcopy(doc)
            except Exception:
                continue

    if max_id_prev >= 0 or latest_positions_prev:
        return copy.deepcopy(latest_positions_prev), max_id_prev, latest_record_prev

    if latest_any_key is not None and latest_positions_any_prev:
        return (
            copy.deepcopy(latest_positions_any_prev),
            latest_any_key[2],
            latest_record_any_prev,
        )

    # 如果找不到任何记录（既没有当天的记录，也没有任何过去记录），返回初始现金持仓
    initial_cash = _load_initial_cash()
    return {"CASH": initial_cash}, -1, None


def _normalize_action_payload(action: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return None
    normalized = copy.deepcopy(action)
    normalized["action"] = str(normalized.get("action") or "").strip()
    action_symbol = normalize_symbol(normalized.get("symbol"))
    normalized["symbol"] = action_symbol or ""
    return normalized


def _actions_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = record.get("actions")
    if isinstance(actions, list):
        normalized_actions = []
        for action in actions:
            normalized = _normalize_action_payload(action)
            if normalized:
                normalized_actions.append(normalized)
        return normalized_actions
    action = _normalize_action_payload(record.get("this_action"))
    if action and action.get("action") != "seed":
        return [action]
    return []


def _fills_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    fills = record.get("fills")
    if isinstance(fills, list):
        normalized_fills = []
        for fill in fills:
            normalized = _normalize_action_payload(fill)
            if normalized and normalized.get("action") in {"buy", "sell"}:
                normalized_fills.append(normalized)
        return normalized_fills
    action = _normalize_action_payload(record.get("this_action"))
    if action and action.get("action") in {"buy", "sell"}:
        return [action]
    return []


def _summarize_record_action(actions: List[Dict[str, Any]], fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(fills) == 1:
        fill = fills[0]
        return {
            "action": fill.get("action"),
            "symbol": fill.get("symbol", ""),
            "amount": fill.get("amount", 0),
        }
    if len(fills) > 1:
        return {
            "action": "multi_trade",
            "symbol": "",
            "amount": 0,
            "actions_count": len(fills),
        }
    if actions:
        latest = actions[-1]
        return {
            "action": latest.get("action", "no_trade"),
            "symbol": latest.get("symbol", ""),
            "amount": latest.get("amount", 0),
        }
    return {"action": "no_trade", "symbol": "", "amount": 0}


def upsert_position_record(modelname: str, record: Dict[str, Any]) -> None:
    """
    在 position.jsonl 中根据 (date + decision_time) 覆盖或追加记录。
    如果 record 已包含 id，则复用；否则分配新的递增 id。
    """
    position_file = get_position_file_path(modelname, for_write=True)
    position_path = position_file.parent

    os.makedirs(position_path, exist_ok=True)

    # 跨进程锁：保护 position.jsonl 的读-改-写，避免并发写导致 tmp 被抢/丢更新。
    lock_file = position_file.with_name(position_file.name + ".lock")
    with _FileLock(str(lock_file), timeout=0.0):
        target_date = record.get("date")
        target_decision_time = normalize_decision_time(target_date, record.get("decision_time"))

        existing_records: List[Dict[str, Any]] = []
        existing_id: Optional[int] = record.get("id")
        matching_records: List[Dict[str, Any]] = []
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

                    doc_date = doc.get("date")
                    doc_decision_time = normalize_decision_time(doc_date, doc.get("decision_time"))
                    if doc_date == target_date and doc_decision_time == target_decision_time:
                        # 已存在同一 (date + decision_time) 的记录：用新的 record 覆盖它
                        if existing_id is None:
                            existing_id = current_id
                        normalized_match = copy.deepcopy(doc)
                        normalized_match["positions"] = normalize_positions(normalized_match.get("positions", {}))
                        normalized_match["decision_time"] = doc_decision_time
                        matching_records.append(normalized_match)
                        continue

                    normalized_doc = copy.deepcopy(doc)
                    normalized_doc["positions"] = normalize_positions(normalized_doc.get("positions", {}))
                    normalized_doc["decision_time"] = doc_decision_time
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
        clean_record["decision_time"] = target_decision_time

        action = clean_record.get("this_action")
        if isinstance(action, dict):
            action_symbol = normalize_symbol(action.get("symbol"))
            action["symbol"] = action_symbol or ""

        clean_run_id = clean_record.get("ledger_run_id")
        if clean_run_id:
            matching_records_for_merge = [
                matched for matched in matching_records
                if matched.get("ledger_run_id") == clean_run_id
            ]
        else:
            # Older records may not carry a run id. Preserve compatibility for
            # those records while keeping new runs separated by run id.
            matching_records_for_merge = matching_records

        merged_actions: List[Dict[str, Any]] = []
        merged_fills: List[Dict[str, Any]] = []
        for matched in matching_records_for_merge:
            merged_actions.extend(_actions_from_record(matched))
            merged_fills.extend(_fills_from_record(matched))
        merged_actions.extend(_actions_from_record(clean_record))
        merged_fills.extend(_fills_from_record(clean_record))
        if merged_actions:
            clean_record["actions"] = merged_actions
        if merged_fills:
            clean_record["fills"] = merged_fills
        if merged_actions or merged_fills:
            clean_record["this_action"] = _summarize_record_action(merged_actions, merged_fills)

        existing_records.append(clean_record)

        def _sort_key(item: Dict[str, Any]):
            return (
                item.get("date", ""),
                item.get("decision_time", ""),
                item.get("id", 0),
            )

        existing_records.sort(key=_sort_key)

        # 原子写入：使用唯一 tmp 文件名，配合锁避免并发抢同名 tmp
        tmp_file = position_file.with_name(position_file.name + f".tmp.{os.getpid()}")
        try:
            with tmp_file.open("w", encoding="utf-8") as f:
                for doc in existing_records:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_file, position_file)
        finally:
            # 清理残留 tmp（如写入失败）
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass

def add_no_trade_record(
    today_date: str,
    decision_time: str,
    decision_count: int,
    modelname: str,
    ledger_run_id: Optional[str] = None,
):
    """
    添加不交易记录。从 ../data_flow/trading_summary_each_agent/{modelname}/position/position.jsonl 中前一日最后一条持仓，并更新在今日的position.jsonl文件中。
    Args:
        today_date: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        decision_time: 决策时间字符串
        decision_count: 决策序号
        modelname: 模型名称，用于构建文件路径。

    Returns:
        None
    """
    current_position, current_action_id, latest_record = get_current_position(today_date, modelname)
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
        "actions": [
            {
                "action": "no_trade",
                "symbol": "",
                "amount": 0,
                "decision_time": decision_time,
                "decision_count": decision_count,
            }
        ],
        "fills": [],
        "positions": sanitized_positions,
    }
    if ledger_run_id:
        save_item["ledger_run_id"] = ledger_run_id

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


# 向后兼容别名（已弃用，将在未来版本中移除）
def get_yesterday_date(today_date: str) -> str:
    """Deprecated: Use calculate_previous_trading_date instead."""
    return calculate_previous_trading_date(today_date)

def get_today_init_position(today_date: str, modelname: str) -> Dict[str, any]:
    """Deprecated: Use get_initial_position_for_date instead."""
    return get_initial_position_for_date(today_date, modelname)

def get_latest_position(today_date: str, modelname: str) -> tuple[Dict[str, any], int, Optional[Dict[str, Any]]]:
    """Deprecated: Use get_current_position instead."""
    return get_current_position(today_date, modelname)
