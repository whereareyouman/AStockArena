# -*- coding: utf-8 -*-
"""高性能交易 API 服务器

架构设计：读写分离与后台同步 (CQRS-lite)
- 内存状态源 (Single Source of Truth)：GlobalState 单例
- 双路后台工人：MarketWorker + PortfolioWorker
- 非阻塞架构：API 接口永远秒回缓存数据 (Stale-While-Revalidate)
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from agent_engine.agent.agent import AgenticWorkflow as BaseAgent

# --- 配置与常量 ---
LOG_DIR = Path(__file__).parent / "logs" / "jobs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data_flow"
CONFIG_DIR = Path(__file__).parent / "settings"
DEFAULT_CONFIG = CONFIG_DIR / "default_config.json"
RUNTIME_ENV = Path(__file__).parent / "runtime_env.json"
DATA_FILE = DATA_DIR / "ai_stock_data.json"

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("API")


# =============================================================================
# 全局内存状态 (The In-Memory State) - 核心优化
# =============================================================================
class GlobalState:
    """
    全局内存状态容器。
    所有 API 读取请求直接从这里获取数据，不再进行 IO 操作。
    """
    def __init__(self):
        # 行情数据
        self.market_quotes: Dict[str, Dict[str, Any]] = {}  # 符号 -> 行情快照
        self.monitored_symbols: Set[str] = set()            # 需要拉取行情的股票池
        
        # 持仓与统计
        self.portfolios: Dict[str, Dict[str, Any]] = {}     # 模型签名 -> 资产概况
        self.model_stats: Dict[str, Dict[str, Any]] = {}    # 模型签名 -> 完整统计
        self.position_records: Dict[str, List[Dict]] = {}   # 签名 -> 历史记录缓存
        self.position_mtimes: Dict[str, float] = {}         # 文件修改时间追踪
        
        # 大文件缓存
        self.stock_history_cache: Dict[str, Any] = {}       # ai_stock_data.json 缓存
        self.stock_history_mtime: float = 0
        
        # 任务管理
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # 系统状态
        self.system_status: Dict[str, Any] = {
            "initialized": False,
            "market_worker_running": False,
            "portfolio_worker_running": False,
            "last_market_update": None,
            "last_portfolio_update": None,
        }
        
        # 系统配置（从 default_config.json 加载）
        self.system_config: Dict[str, Any] = {}
        
        # 锁
        self._quote_lock = asyncio.Lock()
        self._portfolio_lock = asyncio.Lock()

    def update_quote(self, symbol: str, data: Dict[str, Any]):
        self.market_quotes[symbol] = data

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.market_quotes.get(symbol)

    def update_portfolio(self, signature: str, data: Dict[str, Any]):
        self.portfolios[signature] = data

    def get_portfolio(self, signature: str) -> Optional[Dict[str, Any]]:
        return self.portfolios.get(signature)

    def update_model_stats(self, signature: str, data: Dict[str, Any]):
        self.model_stats[signature] = data

    def get_model_stats(self, signature: str) -> Optional[Dict[str, Any]]:
        return self.model_stats.get(signature)


# 全局单例
APP_STATE = GlobalState()


# =============================================================================
# 辅助函数
# =============================================================================
def _truthy_env(name: str) -> bool:
    return os.getenv(name, "false").lower() in ("1", "true", "yes")


def _normalize_code(c: str) -> str:
    c = (c or "").strip().upper()
    if len(c) == 6 and c.isdigit():
        if c.startswith(("688", "689", "600", "601", "603", "605", "730", "735")):
            return f"SH{c}"
        if c.startswith(("000", "001", "002", "003", "300", "301", "302")):
            return f"SZ{c}"
    return c


def _symbol_candidates(symbol: str) -> List[str]:
    norm = (symbol or "").upper()
    plain = norm
    if norm.startswith(("SH", "SZ")) and len(norm) > 2:
        plain = norm[2:]
    candidates = {norm, plain, f"SH{plain}", f"SZ{plain}"}
    return [c for c in candidates if c]


def _load_config_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"配置文件解析失败: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置文件读取失败: {e}")


def _load_config_dict_safe(path: Path) -> Dict[str, Any]:
    """不抛异常的版本，用于后台 worker"""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _get_enabled_signatures(config_path: Optional[Path] = None) -> List[str]:
    path = config_path or DEFAULT_CONFIG
    cfg = _load_config_dict_safe(path)
    return [
        m["signature"]
        for m in cfg.get("models", [])
        if m.get("enabled", True) and m.get("signature")
    ]


def _get_tsl_credentials() -> tuple:
    user = os.getenv("TSL_USERNAME") or os.getenv("TSL_USER")
    pwd = os.getenv("TSL_PASSWORD") or os.getenv("TSL_PASS")
    server = os.getenv("TSL_SERVER", "tsl.tinysoft.com.cn")
    try:
        port = int(os.getenv("TSL_PORT", "443"))
    except Exception:
        port = 443
    return user, pwd, server, port


def _load_runtime_signature() -> str | None:
    try:
        if RUNTIME_ENV.exists():
            with open(RUNTIME_ENV, "r", encoding="utf-8") as f:
                rt = json.load(f)
            return rt.get("SIGNATURE")
    except Exception:
        return None
    return None


def _position_file_for_signature(signature: str | None) -> Path:
    sig = signature or _load_runtime_signature()
    if not sig:
        raise HTTPException(status_code=400, detail="No signature provided and runtime_env SIGNATURE missing")
    return DATA_DIR / "trading_summary_each_agent" / sig / "position" / "position.jsonl"


def _load_initial_cash() -> float:
    """Load initial_cash from settings/default_config.json (fallback to 1000000)."""
    try:
        if DEFAULT_CONFIG.exists():
            with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            agent_cfg = (cfg or {}).get("agent_config", {})
            return float(agent_cfg.get("initial_cash", 1000000.0))
    except Exception:
        pass
    return 1000000.0


def _read_jsonl_tail(path: Path, limit: int = 100) -> list[dict]:
    """高效读取 JSONL 文件尾部"""
    if not path.exists():
        return []
    lines: list[dict] = []
    try:
        with open(path, "rb") as f:
            try:
                # 快速定位到文件尾部
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # 读取最后约 256KB（足以覆盖大多数 limit）
                seek_pos = max(0, size - 256 * 1024)
                f.seek(seek_pos)
                content = f.read().decode("utf-8", errors="ignore")
            except OSError:
                f.seek(0)
                content = f.read().decode("utf-8", errors="ignore")

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
        return lines[-limit:]
    except Exception:
        return []


def _sanitize_provider_detail(detail: str) -> str:
    lowered = detail.lower()
    if "429" in detail or "too many request" in lowered or "rate limit" in lowered:
        return "上游数据源暂时繁忙，请稍后再试。"
    if "login failed" in lowered or "missing" in lowered:
        return "无法连接至行情数据提供方，请稍后重试。"
    return detail


# =============================================================================
# 后台工人 (Background Workers) - 纯本地数据模式
# =============================================================================

# 数据模式配置
# USE_LOCAL_DATA_ONLY=true: 完全使用本地 ai_stock_data.json，适合回测/演示
# USE_LOCAL_DATA_ONLY=false: 尝试连接 TinySoft，失败则降级到本地
USE_LOCAL_DATA_ONLY = os.getenv("USE_LOCAL_DATA_ONLY", "true").lower() in ("1", "true", "yes")


def _find_price_at_time(
    data_list: List[Dict[str, Any]],
    target_datetime: Optional[str],
    target_date: Optional[str],
) -> tuple[float, float, str, str]:
    """
    在历史数据列表中查找最接近目标时间的价格。
    
    策略：
    1. 如果有 target_datetime (如 "2025-09-19 14:00:00")，精确匹配或找最近的 <= 记录
    2. 如果只有 target_date (如 "2025-09-19")，找该日期内最后一条记录
    3. 如果都没有，返回列表最后一条
    
    Returns: (price, volume, data_time, source_type)
    """
    if not data_list:
        return 0.0, 0.0, "", "empty"
    
    # 无目标时间，返回最后一条
    if not target_datetime and not target_date:
        last = data_list[-1]
        price = float(last.get("buy1") or last.get("close") or 0)
        volume = float(last.get("vol") or 0)
        data_time = last.get("date") or last.get("time") or ""
        return price, volume, data_time, "latest"
    
    # 有目标时间，进行匹配
    search_key = target_datetime or target_date
    best_match = None
    
    # 倒序遍历，找第一个 <= search_key 的记录
    for item in reversed(data_list):
        item_time = item.get("date") or item.get("time") or ""
        
        # 对于小时线，格式是 "2025-09-19 14:00:00"
        # 对于日线，格式是 "2025-09-19"
        
        if target_datetime:
            # 精确时间匹配（小时线）
            if item_time <= target_datetime:
                best_match = item
                break
        elif target_date:
            # 日期匹配
            item_date = item_time[:10] if len(item_time) >= 10 else item_time
            if item_date <= target_date:
                best_match = item
                break
    
    if best_match:
        price = float(best_match.get("buy1") or best_match.get("close") or 0)
        volume = float(best_match.get("vol") or 0)
        data_time = best_match.get("date") or best_match.get("time") or ""
        return price, volume, data_time, f"synced@{search_key}"
    
    # 找不到更早的记录，用第一条
    first = data_list[0]
    price = float(first.get("buy1") or first.get("close") or 0)
    volume = float(first.get("vol") or 0)
    data_time = first.get("date") or first.get("time") or ""
    return price, volume, data_time, "earliest"


def _get_current_simulation_time() -> tuple[Optional[str], Optional[str]]:
    """
    从所有模型的持仓记录中，获取当前系统的"模拟时间"。
    
    Returns: (decision_time, date)
        - decision_time: 如 "2025-09-19 14:00:00" (小时级精度)
        - date: 如 "2025-09-19" (日级精度)
    """
    latest_datetime = None
    latest_date = None
    
    # 遍历所有模型，找最新的时间戳
    for sig, portfolio in APP_STATE.portfolios.items():
        dt = portfolio.get("decision_time")
        d = portfolio.get("date")
        
        if dt and (latest_datetime is None or dt > latest_datetime):
            latest_datetime = dt
        if d and (latest_date is None or d > latest_date):
            latest_date = d
    
    # 也检查 position_records 中的原始记录
    for sig, records in APP_STATE.position_records.items():
        if records:
            last = records[-1]
            dt = last.get("decision_time")
            d = last.get("date")
            if dt and (latest_datetime is None or dt > latest_datetime):
                latest_datetime = dt
            if d and (latest_date is None or d > latest_date):
                latest_date = d
    
    return latest_datetime, latest_date


async def market_data_worker():
    """
    行情同步 Worker - 支持双模式切换
    
    模式 1 (USE_LOCAL_DATA_ONLY=true，默认):
        - 完全使用本地 ai_stock_data.json
        - 根据 position.jsonl 的时间戳同步价格
        - 适合回测/演示，避免时间穿越
    
    模式 2 (USE_LOCAL_DATA_ONLY=false):
        - 优先尝试 TinySoft 实时行情
        - 失败则降级到本地缓存
        - 适合实盘监控
    """
    mode_name = "纯本地模式" if USE_LOCAL_DATA_ONLY else "混合模式(TinySoft优先)"
    logger.info(f"🚀 启动行情同步任务 ({mode_name})...")
    APP_STATE.system_status["market_worker_running"] = True
    APP_STATE.system_status["data_mode"] = "local-only" if USE_LOCAL_DATA_ONLY else "hybrid"

    # 混合模式：延迟加载 pyTSL
    ts = None
    client = None
    logged_in = False
    
    if not USE_LOCAL_DATA_ONLY:
        try:
            import pyTSL
            ts = pyTSL
            logger.info("✅ pyTSL 已加载，将尝试获取实时行情")
        except ImportError:
            logger.warning("⚠️ 未找到 pyTSL，将使用本地缓存模式")

    while True:
        try:
            # 1. 获取当前系统的"模拟时间"（本地模式使用）
            sim_datetime, sim_date = _get_current_simulation_time()
            
            # 2. 确定需要更新的股票列表
            symbols = list(APP_STATE.monitored_symbols)
            if not symbols:
                symbols = list(APP_STATE.stock_history_cache.keys())[:50]
                cfg = _load_config_dict_safe(DEFAULT_CONFIG)
                for model in cfg.get("models", []):
                    if model.get("enabled", True):
                        for s in model.get("stock_symbols", []):
                            norm = _normalize_code(s)
                            if norm not in symbols:
                                symbols.append(norm)

            if not symbols:
                await asyncio.sleep(2)
                continue

            quotes_batch: Dict[str, Dict[str, Any]] = {}
            data_source = "local"

            # ========== 模式 2: 混合模式 - 尝试 TinySoft ==========
            if not USE_LOCAL_DATA_ONLY and ts:
                user, pwd, server, port = _get_tsl_credentials()
                if user and pwd:
                    try:
                        # 登录
                        if client is None or not logged_in:
                            client = ts.Client(user, pwd, server, port)
                            login_res = client.login()
                            logged_in = (login_res == 1)
                            if logged_in:
                                logger.info("✅ TinySoft 连接成功")
                            else:
                                logger.warning("⚠️ TinySoft 登录失败，降级到本地模式")

                        # 批量获取实时行情
                        if logged_in:
                            now = datetime.now()
                            begin_time = now - timedelta(days=1)
                            
                            for code in symbols:
                                try:
                                    r = client.query(
                                        stock=code,
                                        begin_time=begin_time,
                                        end_time=now,
                                        cycle='60分钟线',
                                        fields='date, close, vol, amount, buy1'
                                    )
                                    if r.error() == 0:
                                        df = r.dataframe()
                                        if not df.empty:
                                            last = df.iloc[-1]
                                            price = float(last.get('buy1') or last.get('close') or 0)
                                            volume = float(last.get('vol') or 0)
                                            amount = float(last.get('amount') or 0)
                                            
                                            # 计算涨跌幅
                                            change_pct = 0.0
                                            if len(df) >= 2:
                                                prev_close = float(df.iloc[-2].get('close') or 0)
                                                if prev_close > 0:
                                                    change_pct = (price / prev_close - 1) * 100
                                            
                                            quotes_batch[code] = {
                                                "code": code,
                                                "price": round(price, 4),
                                                "volume": volume,
                                                "turnover": round(amount / 1e8, 4),
                                                "changePercent": round(change_pct, 2),
                                                "ts": datetime.utcnow().isoformat() + "Z",
                                                "source": "tinysoft",
                                            }
                                            data_source = "tinysoft"
                                except Exception as e:
                                    logger.debug(f"查询 {code} 失败: {e}")
                    except Exception as e:
                        logger.warning(f"TinySoft 连接异常: {e}")
                        logged_in = False
                        client = None

            # ========== 本地数据处理（纯本地模式 或 TinySoft 未获取到的股票）==========
            sync_mode = "latest" if not sim_datetime and not sim_date else f"synced@{sim_datetime or sim_date}"

            for code in symbols:
                # 如果已经从 TinySoft 获取到了，跳过
                if code in quotes_batch:
                    continue

                # 尝试多种代码格式匹配
                stock_data = None
                for cand in _symbol_candidates(code):
                    stock_data = APP_STATE.stock_history_cache.get(cand)
                    if stock_data:
                        break

                price, volume, data_time, source = 0.0, 0.0, "", "not-found"
                change_pct = 0.0

                if stock_data:
                    hourly = stock_data.get("小时线行情") or []
                    daily = stock_data.get("日线行情") or []

                    # 本地模式：根据模拟时间查找
                    # 混合模式：直接取最新数据
                    if USE_LOCAL_DATA_ONLY:
                        # 纯本地模式：精确时间同步
                        if hourly:
                            price, volume, data_time, source = _find_price_at_time(
                                hourly, sim_datetime, sim_date
                            )
                            source = f"hourly:{source}"
                        elif daily:
                            price, volume, data_time, source = _find_price_at_time(
                                daily, None, sim_date
                            )
                            source = f"daily:{source}"
                    else:
                        # 混合模式：取最新数据作为兜底
                        if hourly:
                            last = hourly[-1]
                            price = float(last.get("buy1") or last.get("close") or 0)
                            volume = float(last.get("vol") or 0)
                            data_time = last.get("date") or ""
                            source = "local-fallback:hourly"
                        elif daily:
                            last = daily[-1]
                            price = float(last.get("close") or 0)
                            volume = float(last.get("vol") or 0)
                            data_time = last.get("date") or ""
                            source = "local-fallback:daily"

                    # 计算涨跌幅
                    target_list = hourly if hourly else daily
                    if len(target_list) >= 2 and data_time:
                        for i, item in enumerate(target_list):
                            if (item.get("date") or "") == data_time and i > 0:
                                prev_close = float(target_list[i-1].get("close") or 0)
                                if prev_close > 0:
                                    change_pct = (price / prev_close - 1) * 100
                                break

                quotes_batch[code] = {
                    "code": code,
                    "price": round(price, 4),
                    "volume": volume,
                    "changePercent": round(change_pct, 2),
                    "ts": data_time if USE_LOCAL_DATA_ONLY else datetime.utcnow().isoformat() + "Z",
                    "source": source,
                    "sim_time": sim_datetime or sim_date if USE_LOCAL_DATA_ONLY else None,
                }

            # 4. 批量更新全局状态
            for code, data in quotes_batch.items():
                APP_STATE.update_quote(code, data)

            APP_STATE.system_status["last_market_update"] = datetime.utcnow().isoformat()
            APP_STATE.system_status["sim_datetime"] = sim_datetime if USE_LOCAL_DATA_ONLY else None
            APP_STATE.system_status["sim_date"] = sim_date if USE_LOCAL_DATA_ONLY else None
            APP_STATE.system_status["sync_mode"] = sync_mode if USE_LOCAL_DATA_ONLY else data_source
            
            logger.debug(f"📊 行情同步完成: {len(quotes_batch)} 只, 模式: {data_source}")

        except Exception as e:
            logger.error(f"❌ 行情同步失败: {e}")

        # 本地模式每秒刷新，混合模式每3秒（减少网络请求）
        await asyncio.sleep(1 if USE_LOCAL_DATA_ONLY else 3)


async def portfolio_watcher_worker():
    """
    后台任务：监视 position.jsonl 文件，计算 PnL 并更新内存状态。
    每秒检查一次文件变化。
    """
    logger.info("🚀 启动持仓监控后台任务...")
    APP_STATE.system_status["portfolio_worker_running"] = True

    initial_cash = _load_initial_cash()

    while True:
        try:
            signatures = _get_enabled_signatures()
            all_held_symbols: Set[str] = set()

            for sig in signatures:
                pos_file = _position_file_for_signature(sig)
                if not pos_file.exists():
                    continue

                # 检查文件是否更新
                try:
                    mtime = pos_file.stat().st_mtime
                except Exception:
                    continue

                # 即使文件未变化，也要重新计算 PnL（因为行情可能变了）
                file_changed = (mtime != APP_STATE.position_mtimes.get(sig))
                
                if file_changed:
                    APP_STATE.position_mtimes[sig] = mtime
                    # 重新读取文件
                    records = _read_jsonl_tail(pos_file, limit=2000)
                    APP_STATE.position_records[sig] = records
                else:
                    records = APP_STATE.position_records.get(sig, [])

                if not records:
                    continue

                latest = records[-1]
                positions = latest.get("positions", {}) or {}

                # 计算资产详情
                cash = float(positions.get("CASH", 0) or 0)
                total_equity = cash
                holdings: List[Dict[str, Any]] = []

                for code, detail in positions.items():
                    if code == "CASH":
                        continue
                    if not isinstance(detail, dict):
                        continue
                    shares = detail.get("shares", 0) or 0
                    if shares <= 0:
                        continue

                    norm_code = _normalize_code(code)
                    all_held_symbols.add(norm_code)

                    # 从全局行情缓存取价格（极速！）
                    quote = APP_STATE.get_quote(norm_code) or {}
                    entry_price = float(detail.get("avg_price") or 0)
                    current_price = quote.get("price") or entry_price or 0

                    market_value = shares * float(current_price)
                    total_equity += market_value
                    cost_basis = shares * entry_price
                    pnl = market_value - cost_basis
                    pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0

                    holdings.append({
                        "symbol": norm_code,
                        "shares": shares,
                        "entry_price": round(entry_price, 2),
                        "current_price": round(float(current_price), 2),
                        "market_value": round(market_value, 2),
                        "pnl": round(pnl, 2),
                        "pnl_percent": round(pnl_percent, 2),
                        "purchase_date": detail.get("purchase_date", ""),
                        "valuation_source": quote.get("source", "fallback"),
                    })

                # 计算统计指标
                total_return_pct = (total_equity / initial_cash - 1.0) * 100.0

                # 计算日收益序列和 Sharpe
                by_date: Dict[str, Dict] = {}
                for rec in records:
                    d = rec.get("date")
                    if not d:
                        continue
                    prev = by_date.get(d)
                    if prev is None or (rec.get("id", -1) > prev.get("id", -1)):
                        by_date[d] = rec

                dates_sorted = sorted(by_date.keys())
                daily_returns = []
                equity_series = []

                for d in dates_sorted:
                    rec = by_date[d]
                    pos = rec.get("positions", {}) or {}
                    eq = float(pos.get("CASH", 0) or 0)
                    for c, det in pos.items():
                        if c == "CASH":
                            continue
                        if isinstance(det, dict):
                            sh = det.get("shares", 0) or 0
                            if sh > 0:
                                nc = _normalize_code(c)
                                q = APP_STATE.get_quote(nc) or {}
                                pr = q.get("price") or det.get("avg_price", 0)
                                eq += sh * float(pr)
                    equity_series.append(eq)

                for i in range(1, len(equity_series)):
                    if equity_series[i-1] > 0:
                        daily_returns.append((equity_series[i] / equity_series[i-1] - 1) * 100)

                # 计算 Sharpe (年化)
                sharpe = 0.0
                if len(daily_returns) > 1:
                    import statistics
                    mean_ret = statistics.mean(daily_returns)
                    std_ret = statistics.stdev(daily_returns)
                    if std_ret > 0:
                        sharpe = (mean_ret * (252 ** 0.5)) / std_ret

                # 计算最大回撤
                peak = initial_cash
                max_dd = 0.0
                for eq in equity_series:
                    if eq > peak:
                        peak = eq
                    dd = (eq / peak - 1.0) * 100.0 if peak > 0 else 0.0
                    if dd < max_dd:
                        max_dd = dd

                # 交易次数
                trade_count = sum(
                    1 for rec in records
                    if (rec.get("this_action") or {}).get("action") in {"buy", "sell"}
                )

                # 更新全局状态
                portfolio_data = {
                    "signature": sig,
                    "date": latest.get("date"),
                    "decision_time": latest.get("decision_time"),
                    "total_equity": round(total_equity, 2),
                    "cash": round(cash, 2),
                    "holdings_count": len(holdings),
                    "holdings": holdings,
                    "updated_at": datetime.utcnow().isoformat(),
                }
                APP_STATE.update_portfolio(sig, portfolio_data)

                stats_data = {
                    "signature": sig,
                    "latest_date": latest.get("date"),
                    "total_return_pct": round(total_return_pct, 2),
                    "sharpe_ratio": round(sharpe, 2),
                    "max_drawdown_pct": round(max_dd, 2),
                    "position_count": len(holdings),
                    "cash": round(cash, 2),
                    "equity": round(total_equity, 2),
                    "last_action": latest.get("this_action"),
                    "total_records": len(records),
                    "trade_count": trade_count,
                    "holdings": holdings,
                    "valuation_source": "worker-computed",
                    "updated_at": datetime.utcnow().isoformat(),
                }
                APP_STATE.update_model_stats(sig, stats_data)

            # 更新关注列表
            APP_STATE.monitored_symbols.update(all_held_symbols)
            APP_STATE.system_status["last_portfolio_update"] = datetime.utcnow().isoformat()

        except Exception as e:
            logger.error(f"❌ 持仓监控失败: {e}")

        await asyncio.sleep(1)  # 每秒检查一次


async def stock_data_loader_worker():
    """
    后台任务：懒加载/缓存大的 ai_stock_data.json 文件。
    仅在文件变化时重新加载。
    """
    logger.info("🚀 启动历史数据加载任务...")
    
    while True:
        try:
            if DATA_FILE.exists():
                mtime = DATA_FILE.stat().st_mtime
                if mtime != APP_STATE.stock_history_mtime:
                    logger.info("📂 重载 ai_stock_data.json ...")
                    start = time.time()
                    with open(DATA_FILE, "r", encoding="utf-8") as f:
                        APP_STATE.stock_history_cache = json.load(f)
                    APP_STATE.stock_history_mtime = mtime
                    elapsed = time.time() - start
                    logger.info(f"✅ 历史数据加载完成，耗时 {elapsed:.2f}s，共 {len(APP_STATE.stock_history_cache)} 只股票")
        except Exception as e:
            logger.error(f"❌ 历史数据加载失败: {e}")

        await asyncio.sleep(10)  # 每 10 秒检查一次


# =============================================================================
# FastAPI 生命周期
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动后台任务"""
    logger.info("=" * 60)
    logger.info("🎯 高性能交易 API 启动中...")
    logger.info("=" * 60)
    
    # 预加载历史数据
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                APP_STATE.stock_history_cache = json.load(f)
            APP_STATE.stock_history_mtime = DATA_FILE.stat().st_mtime
            logger.info(f"✅ 预加载完成: {len(APP_STATE.stock_history_cache)} 只股票")
        except Exception as e:
            logger.warning(f"⚠️ 预加载失败: {e}")

    # 加载系统配置
    try:
        config = _load_config_dict_safe(DEFAULT_CONFIG)
        if config:
            APP_STATE.system_config = config
            enabled_count = sum(1 for m in config.get("models", []) if m.get("enabled", True))
            logger.info(f"✅ 系统配置加载完成（{enabled_count}/{len(config.get('models', []))} 个模型启用）")
        else:
            logger.warning("⚠️ 未能加载系统配置")
    except Exception as e:
        logger.warning(f"⚠️ 系统配置加载失败: {e}")

    # 启动后台任务
    tasks = [
        asyncio.create_task(market_data_worker()),
        asyncio.create_task(portfolio_watcher_worker()),
        asyncio.create_task(stock_data_loader_worker()),
    ]
    
    APP_STATE.system_status["initialized"] = True
    logger.info("✅ 后台任务已启动")
    
    yield
    
    # 清理
    for task in tasks:
        task.cancel()
    logger.info("👋 API 服务已关闭")


app = FastAPI(
    title="高性能交易 API",
    description="采用后台同步架构，API 响应延迟 < 1ms",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 极速 API 接口 (Zero-IO in handler) - 核心优化
# =============================================================================
@app.get("/")
async def root_status():
    """返回系统状态，包含当前模拟时间"""
    sim_datetime = APP_STATE.system_status.get("sim_datetime")
    sim_date = APP_STATE.system_status.get("sim_date")
    
    return {
        "status": "ok",
        "mode": "high-performance",
        "data_mode": APP_STATE.system_status.get("data_mode", "local-only"),
        "message": "AStockArena 高性能后端运行中 (纯本地数据模式)",
        
        # 模拟时间信息（关键！）
        "simulation": {
            "current_datetime": sim_datetime,
            "current_date": sim_date,
            "sync_mode": APP_STATE.system_status.get("sync_mode", "initializing"),
            "note": "API 返回的价格与 Agent 决策时看到的价格一致" if sim_datetime or sim_date else "等待交易数据...",
        },
        
        # 缓存统计
        "cache_stats": {
            "monitored_symbols": len(APP_STATE.monitored_symbols),
            "cached_quotes": len(APP_STATE.market_quotes),
            "cached_portfolios": len(APP_STATE.portfolios),
            "stock_history_count": len(APP_STATE.stock_history_cache),
        },
        
        "system_status": APP_STATE.system_status,
    }


@app.get("/api/market/quotes")
async def market_quotes(codes: str = Query(..., description="Comma separated codes")):
    """
    极速行情接口：直接返回内存缓存，延迟 < 1ms。
    包含 AI 持仓统计信息。
    """
    code_list = [_normalize_code(c) for c in codes.split(",") if c.strip()]
    if not code_list:
        raise HTTPException(status_code=400, detail="codes query param is required")

    # 将请求的股票加入关注列表
    APP_STATE.monitored_symbols.update(code_list)

    # 预计算每只股票的 AI 持仓统计
    ai_stats: Dict[str, Dict[str, Any]] = {}
    for code in code_list:
        ai_stats[code] = {"holding_count": 0, "trade_volume": 0, "attention_score": 0}

    # 统计 AI 持仓数量（当前持仓）
    for sig, portfolio in APP_STATE.portfolios.items():
        for holding in portfolio.get("holdings", []):
            symbol = _normalize_code(holding.get("symbol", ""))
            if symbol in ai_stats:
                ai_stats[symbol]["holding_count"] += 1

    # 计算 AI 关注度（30天内，每有一天有一个 AI 持有该股就 +1 分）
    # 以及 AI 交易量
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    for sig, records in APP_STATE.position_records.items():
        # 用于追踪该模型在每天是否持有某只股票
        daily_holdings: Dict[str, Set[str]] = {}  # date -> set of symbols held
        
        for rec in records:
            rec_date = rec.get("date")
            if not rec_date or rec_date < cutoff_date:
                continue
            
            # 检查该记录中持有哪些股票
            positions = rec.get("positions", {}) or {}
            for code_key, detail in positions.items():
                if code_key == "CASH":
                    continue
                if isinstance(detail, dict) and (detail.get("shares") or 0) > 0:
                    norm_code = _normalize_code(code_key)
                    if norm_code in ai_stats:
                        if rec_date not in daily_holdings:
                            daily_holdings[rec_date] = set()
                        daily_holdings[rec_date].add(norm_code)
            
            # 统计交易量
            action = rec.get("this_action") or {}
            symbol = _normalize_code(action.get("symbol", ""))
            if symbol in ai_stats and action.get("action") in ("buy", "sell"):
                ai_stats[symbol]["trade_volume"] += abs(action.get("amount") or 0)
        
        # 累加关注度：每天每个模型持有该股就 +1
        for date, symbols in daily_holdings.items():
            for symbol in symbols:
                if symbol in ai_stats:
                    ai_stats[symbol]["attention_score"] += 1

    results = []
    source = "worker-cache"

    for code in code_list:
        # 直接从内存获取
        quote = APP_STATE.get_quote(code)

        if not quote:
            # 回退到历史缓存
            hist = APP_STATE.stock_history_cache.get(code)
            fallback_price = 0
            if hist:
                hourly = hist.get("小时线行情") or []
                if hourly:
                    fallback_price = float(hourly[-1].get("close") or 0)
                elif hist.get("日线行情"):
                    daily = hist["日线行情"]
                    if daily:
                        fallback_price = float(daily[-1].get("close") or 0)

            quote = {
                "code": code,
                "price": round(fallback_price, 4),
                "changePercent": 0,
                "volume": 0,
                "turnover": 0,
                "ts": datetime.utcnow().isoformat() + "Z",
                "source": "history-fallback",
            }
            source = "mixed"
        else:
            quote = dict(quote)

        # 添加 AI 统计信息
        stats = ai_stats.get(code, {})
        quote["aiHoldingCount"] = stats.get("holding_count", 0)      # 当前持仓的 AI 数量
        quote["aiTradeVolume"] = stats.get("trade_volume", 0)        # 30天内 AI 总交易量
        quote["aiAttentionScore"] = stats.get("attention_score", 0)  # 30天内 AI 关注度（每天每模型持有+1）

        results.append(quote)

    return {"quotes": results, "source": source}


@app.get("/api/system/config")
async def get_system_config():
    """
    返回系统配置信息（交易规则、模型配置、数据源状态等）
    """
    config = APP_STATE.system_config.copy() if APP_STATE.system_config else {}
    
    # 统计启用的模型数量
    enabled_models = [
        m for m in config.get("models", [])
        if m.get("enabled", True)
    ]
    model_count = len(enabled_models)
    
    # 获取数据源状态（从 system_status 获取）
    data_mode = APP_STATE.system_status.get("data_mode", "local-only")
    # 根据数据模式判断连接状态
    if data_mode == "local-only":
        data_source_status = "connected"  # 本地数据模式总是连接状态
    elif data_mode in ["hybrid", "tinystock"]:
        # 检查是否有行情数据更新
        last_update = APP_STATE.system_status.get("last_market_update")
        data_source_status = "connected" if last_update else "disconnected"
    else:
        data_source_status = "unknown"
    
    # 格式化配置信息
    trading_rules = config.get("trading_rules", {})
    risk_management = config.get("risk_management", {})
    agent_config = config.get("agent_config", {})
    data_config = config.get("data_config", {})
    
    return {
        "trading_rules": {
            "commission_rate": trading_rules.get("commission_rate", 0.0003),
            "commission_rate_percent": round(trading_rules.get("commission_rate", 0.0003) * 100, 4),
            "min_commission": trading_rules.get("min_commission", 5.0),
            "stamp_duty_rate": trading_rules.get("stamp_duty_rate", 0.0005),
            "stamp_duty_rate_percent": round(trading_rules.get("stamp_duty_rate", 0.0005) * 100, 4),
            "t_plus_one_enabled": True,  # T+1 是硬编码的交易规则
        },
        "risk_management": {
            "single_stock_max_position": risk_management.get("single_stock_max_position", 0.50),
            "single_stock_max_position_percent": round(risk_management.get("single_stock_max_position", 0.50) * 100, 2),
        },
        "agent_config": {
            "initial_cash": agent_config.get("initial_cash", 1000000.0),
            "max_steps": agent_config.get("max_steps", 30),
            "max_retries": agent_config.get("max_retries", 3),
            "decision_frequency": "hourly",  # 决策频率是每小时
            "auto_trading_enabled": True,  # 自动交易默认开启
        },
        "models": {
            "total_count": len(config.get("models", [])),
            "enabled_count": model_count,
            "enabled_models": [
                {
                    "name": m.get("name"),
                    "signature": m.get("signature"),
                    "basemodel": m.get("basemodel"),
                }
                for m in enabled_models
            ],
        },
        "data_source": {
            "status": data_source_status,
            "mode": data_mode,
            "update_frequency": "realtime" if data_mode != "unknown" else "unknown",
        },
        "data_config": {
            "stock_json_path": data_config.get("stock_json_path", "./data_flow/ai_stock_data.json"),
            "news_csv_path": data_config.get("news_csv_path", "./data_flow/news.csv"),
            "history_days": 30,  # 历史数据天数（可从配置或实际数据计算）
        },
    }


@app.get("/api/live/model-stats")
async def live_model_stats(signature: str | None = None):
    """
    极速资产统计接口：直接返回 Worker 计算好的结果，延迟 < 1ms。
    """
    sig = signature or _load_runtime_signature()
    
    stats = APP_STATE.get_model_stats(sig)
    if stats:
        return stats

    # 如果还没准备好，返回初始化状态
    return {
        "signature": sig,
        "equity": 0,
        "cash": 0,
        "total_return_pct": 0,
        "sharpe_ratio": 0,
        "max_drawdown_pct": 0,
        "position_count": 0,
        "trade_count": 0,
        "holdings": [],
        "status": "initializing",
        "note": "后台 worker 正在初始化数据，请稍候...",
    }


@app.get("/api/live/current-positions")
async def live_current_positions(signature: str | None = None):
    """
    极速持仓接口：直接从内存读取。
    """
    sig = signature or _load_runtime_signature()
    
    portfolio = APP_STATE.get_portfolio(sig)
    if portfolio:
        return {
            "positions": portfolio.get("holdings", []),
            "cash": portfolio.get("cash", 0),
            "total_equity": portfolio.get("total_equity", 0),
            "date": portfolio.get("date"),
            "valuation_source": "worker-computed",
        }

    return {
        "positions": [],
        "cash": 0,
        "total_equity": 0,
        "date": None,
        "status": "initializing",
    }


@app.get("/api/live/position-lines")
async def live_position_lines(
    limit: int = Query(100, ge=1, le=2000),
    signature: str | None = None
):
    """返回最近 N 条持仓记录（用于图表）"""
    sig = signature or _load_runtime_signature()
    
    records = APP_STATE.position_records.get(sig, [])
    if not records:
        # 尝试直接读取
        try:
            pos_file = _position_file_for_signature(sig)
            records = _read_jsonl_tail(pos_file, limit)
        except Exception:
            pass

    out = []
    for it in records[-limit:]:
        positions = it.get("positions", {}) or {}
        cash = positions.get("CASH")
        cnt = sum(
            1 for k, v in positions.items()
            if k != "CASH" and isinstance(v, dict) and v.get("shares", 0) > 0
        )
        out.append({
            "date": it.get("date"),
            "id": it.get("id"),
            "cash": cash,
            "positions_count": cnt,
            "action": (it.get("this_action") or {}).get("action"),
            "symbol": (it.get("this_action") or {}).get("symbol"),
            "amount": (it.get("this_action") or {}).get("amount"),
        })
    return {"items": out}


@app.get("/api/live/latest-position")
async def live_latest_position(signature: str | None = None):
    """返回最新持仓记录"""
    sig = signature or _load_runtime_signature()
    
    records = APP_STATE.position_records.get(sig, [])
    if records:
        return {"item": records[-1]}

    # 回退到文件读取
    try:
        pos_file = _position_file_for_signature(sig)
        items = _read_jsonl_tail(pos_file, limit=1)
        if items:
            return {"item": items[-1]}
    except Exception:
        pass

    return {"item": None}


@app.get("/api/live/pnl-series")
async def live_pnl_series(
    signature: str | None = None,
    days: int = Query(30, ge=1, le=365),
    valuation: str = Query("equity", description="cash or equity")
):
    """返回每日 PnL 序列"""
    sig = signature or _load_runtime_signature()
    initial_cash = _load_initial_cash()
    
    records = APP_STATE.position_records.get(sig, [])
    if not records:
        try:
            pos_file = _position_file_for_signature(sig)
            records = _read_jsonl_tail(pos_file, limit=10000)
        except Exception:
            return {"items": [], "valuation_used": valuation}

    # 按日期分组
    by_date: Dict[str, Dict] = {}
    for it in records:
        d = it.get("date")
        if not d:
            continue
        prev = by_date.get(d)
        if prev is None or (it.get("id", -1) > prev.get("id", -1)):
            by_date[d] = it

    dates_sorted = sorted(by_date.keys())[-days:]
    out = []

    for d in dates_sorted:
        rec = by_date[d]
        positions = rec.get("positions", {}) or {}
        cash = float(positions.get("CASH", 0) or 0)

        if valuation.lower() != "equity":
            ret_pct = (cash / initial_cash - 1.0) * 100.0
            out.append({"date": d, "returnPct": round(ret_pct, 2), "cash": round(cash, 2)})
        else:
            equity = cash
            for code, det in positions.items():
                if code == "CASH":
                    continue
                if isinstance(det, dict):
                    shares = det.get("shares", 0) or 0
                    if shares > 0:
                        norm = _normalize_code(code)
                        q = APP_STATE.get_quote(norm) or {}
                        price = q.get("price") or det.get("avg_price", 0)
                        equity += shares * float(price)

            ret_pct = (equity / initial_cash - 1.0) * 100.0
            out.append({"date": d, "returnPct": round(ret_pct, 2), "equity": round(equity, 2)})

    return {"items": out, "valuation_used": valuation}


@app.get("/api/live/recent-decisions")
async def live_recent_decisions(
    signature: str | None = None,
    limit: int = Query(20, ge=1, le=100)
):
    """获取最近交易决策"""
    sig = signature or _load_runtime_signature()
    
    records = APP_STATE.position_records.get(sig, [])
    if not records:
        try:
            pos_file = _position_file_for_signature(sig)
            records = _read_jsonl_tail(pos_file, limit * 2)
        except Exception:
            return {"decisions": []}

    decisions = []
    for it in reversed(records):
        action = it.get("this_action") or {}
        positions = it.get("positions", {}) or {}
        
        holdings = sum(
            1 for k, v in positions.items()
            if k != "CASH" and isinstance(v, dict) and v.get("shares", 0) > 0
        )

        decisions.append({
            "date": it.get("date"),
            "time": it.get("decision_time"),
            "count": it.get("decision_count"),
            "action": action.get("action"),
            "symbol": action.get("symbol"),
            "amount": action.get("amount"),
            "cash": float(positions.get("CASH", 0) or 0),
            "holdings": holdings,
            "id": it.get("id"),
        })

        if len(decisions) >= limit:
            break

    return {"decisions": decisions}


@app.get("/api/live/stock-detail")
async def live_stock_detail(
    symbol: str = Query(..., description="股票代码"),
    history_limit: int = Query(60, ge=10, le=200),
    news_limit: int = Query(6, ge=1, le=20),
):
    """股票详情接口（读取缓存）"""
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不能为空")

    normalized = _normalize_code(symbol)
    
    # 从缓存获取
    stock_payload = APP_STATE.stock_history_cache.get(normalized)
    if not stock_payload:
        for cand in _symbol_candidates(symbol):
            stock_payload = APP_STATE.stock_history_cache.get(cand)
            if stock_payload:
                break

    if not stock_payload:
        raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的数据缓存")

    # 构造响应
    hourly_raw = stock_payload.get("小时线行情", []) or []
    hourly_window = hourly_raw[-history_limit:]
    hourly_series = [
        {
            "time": item.get("date"),
            "price": item.get("close"),
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "volume": item.get("vol"),
            "amount": item.get("amount"),
            "bid": item.get("buy1"),
        }
        for item in hourly_window
        if item.get("date")
    ]

    # 技术指标数据
    hourly_indicators_raw = stock_payload.get("小时线指标", []) or []
    hourly_indicators_window = hourly_indicators_raw[-history_limit:]
    hourly_indicators = [
        {
            "time": item.get("Date") or item.get("date"),
            "close": item.get("CLOSE") or item.get("close"),
            "K": item.get("K"),
            "D": item.get("D"),
            "J": item.get("J"),
            "BOLL": item.get("BOLL"),
            "BOLL_upper": item.get("UPR"),
            "BOLL_lower": item.get("DWN"),
        }
        for item in hourly_indicators_window
        if item.get("Date") or item.get("date")
    ]

    # 日线行情
    daily_raw = stock_payload.get("日线行情", []) or []
    daily_window = daily_raw[-history_limit:]
    daily_series = [
        {
            "time": item.get("date"),
            "price": item.get("close"),
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "volume": item.get("vol"),
            "amount": item.get("amount"),
        }
        for item in daily_window
        if item.get("date")
    ]

    # 日线指标数据
    daily_indicators_raw = stock_payload.get("日线指标", []) or []
    daily_indicators_window = daily_indicators_raw[-history_limit:]
    daily_indicators = [
        {
            "time": item.get("Date") or item.get("date"),
            "close": item.get("CLOSE") or item.get("close"),
            "K": item.get("K"),
            "D": item.get("D"),
            "J": item.get("J"),
            "BOLL": item.get("BOLL"),
            "BOLL_upper": item.get("UPR"),
            "BOLL_lower": item.get("DWN"),
        }
        for item in daily_indicators_window
        if item.get("Date") or item.get("date")
    ]

    # 最新行情
    quote = APP_STATE.get_quote(normalized) or {}

    summary = {
        "symbol": normalized,
        "name": stock_payload.get("名称") or stock_payload.get("name"),
        "latest_time": hourly_window[-1].get("date") if hourly_window else None,
        "latest_price": quote.get("price") or (hourly_window[-1].get("close") if hourly_window else None),
        "change_percent": stock_payload.get("涨跌幅"),
        "turnover_rate": stock_payload.get("换手率"),
        "volume": stock_payload.get("成交量"),
    }

    # AI 持仓统计
    ai_positions = []
    ai_trades = []
    for sig, portfolio in APP_STATE.portfolios.items():
        for holding in portfolio.get("holdings", []):
            if _normalize_code(holding.get("symbol", "")) == normalized:
                ai_positions.append({
                    "signature": sig,
                    **holding,
                })
                break

    # 从记录中提取交易
    for sig, records in APP_STATE.position_records.items():
        for rec in reversed(records[-50:]):
            action = rec.get("this_action") or {}
            if _normalize_code(action.get("symbol", "")) == normalized:
                ai_trades.append({
                    "signature": sig,
                    "date": rec.get("date"),
                    "decision_time": rec.get("decision_time"),
                    "action": action.get("action"),
                    "amount": action.get("amount"),
                })
                if len(ai_trades) >= 20:
                    break

    return {
        "summary": summary,
        "hourly_prices": hourly_series,
        "hourly_indicators": hourly_indicators,
        "daily_prices": daily_series,
        "daily_indicators": daily_indicators,
        "ai_positions": ai_positions,
        "ai_trades": ai_trades[:20],
        "ai_summary": {
            "holding_count": len(ai_positions),
            "trade_volume": sum(abs(t.get("amount") or 0) for t in ai_trades),
            "holding_models": [p.get("signature") for p in ai_positions],
        },
    }


@app.get("/api/live/news")
async def live_latest_news(
    limit: int = Query(10, ge=1, le=50),
    symbols: str | None = None
):
    """获取最新新闻"""
    import pandas as pd

    news_path = DATA_DIR / "news.csv"
    if not news_path.exists():
        return {"news": [], "note": "No news data available"}

    try:
        df = None
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']:
            try:
                df = pd.read_csv(news_path, encoding=encoding)
                break
            except Exception:
                continue

        if df is None or df.empty:
            return {"news": [], "note": "Failed to read news data"}

        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if 'symbol' in df.columns:
                df = df[df['symbol'].str.upper().isin(symbol_list)]

        time_col = 'publish_time' if 'publish_time' in df.columns else 'search_time'
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            df = df.sort_values(by=time_col, ascending=False)

        df = df.head(limit)

        news_items = []
        for _, row in df.iterrows():
            news_items.append({
                "title": str(row.get('title', '')),
                "content": str(row.get('content', ''))[:200],
                "publish_time": str(row.get('publish_time', '')),
                "symbol": str(row.get('symbol', '')),
                "source": str(row.get('source', 'Unknown')),
                "url": str(row.get('url', '')),
            })

        return {"news": news_items}

    except Exception as e:
        logger.warning(f"⚠️ News endpoint error: {e}")
        return {"news": [], "error": "新闻源暂时不可用"}


# =============================================================================
# LLM 会话管理 (保留原有功能)
# =============================================================================
class LLMChatRequest(BaseModel):
    signature: str
    prompt: str
    config_path: Optional[str] = None
    reset: bool = False
    system_prompt: Optional[str] = None


class LLMSession:
    """LLM 会话管理器"""

    def __init__(self, *, agent: BaseAgent, system_prompt: str, config_path: Path):
        self.agent = agent
        self.system_prompt = system_prompt
        self.config_path = config_path
        self.history: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        self.lock = asyncio.Lock()
        self.created_at = datetime.utcnow()

    def reset_history(self, *, system_prompt: Optional[str] = None) -> None:
        if system_prompt:
            self.system_prompt = system_prompt
        self.history = [SystemMessage(content=self.system_prompt)]


LLM_SESSIONS: Dict[str, LLMSession] = {}
_LLM_SESSION_LOCK: Optional[asyncio.Lock] = None


def _get_llm_session_lock() -> asyncio.Lock:
    global _LLM_SESSION_LOCK
    if _LLM_SESSION_LOCK is None:
        _LLM_SESSION_LOCK = asyncio.Lock()
    return _LLM_SESSION_LOCK


def _session_cache_key(signature: str, config_path: Path) -> str:
    return f"{signature}::{str(config_path.resolve())}"


def _resolve_config_path(config_path: Optional[str]) -> Path:
    if not config_path:
        return DEFAULT_CONFIG
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path(__file__).parent / candidate).resolve()
    return candidate


def _select_model_config(config: Dict[str, Any], signature: str) -> Dict[str, Any]:
    for item in config.get("models", []):
        if item.get("signature") == signature:
            if not item.get("enabled", True):
                raise HTTPException(status_code=400, detail=f"模型 {signature} 已被禁用")
            return item
    raise HTTPException(status_code=404, detail=f"未找到 signature={signature} 的模型配置")


def _default_llm_system_prompt(signature: str, basemodel: Optional[str]) -> str:
    suffix = f"（底座：{basemodel}）" if basemodel else ""
    return (
        f"你是交易代理 {signature}{suffix} 的对话接口，只能使用中文回答，"
        "回答时保持简洁，避免执行真实交易，仅做策略分析或解释。"
    )


async def _build_agent_for_signature(signature: str, config_path: Path) -> BaseAgent:
    config = _load_config_dict(config_path)
    agent_config = config.get("agent_config", {})
    data_config = config.get("data_config", {})
    log_config = config.get("log_config", {})
    trading_rules = config.get("trading_rules", {})
    risk_management = config.get("risk_management", {})
    date_range = config.get("date_range", {}) or {}
    init_date = date_range.get("init_date") or datetime.utcnow().strftime("%Y-%m-%d")

    model_cfg = _select_model_config(config, signature)
    basemodel = model_cfg.get("basemodel")
    if not basemodel:
        raise HTTPException(status_code=400, detail=f"模型 {signature} 缺少 basemodel 字段")

    stock_symbols = model_cfg.get("stock_symbols") or BaseAgent.DEFAULT_STOCK_SYMBOLS
    stock_json_path = data_config.get("stock_json_path", "./data_flow/ai_stock_data.json")
    news_csv_path = data_config.get("news_csv_path", "./data_flow/news.csv")
    macro_csv_path = data_config.get("macro_csv_path")
    log_path = log_config.get("log_path", "./data_flow/agent_data")

    agent = BaseAgent(
        signature=signature,
        basemodel=basemodel,
        stock_symbols=stock_symbols,
        stock_json_path=stock_json_path,
        news_csv_path=news_csv_path,
        macro_csv_path=macro_csv_path,
        log_path=log_path,
        max_steps=agent_config.get("max_steps", 10),
        max_retries=agent_config.get("max_retries", 3),
        base_delay=agent_config.get("base_delay", 0.5),
        openai_base_url=model_cfg.get("openai_base_url"),
        openai_api_key=model_cfg.get("openai_api_key"),
        google_api_key=model_cfg.get("google_api_key"),
        safety_settings=model_cfg.get("safety_settings"),
        initial_cash=agent_config.get("initial_cash", 1_000_000.0),
        init_date=init_date,
        trading_rules=trading_rules,
        risk_management=risk_management,
        force_replay=bool(model_cfg.get("force_replay", False)),
    )
    await agent.initialize()
    return agent


async def _get_or_create_llm_session(
    signature: str,
    config_path: Path,
    system_prompt_override: Optional[str] = None,
) -> LLMSession:
    cache_key = _session_cache_key(signature, config_path)
    session = LLM_SESSIONS.get(cache_key)
    if session:
        if system_prompt_override and system_prompt_override != session.system_prompt:
            session.reset_history(system_prompt=system_prompt_override)
        return session

    lock = _get_llm_session_lock()
    async with lock:
        session = LLM_SESSIONS.get(cache_key)
        if session:
            if system_prompt_override and system_prompt_override != session.system_prompt:
                session.reset_history(system_prompt=system_prompt_override)
            return session

        agent = await _build_agent_for_signature(signature, config_path)
        prompt = system_prompt_override or _default_llm_system_prompt(signature, agent.basemodel)
        session = LLMSession(agent=agent, system_prompt=prompt, config_path=config_path)
        LLM_SESSIONS[cache_key] = session
        return session


async def _invoke_llm_session(session: LLMSession, prompt: str) -> Dict[str, Any]:
    if not session.agent or not session.agent.model:
        raise HTTPException(status_code=500, detail="LLM 模型尚未初始化完成")

    user_msg = HumanMessage(content=prompt)
    history = session.history + [user_msg]
    try:
        response = await session.agent.model.ainvoke(history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {e}")

    session.history.extend([user_msg, response])
    content = getattr(response, "content", None)
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "\n".join(filter(None, text_parts)) or str(content)
    elif content is None:
        content = str(response)

    return {
        "signature": session.agent.signature,
        "model": session.agent.basemodel,
        "response": content,
        "history_length": len(session.history),
        "usage": getattr(response, "usage_metadata", None),
        "created_at": session.created_at.isoformat() + "Z",
    }


@app.get("/api/llm/ping")
async def llm_ping(config_path: Optional[str] = None):
    """LLM 健康检查"""
    resolved_path = _resolve_config_path(config_path)
    config = _load_config_dict(resolved_path)
    available = [
        m.get("signature")
        for m in config.get("models", [])
        if m.get("enabled", True) and m.get("signature")
    ]
    return {
        "status": "ok",
        "session_count": len(LLM_SESSIONS),
        "available_signatures": available,
        "config_path": str(resolved_path),
    }


@app.post("/api/llm/ask")
async def llm_ask(payload: LLMChatRequest):
    """LLM 对话接口"""
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    resolved_path = _resolve_config_path(payload.config_path)
    session = await _get_or_create_llm_session(
        payload.signature,
        resolved_path,
        system_prompt_override=payload.system_prompt,
    )

    if payload.reset:
        session.reset_history(system_prompt=payload.system_prompt)

    async with session.lock:
        return await _invoke_llm_session(session, prompt)


# =============================================================================
# 任务管理接口 (保留原有功能)
# =============================================================================
from utils.backup_utils import run_backup_snapshot


@app.post("/api/run-trading")
async def run_trading(config_path: str | None = None):
    """启动交易脚本"""
    job_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat() + "Z"
    log_file = LOG_DIR / f"{job_id}.log"

    if not _truthy_env("SKIP_API_BACKUP"):
        ok = run_backup_snapshot(reason="api_run_trading")
        if not ok:
            logger.warning("⚠️ Pre-run backup failed")

    cmd = [sys.executable, str(Path(__file__).parent / "main.py")]
    if config_path:
        cmd.append(config_path)

    try:
        lf = open(log_file, "wb")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=lf,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start process: {e}")

    APP_STATE.jobs[job_id] = {
        "id": job_id,
        "pid": proc.pid,
        "started_at": started_at,
        "status": "running",
        "returncode": None,
        "log_file": str(log_file),
        "process": proc,
    }

    return {"job_id": job_id, "pid": proc.pid, "started_at": started_at}


@app.post("/api/backup")
async def trigger_backup(retain: int = Query(5, ge=1, le=50)):
    ok = run_backup_snapshot(reason="api_manual", retain=retain)
    if not ok:
        raise HTTPException(status_code=500, detail="无法完成备份")
    return {"status": "ok", "retain": retain}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    job = APP_STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    proc: subprocess.Popen = job.get("process")
    if proc is not None:
        rc = proc.poll()
        if rc is None:
            status = "running"
        else:
            status = "finished" if rc == 0 else "failed"
            job["returncode"] = rc
            job["status"] = status
            job.pop("process", None)
    else:
        status = job.get("status", "unknown")

    job["status"] = status

    log_text = None
    try:
        lfpath = Path(job["log_file"])
        if lfpath.exists():
            with open(lfpath, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, 2)
                size = f.tell()
                start = max(0, size - 2000)
                f.seek(start)
                log_text = f.read()
    except Exception:
        log_text = None

    return {
        "id": job_id,
        "pid": job.get("pid"),
        "status": job.get("status"),
        "started_at": job.get("started_at"),
        "returncode": job.get("returncode"),
        "log_tail": log_text,
    }


@app.get("/api/jobs")
async def list_jobs():
    items = []
    for j in APP_STATE.jobs.values():
        items.append({
            "id": j["id"],
            "pid": j["pid"],
            "status": j["status"],
            "started_at": j["started_at"],
            "log_file": j["log_file"],
        })
    return {"jobs": items}


@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str):
    job = APP_STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    proc: subprocess.Popen = job.get("process")
    if not proc:
        raise HTTPException(status_code=400, detail="Process already finished")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    job["status"] = "terminated"
    job.pop("process", None)
    return {"id": job_id, "status": job["status"]}


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
