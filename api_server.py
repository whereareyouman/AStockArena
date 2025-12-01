from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel
import json
import asyncio
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from agent.base_agent.base_agent import BaseAgent

app = FastAPI()

# Allow CORS so the Vite dev server (localhost:5173) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root_status():
    """Return a friendly status message so hitting '/' doesn't 404."""
    return {
        "status": "ok",
        "message": "AI-Trader backend is running. All REST APIs live under /api/* (e.g. /api/jobs, /api/llm/ping).",
    }


DATA_FILE = Path(__file__).parent / "data" / "ai_stock_data.json"
_AI_STOCK_CACHE: Dict[str, Any] = {}
_AI_STOCK_CACHE_MTIME: Optional[float] = None

import subprocess
import uuid
import sys
from datetime import datetime, timedelta
import os

from tools.backup_utils import run_backup_snapshot

# Simple job manager in-memory (processes are not persisted across server restarts)
JOBS: Dict[str, Dict[str, Any]] = {}
LOG_DIR = Path(__file__).parent / "logs" / "jobs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_ENV = Path(__file__).parent / "runtime_env.json"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default_config.json"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "false").lower() in ("1", "true", "yes")


def _sanitize_provider_detail(detail: str) -> str:
    lowered = detail.lower()
    if "429" in detail or "too many request" in lowered or "rate limit" in lowered:
        return "上游数据源暂时繁忙，请稍后再试。"
    if "login failed" in lowered or "missing" in lowered:
        return "无法连接至行情数据提供方，请稍后重试。"
    return detail


def _raise_provider_error(status_code: int, detail: str):
    print(f"⚠️ Provider error ({status_code}): {detail}")
    raise HTTPException(status_code=status_code, detail=_sanitize_provider_detail(detail))


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
    return Path(__file__).parent / "data" / "agent_data" / sig / "position" / "position.jsonl"

def _read_jsonl_tail(path: Path, limit: int = 100) -> list[dict]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Position file not found: {path}")
    lines: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except Exception:
                    continue
        return lines[-limit:]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read positions: {e}")

def _load_initial_cash() -> float:
    """Load initial_cash from configs/default_config.json (fallback to 1000000)."""
    try:
        if DEFAULT_CONFIG.exists():
            with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            agent_cfg = (cfg or {}).get("agent_config", {})
            return float(agent_cfg.get("initial_cash", 1000000.0))
    except Exception:
        pass
    return 1000000.0


def _load_stock_data_cache() -> Dict[str, Any]:
    """Read ai_stock_data.json once and reuse until文件修改。"""
    global _AI_STOCK_CACHE, _AI_STOCK_CACHE_MTIME
    try:
        if not DATA_FILE.exists():
            return {}
        mtime = DATA_FILE.stat().st_mtime
        if _AI_STOCK_CACHE_MTIME != mtime:
            with DATA_FILE.open("r", encoding="utf-8") as f:
                _AI_STOCK_CACHE = json.load(f)
            _AI_STOCK_CACHE_MTIME = mtime
    except Exception:
        _AI_STOCK_CACHE = {}
        _AI_STOCK_CACHE_MTIME = None
    return _AI_STOCK_CACHE


def _symbol_candidates(symbol: str) -> List[str]:
    norm = (symbol or "").upper()
    plain = norm
    if norm.startswith(("SH", "SZ")) and len(norm) > 2:
        plain = norm[2:]
    candidates = {norm, plain, f"SH{plain}", f"SZ{plain}"}
    return [c for c in candidates if c]


def _safe_parse_dt(value: Optional[str], fmt: str):
    if not value:
        return None
    try:
        return datetime.strptime(value, fmt)
    except Exception:
        return None


def _pick_price(entries: List[Dict[str, Any]], target_str: Optional[str], fmt: str):
    if not entries:
        return None

    target_dt = _safe_parse_dt(target_str, fmt) if target_str else None
    if target_dt:
        for entry in reversed(entries):
            entry_dt = _safe_parse_dt(entry.get("date"), fmt)
            if entry_dt and entry_dt == target_dt:
                return entry.get("close") or entry.get("buy1")
        for entry in reversed(entries):
            entry_dt = _safe_parse_dt(entry.get("date"), fmt)
            if entry_dt and entry_dt <= target_dt:
                return entry.get("close") or entry.get("buy1")

    last = entries[-1]
    return last.get("close") or last.get("buy1")


def _lookup_cached_price(symbol: str, decision_time: Optional[str], date_only: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    data = _load_stock_data_cache()
    payload = None
    for cand in _symbol_candidates(symbol):
        payload = data.get(cand)
        if payload:
            break
    if not payload:
        return None, None

    hourly_entries = payload.get("小时线行情") or []
    price = _pick_price(hourly_entries, decision_time, "%Y-%m-%d %H:%M:%S")
    if price is not None:
        try:
            return float(price), "cache-hourly"
        except Exception:
            return None, None

    daily_entries = payload.get("日线行情") or []
    price = _pick_price(daily_entries, date_only or (decision_time[:10] if decision_time else None), "%Y-%m-%d")
    if price is not None:
        try:
            return float(price), "cache-daily"
        except Exception:
            return None, None

    return None, None


def _estimate_equity_for_positions(
    positions: Dict[str, Any],
    decision_time: Optional[str],
    date_only: Optional[str],
) -> tuple[float, float, List[Dict[str, Any]], str]:
    cash_val = float(positions.get("CASH", 0.0) or 0.0)
    total_equity = cash_val
    entries: List[Dict[str, Any]] = []
    sources: set[str] = set()

    for symbol, details in (positions or {}).items():
        if symbol == "CASH":
            continue
        if not isinstance(details, dict):
            continue
        shares = details.get("shares", 0) or 0
        if shares <= 0:
            continue
        entry_price = details.get("avg_price")
        if entry_price is not None:
            try:
                entry_price = float(entry_price)
            except (TypeError, ValueError):
                entry_price = None

        price, source = _lookup_cached_price(symbol, decision_time, date_only)
        if price is None:
            if entry_price is not None:
                price = entry_price
            else:
                price = 0.0
            source = "fallback"

        market_value = float(price) * shares
        total_equity += market_value
        sources.add(source or "fallback")

        entries.append({
            "symbol": symbol,
            "shares": shares,
            "entry_price": entry_price if entry_price is not None else price,
            "mark_price": round(float(price), 4),
            "market_value": round(market_value, 2),
            "valuation_source": source or "fallback",
        })

    if not sources:
        sources.add("cash-only")
    equity_source = "mixed" if len(sources) > 1 else next(iter(sources))
    return cash_val, total_equity, entries, equity_source


def _collect_ai_interest(target_symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Aggregate current AI holdings/trades per symbol."""
    normalized_targets = (
        { _normalize_code(sym) for sym in target_symbols }
        if target_symbols else None
    )
    signatures = _enabled_model_signatures()
    interest: Dict[str, Dict[str, Any]] = {}

    def _ensure_entry(symbol: str) -> Dict[str, Any]:
        return interest.setdefault(symbol, {
            "holding_count": 0,
            "total_shares": 0,
            "holding_models": set(),
            "trade_volume": 0.0,
        })

    for sig in signatures:
        try:
            records = _read_jsonl_tail(_position_file_for_signature(sig), limit=200)
        except HTTPException:
            continue
        if not records:
            continue
        latest = records[-1]
        latest_date = latest.get("date")
        positions = latest.get("positions", {}) or {}
        for symbol, details in positions.items():
            if symbol == "CASH":
                continue
            sym_norm = _normalize_code(symbol)
            if normalized_targets and sym_norm not in normalized_targets:
                continue
            shares = details.get("shares", 0) or 0
            if shares <= 0:
                continue
            entry = _ensure_entry(sym_norm)
            entry["holding_count"] += 1
            entry["total_shares"] += shares
            entry["holding_models"].add(sig)

        for rec in records:
            if latest_date and rec.get("date") != latest_date:
                continue
            action = rec.get("this_action") or {}
            sym = action.get("symbol")
            if not sym or action.get("action") not in {"buy", "sell"}:
                continue
            sym_norm = _normalize_code(sym)
            if normalized_targets and sym_norm not in normalized_targets:
                continue
            amount = abs(action.get("amount") or 0)
            if amount <= 0:
                continue
            entry = _ensure_entry(sym_norm)
            entry["trade_volume"] += amount

    total_models = max(1, len(signatures))
    result: Dict[str, Dict[str, Any]] = {}
    for symbol, data in interest.items():
        holding_models = sorted(data["holding_models"])
        total_shares = data["total_shares"]
        trade_volume = data["trade_volume"]
        turnover_percent = 0.0
        if total_shares > 0:
            turnover_percent = round(trade_volume / total_shares * 100.0, 2)
        attention_score = round(min(100.0, data["holding_count"] / total_models * 100.0), 2)
        result[symbol] = {
            "holding_count": data["holding_count"],
            "total_shares": total_shares,
            "trade_volume": trade_volume,
            "turnover_percent": turnover_percent,
            "attention_score": attention_score,
            "holding_models": holding_models,
        }
    return result


def _latest_position_record(signature: Optional[str]) -> tuple[dict, Optional[dict]]:
    pos_file = _position_file_for_signature(signature)
    items = _read_jsonl_tail(pos_file, limit=1)
    if not items:
        return {}, None
    latest = items[-1]
    positions = latest.get("positions", {}) or {}
    return positions, latest


def _build_holdings_with_prices(
    signature: Optional[str],
) -> tuple[float, float, List[Dict[str, Any]], dict, str, str]:
    """Return cash, total equity, enriched holdings list, raw positions, latest date, source."""
    positions, latest_record = _latest_position_record(signature)
    if not positions:
        return 0.0, 0.0, [], {}, "", "cash-only"

    latest_date = latest_record.get("date", "") if isinstance(latest_record, dict) else ""
    latest_time = latest_record.get("decision_time") if isinstance(latest_record, dict) else ""

    cash, total_equity, entries, equity_source = _estimate_equity_for_positions(
        positions, latest_time, latest_date
    )

    holdings: List[Dict[str, Any]] = []
    for entry in entries:
        entry_price = entry.get("entry_price") or 0.0
        mark_price = entry.get("mark_price", 0.0)
        shares = entry.get("shares", 0)
        market_value = entry.get("market_value", 0.0)
        cost_basis = entry_price * shares
        pnl = market_value - cost_basis
        pnl_percent = (pnl / cost_basis * 100) if cost_basis else 0.0

        holdings.append({
            "symbol": entry["symbol"],
            "shares": shares,
            "purchase_date": (positions.get(entry["symbol"], {}) or {}).get("purchase_date", ""),
            "entry_price": round(entry_price, 2),
            "current_price": round(mark_price, 2),
            "market_value": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_percent, 2),
            "valuation_source": entry.get("valuation_source", "fallback"),
        })

    return cash, total_equity, holdings, positions, latest_date, equity_source


class LLMChatRequest(BaseModel):
    signature: str
    prompt: str
    config_path: Optional[str] = None
    reset: bool = False
    system_prompt: Optional[str] = None


class LLMSession:
    """In-memory session wrapper that keeps model pointer and conversation history."""

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
    """Return the shared session map lock (lazy init to avoid loop issues)."""
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


def _load_config_dict(target_path: Path) -> Dict[str, Any]:
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {target_path}")
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"配置文件解析失败: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置文件读取失败: {e}")


def _enabled_model_signatures(config_path: Optional[Path] = None) -> List[str]:
    """Return enabled model signatures from config."""
    path = config_path or DEFAULT_CONFIG
    config = _load_config_dict(path)
    signatures: List[str] = []
    for item in config.get("models", []):
        sig = item.get("signature")
        if not sig:
            continue
        if item.get("enabled", True):
            signatures.append(sig)
    return signatures


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
    stock_json_path = data_config.get("stock_json_path", "./data/ai_stock_data.json")
    news_csv_path = data_config.get("news_csv_path", "./data/news.csv")
    macro_csv_path = data_config.get("macro_csv_path")
    log_path = log_config.get("log_path", "./data/agent_data")

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
        # LangChain 可能返回富内容，这里仅保留文本部分
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
    """Lightweight health-check for the new LLM endpoints."""
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
    """Expose BaseAgent 的 LLM 能力为可调用接口."""
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

## Removed test endpoints /api/hello and /api/positions


@app.get("/api/live/position-lines")
async def live_position_lines(limit: int = Query(100, ge=1, le=2000), signature: str | None = None):
    """Return the last N position records for visualization.

    Each item includes {date, id, cash, positions_count} for lightweight charting.
    """
    pos_file = _position_file_for_signature(signature)
    items = _read_jsonl_tail(pos_file, limit)
    out = []
    for it in items:
        positions = it.get("positions", {}) or {}
        cash = positions.get("CASH")
        # count non-zero holdings excluding CASH
        cnt = sum(1 for k, v in positions.items() if k != "CASH" and isinstance(v, (int, float)) and v)
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
    """Return the most recent full position record."""
    pos_file = _position_file_for_signature(signature)
    items = _read_jsonl_tail(pos_file, limit=1)
    if not items:
        return {"item": None}
    return {"item": items[-1]}


@app.get("/api/live/pnl-series")
async def live_pnl_series(
    signature: str | None = None,
    days: int = Query(30, ge=1, le=365),
    valuation: str = Query("cash", description="cash or equity")
):
    """Return daily PnL/return series from position.jsonl.

    - valuation=cash: returnPct = (cash/initial_cash - 1) * 100
    - valuation=equity: returnPct = ((cash + Σ shares×close)/initial_cash - 1) * 100
      Uses TinySoft daily close per symbol for each date; falls back to cash if unavailable.
    """
    pos_file = _position_file_for_signature(signature)
    all_items = _read_jsonl_tail(pos_file, limit=100000)
    if not all_items:
        return {"items": [], "valuation_used": valuation}

    # group by date -> pick last record (max id)
    by_date: Dict[str, Dict[str, Any]] = {}
    for it in all_items:
        d = it.get("date")
        if not d:
            continue
        prev = by_date.get(d)
        if prev is None or (it.get("id", -1) > prev.get("id", -1)):
            by_date[d] = it

    dates_sorted = sorted(by_date.keys())
    # limit to last N days
    if len(dates_sorted) > days:
        dates_sorted = dates_sorted[-days:]

    initial_cash = _load_initial_cash()

    out: list[dict] = []
    if valuation.lower() != "equity":
        for d in dates_sorted:
            rec = by_date[d]
            positions = rec.get("positions", {}) or {}
            cash = positions.get("CASH", 0.0) or 0.0
            try:
                cash_val = float(cash)
            except Exception:
                cash_val = 0.0
            ret_pct = (cash_val / initial_cash - 1.0) * 100.0
            out.append({
                "date": d,
                "returnPct": ret_pct,
                "cash": cash_val,
                "id": rec.get("id")
            })
        return {"items": out, "valuation_used": "cash"}

    for d in dates_sorted:
        rec = by_date[d]
        _, equity_val, _, source = _estimate_equity_for_positions(
            rec.get("positions", {}) or {}, rec.get("decision_time"), rec.get("date")
        )
        ret_pct = (equity_val / initial_cash - 1.0) * 100.0 if initial_cash > 0 else 0.0
        out.append({
            "date": d,
            "returnPct": ret_pct,
            "equity": equity_val,
            "id": rec.get("id"),
            "valuation_source": source,
        })

    return {"items": out, "valuation_used": "equity_cached"}


@app.get("/api/live/model-stats")
async def live_model_stats(signature: str | None = None):
    """Get comprehensive model statistics including latest position, returns, and risk metrics.
    
    Returns:
        - latest_position: most recent portfolio snapshot
        - total_return_pct: overall return since inception
        - daily_returns: list of daily return percentages (for Sharpe calculation)
        - max_drawdown: maximum drawdown percentage
        - position_count: number of current holdings (excluding cash)
        - last_action: most recent trade action
    """
    pos_file = _position_file_for_signature(signature)
    all_items = _read_jsonl_tail(pos_file, limit=100000)
    if not all_items:
        return {"error": "No position data found"}
    
    initial_cash = _load_initial_cash()
    
    # Get latest position
    latest = all_items[-1]
    cash, current_equity, holdings_snapshot, positions, _, equity_source = _build_holdings_with_prices(signature)
    positions = positions or latest.get("positions", {}) or {}
    
    # Count holdings
    holdings_count = len(holdings_snapshot)
    
    total_return_pct = (current_equity / initial_cash - 1.0) * 100.0
    trade_count = sum(
        1
        for rec in all_items
        if (rec.get("this_action") or {}).get("action") in {"buy", "sell"}
    )
    
    # Build equity timeline (per date -> latest record)
    by_date_records: Dict[str, Dict[str, Any]] = {}
    for rec in all_items:
        d = rec.get("date")
        if not d:
            continue
        prev = by_date_records.get(d)
        if prev is None or (rec.get("id", -1) > prev.get("id", -1)):
            by_date_records[d] = rec

    dates_sorted = sorted(by_date_records.keys())
    equity_by_date: Dict[str, float] = {}
    for d in dates_sorted:
        rec = by_date_records[d]
        _, equity_val, _, _ = _estimate_equity_for_positions(
            rec.get("positions", {}) or {}, rec.get("decision_time"), rec.get("date")
        )
        equity_by_date[d] = equity_val

    # Build step-level equity timeline for intraday fallback
    records_sorted = sorted(
        all_items,
        key=lambda item: (
            item.get("date", ""),
            item.get("decision_time", ""),
            item.get("id", 0),
        ),
    )
    equity_steps: List[tuple[str, float]] = []
    for rec in records_sorted:
        _, equity_val, _, _ = _estimate_equity_for_positions(
            rec.get("positions", {}) or {}, rec.get("decision_time"), rec.get("date")
        )
        ts = rec.get("decision_time") or rec.get("date") or str(rec.get("id"))
        equity_steps.append((ts, equity_val))
    
    daily_returns = []
    for i in range(1, len(dates_sorted)):
        prev_val = equity_by_date[dates_sorted[i-1]]
        curr_val = equity_by_date[dates_sorted[i]]
        if prev_val > 0:
            daily_ret = (curr_val / prev_val - 1.0) * 100.0
            daily_returns.append(daily_ret)

    # Fallback to intraday steps if we lack multi-day data
    if len(dates_sorted) <= 1 and len(daily_returns) <= 1 and len(equity_steps) > 1:
        daily_returns = []
        for i in range(1, len(equity_steps)):
            prev_val = equity_steps[i-1][1]
            curr_val = equity_steps[i][1]
            if prev_val > 0:
                daily_returns.append((curr_val / prev_val - 1.0) * 100.0)
    
    # Calculate max drawdown
    peak = initial_cash
    max_dd = 0.0
    if len(dates_sorted) > 1:
        iterable = (equity_by_date[d] for d in dates_sorted)
    else:
        iterable = (val for _, val in equity_steps)
    for val in iterable:
        if val > peak:
            peak = val
        dd = (val / peak - 1.0) * 100.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    
    # Calculate Sharpe ratio (annualized, assuming 252 trading days)
    import statistics
    sharpe = 0.0
    if len(daily_returns) > 1:
        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns)
        if std_return > 0:
            sharpe = (mean_return * 252**0.5) / std_return
    
    return {
        "signature": signature or _load_runtime_signature(),
        "latest_date": latest.get("date"),
        "total_return_pct": round(total_return_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "position_count": holdings_count,
        "cash": round(cash, 2),
        "equity": round(current_equity, 2),
        "last_action": latest.get("this_action"),
        "total_records": len(all_items),
        "trade_count": trade_count,
        "holdings": holdings_snapshot,
        "valuation_source": equity_source,
    }


@app.get("/api/live/current-positions")
async def live_current_positions(signature: str | None = None):
    """Get current portfolio positions with real-time pricing.
    
    Returns list of current holdings with:
    - symbol: stock code
    - shares: number of shares held
    - purchase_date: when position was opened
    - current_price: latest price (from TinySoft if available)
    - entry_price: average cost basis
    - pnl: unrealized profit/loss
    - pnl_percent: return percentage
    """
    cash, total_equity, holdings, _, latest_date, equity_source = _build_holdings_with_prices(signature)
    return {
        "positions": holdings,
        "cash": round(cash, 2),
        "total_equity": round(total_equity, 2),
        "date": latest_date,
        "valuation_source": equity_source,
    }


@app.get("/api/live/stock-detail")
async def live_stock_detail(
    symbol: str = Query(..., description="股票代码，支持 SH/SZ 前缀或 6 位数字"),
    history_limit: int = Query(60, ge=10, le=200),
    news_limit: int = Query(6, ge=1, le=20),
):
    """Return enriched detail for a single stock based on shared snapshot +交易记录。"""
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不能为空")

    normalized = _normalize_code(symbol)
    stock_cache = _load_stock_data_cache()
    stock_payload = stock_cache.get(normalized) or stock_cache.get(symbol.upper())
    if not stock_payload:
        raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的行情缓存")

    ai_interest = _collect_ai_interest([normalized]).get(normalized)

    hourly_raw = stock_payload.get("小时线行情", []) or []
    hourly_window = hourly_raw[-history_limit:]
    hourly_series = [
        {
            "time": item.get("date"),
            "price": item.get("close"),
            "volume": item.get("vol"),
            "amount": item.get("amount"),
            "bid": item.get("buy1"),
        }
        for item in hourly_window
        if item.get("date")
    ]

    indicator_latest = None
    indicators = stock_payload.get("小时线指标") or stock_payload.get("日线指标") or []
    if isinstance(indicators, list) and indicators:
        indicator_latest = indicators[-1]

    summary = {
        "symbol": normalized,
        "name": stock_payload.get("名称") or stock_payload.get("name"),
        "latest_time": hourly_window[-1].get("date") if hourly_window else None,
        "latest_price": hourly_window[-1].get("close") if hourly_window else None,
        "change_percent": stock_payload.get("涨跌幅"),
        "turnover_rate": stock_payload.get("换手率"),
        "volume": stock_payload.get("成交量"),
    }

    # Aggregate AI holdings
    config_path = Path(__file__).parent / "configs" / "default_config.json"
    try:
        config = _load_config_dict(config_path)
        candidate_models = [
            m.get("signature")
            for m in config.get("models", [])
            if m.get("signature") and m.get("enabled", True)
        ]
    except Exception:
        candidate_models = []

    ai_positions: List[Dict[str, Any]] = []
    ai_trades: List[Dict[str, Any]] = []
    for sig in candidate_models:
        try:
            pos_file = _position_file_for_signature(sig)
            records = _read_jsonl_tail(pos_file, limit=200)
            if not records:
                continue
        except Exception:
            continue

        latest = records[-1]
        latest_date = latest.get("date")
        positions = latest.get("positions", {}) or {}
        detail = positions.get(normalized) or positions.get(symbol.upper())
        decision_time = latest.get("decision_time")
        record_date = latest.get("date")

        latest_position_price = None
        if isinstance(detail, dict) and detail.get("shares", 0) > 0:
            shares = detail.get("shares", 0)
            entry_price = detail.get("avg_price")
            mark_price, mark_source = _lookup_cached_price(normalized, decision_time, record_date)
            if mark_price is None:
                mark_price = entry_price or 0.0
                mark_source = "fallback"
            latest_position_price = mark_price
            cost = (float(entry_price) if entry_price else float(mark_price)) * shares
            market_val = float(mark_price) * shares
            pnl = market_val - cost
            pnl_pct = (pnl / cost * 100) if cost else 0.0
            ai_positions.append({
                "signature": sig,
                "shares": shares,
                "avg_price": entry_price,
                "mark_price": mark_price,
                "market_value": round(market_val, 2),
                "pnl": round(pnl, 2),
                "pnl_percent": round(pnl_pct, 2),
                "valuation_source": mark_source,
            })

        # recent trades for this symbol
        latest_dt = None
        if latest_date:
            try:
                latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            except Exception:
                latest_dt = None

        for rec in reversed(records):
            rec_date = rec.get("date")
            if latest_dt and rec_date:
                try:
                    rec_dt = datetime.strptime(rec_date, "%Y-%m-%d")
                    if (latest_dt - rec_dt).days > 3:
                        continue
                except Exception:
                    pass
            action = rec.get("this_action") or {}
            if action.get("symbol") not in {normalized, symbol.upper(), symbol}:
                continue
            trade_price, _ = _lookup_cached_price(normalized, rec.get("decision_time"), rec.get("date"))
            if trade_price is None:
                trade_price = (
                    action.get("price")
                    or action.get("avg_price")
                    or latest_position_price
                    or summary.get("latest_price")
                )
            ai_trades.append({
                "signature": sig,
                "date": rec.get("date"),
                "decision_time": rec.get("decision_time"),
                "action": action.get("action"),
                "amount": action.get("amount"),
                "cash": float((rec.get("positions", {}) or {}).get("CASH", 0.0) or 0.0),
                "id": rec.get("id"),
                "price": trade_price,
            })
            if len(ai_trades) >= 50:
                break

    # News reuse existing endpoint
    news_payload = await live_latest_news(limit=news_limit, symbols=normalized)

    return {
        "summary": summary,
        "hourly_prices": hourly_series,
        "indicators": indicator_latest,
        "ai_positions": ai_positions,
        "ai_trades": ai_trades[: news_limit * 4],
        "ai_summary": {
            "holding_count": ai_interest.get("holding_count") if ai_interest else len(ai_positions),
            "trade_volume": ai_interest.get("trade_volume") if ai_interest else sum(
                max(0, pos.get("shares", 0)) for pos in ai_positions
            ),
            "turnover_percent": ai_interest.get("turnover_percent") if ai_interest else None,
            "holding_models": ai_interest.get("holding_models") if ai_interest else [pos.get("signature") for pos in ai_positions],
        },
        "news": news_payload.get("news", []),
    }


@app.get("/api/live/recent-decisions")
async def live_recent_decisions(
    signature: str | None = None,
    limit: int = Query(20, ge=1, le=100)
):
    """Get recent trading decisions with timestamps and actions.
    
    Returns list of decisions sorted by most recent first, including:
    - date, decision_time, decision_count
    - action type (buy/sell/no_trade)
    - symbol and amount
    - resulting cash and position count
    """
    pos_file = _position_file_for_signature(signature)
    all_items = _read_jsonl_tail(pos_file, limit=limit * 2)  # Get more to ensure we have enough after filtering
    
    decisions = []
    for it in reversed(all_items):  # Most recent first
        action = it.get("this_action") or {}
        positions = it.get("positions", {}) or {}
        
        # Count holdings
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
            "cash": float(positions.get("CASH", 0.0) or 0.0),
            "holdings": holdings,
            "id": it.get("id"),
        })
        
        if len(decisions) >= limit:
            break
    
    return {"decisions": decisions}


@app.get("/api/live/news")
async def live_latest_news(
    limit: int = Query(10, ge=1, le=50),
    symbols: str | None = None
):
    """Get latest news from news.csv, optionally filtered by symbols.
    
    Args:
        limit: number of news items to return
        symbols: comma-separated stock codes (e.g., "SH688008,SH688111")
    """
    import pandas as pd
    from pathlib import Path
    
    news_path = Path(__file__).parent / "data" / "news.csv"
    if not news_path.exists():
        return {"news": [], "note": "No news data available"}
    
    try:
        # Try different encodings
        df = None
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']:
            try:
                df = pd.read_csv(news_path, encoding=encoding)
                break
            except Exception:
                continue
        
        if df is None or df.empty:
            return {"news": [], "note": "Failed to read news data"}
        
        # Filter by symbols if provided
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if 'symbol' in df.columns:
                df = df[df['symbol'].str.upper().isin(symbol_list)]
        
        # Sort by publish_time or search_time (most recent first)
        time_col = 'publish_time' if 'publish_time' in df.columns else 'search_time'
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            df = df.sort_values(by=time_col, ascending=False)
        
        # Get latest N items
        df = df.head(limit)
        
        # Convert to dict
        news_items = []
        for _, row in df.iterrows():
            news_items.append({
                "title": str(row.get('title', '')),
                "content": str(row.get('content', ''))[:200],  # Truncate long content
                "publish_time": str(row.get('publish_time', '')),
                "symbol": str(row.get('symbol', '')),
                "source": str(row.get('source', 'Unknown')),
                "url": str(row.get('url', '')),
            })
        
        return {"news": news_items}
    
    except Exception as e:
        print(f"⚠️ News endpoint error: {e}")
        return {"news": [], "error": "新闻源暂时不可用，请稍后重试。"}


# =============== Market data (TinySoft) ===============
def _get_tsl_credentials() -> tuple[str | None, str | None, str, int]:
    import os
    username = os.getenv("TSL_USERNAME") or os.getenv("TSL_USER")
    password = os.getenv("TSL_PASSWORD") or os.getenv("TSL_PASS")
    server = os.getenv("TSL_SERVER", "tsl.tinysoft.com.cn")
    try:
        port = int(os.getenv("TSL_PORT", "443"))
    except Exception:
        port = 443
    return username, password, server, port


def _normalize_code(c: str) -> str:
    c = (c or "").strip().upper()
    if len(c) == 6 and c.isdigit():
        # heuristic for A-share prefixes
        if c.startswith(("688", "689", "600", "601", "603", "605", "730", "735")):
            return f"SH{c}"
        if c.startswith(("000", "001", "002", "003", "300", "301", "302")):
            return f"SZ{c}"
    return c


def _resolve_stock_payload(symbol: str) -> tuple[str, Optional[Dict[str, Any]]]:
    """Return (normalized_code, payload_from_cache_or_None)."""
    data = _load_stock_data_cache()
    for cand in _symbol_candidates(symbol):
        payload = data.get(cand)
        if payload:
            return cand, payload
    return _normalize_code(symbol), None


def _build_snapshot_quotes(code_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Build quote map using共享快照，提供价格/涨跌幅/成交量等近似值。"""
    quotes: Dict[str, Dict[str, Any]] = {}
    for code in code_list:
        normalized, payload = _resolve_stock_payload(code)
        price = 0.0
        volume = 0.0
        turnover = 0.0
        change_pct = 0.0
        if payload:
            hourly = payload.get("小时线行情") or []
            if hourly:
                last = hourly[-1]
                price = float(last.get("buy1") or last.get("close") or 0.0)
                volume = float(last.get("vol") or 0.0)
                amount = float(last.get("amount") or 0.0)
                turnover = round(amount / 1e8, 4)
            daily = payload.get("日线行情") or []
            if len(daily) >= 2:
                last_close = float(daily[-1].get("close") or 0.0)
                prev_close = float(daily[-2].get("close") or 0.0)
                if price <= 0:
                    price = last_close
                if prev_close > 0 and (price or last_close) > 0:
                    base = price or last_close
                    change_pct = (base / prev_close - 1.0) * 100.0
            elif daily:
                last_close = float(daily[-1].get("close") or 0.0)
                if price <= 0:
                    price = last_close

        quotes[normalized] = {
            "code": normalized,
            "price": round(price, 4),
            "changePercent": round(change_pct, 4),
            "volume": volume,
            "turnover": round(turnover, 4),
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": "shared-snapshot",
        }
    return quotes


@app.get("/api/market/quotes")
async def market_quotes(codes: str):
    """Return live quotes for comma-separated codes (e.g., SH688008,SZ300750).

    Response shape: {quotes: [{code, price, changePercent, volume, turnover, ts}]}
    Fields may be 0 if unavailable; no names/sectors are returned (frontend merges by code).
    """
    code_list = [
        _normalize_code(x) for x in (codes or "").split(",") if x.strip()
    ]
    if not code_list:
        raise HTTPException(status_code=400, detail="codes query param is required")

    ai_interest = _collect_ai_interest(code_list)
    quotes_map: Dict[str, Dict[str, Any]] = {}
    ts_error: Optional[str] = None
    client = None

    try:
        import pyTSL as ts  # type: ignore
    except Exception as e:
        ts = None  # type: ignore
        ts_error = f"pyTSL not available: {e}"

    if ts_error is None and ts is not None:
        user, pwd, server, port = _get_tsl_credentials()
        if not user or not pwd:
            ts_error = "Missing TinySoft credentials"
        else:
            try:
                client = ts.Client(user, pwd, server, port)
                ok = client.login()
                if ok != 1:
                    last_err = getattr(client, "last_error", lambda: "login failed")()
                    ts_error = f"TinySoft login failed: {last_err}"
            except Exception as e:
                ts_error = f"TinySoft connection error: {e}"

    if client and ts_error is None:
        try:
            now = datetime.now()
            begin_time = now - timedelta(days=3)
            end_time = now

            for code in code_list:
                price = 0.0
                change_pct = 0.0
                volume = 0.0
                turnover = 0.0
                try:
                    r_hour = client.query(
                        stock=code,
                        begin_time=begin_time,
                        end_time=end_time,
                        cycle='60分钟线',
                        fields='date, close, vol, amount, buy1'
                    )
                    if r_hour.error() == 0:
                        df = r_hour.dataframe()
                        if not df.empty:
                            last = df.iloc[-1]
                            price = float(last.get('buy1') or last.get('close') or 0.0)
                            volume = float(last.get('vol') or 0.0)
                            amt = float(last.get('amount') or 0.0)
                            turnover = round(amt / 1e8, 4)

                    r_day = client.query(
                        stock=code,
                        begin_time=begin_time,
                        end_time=end_time,
                        cycle='日线',
                        fields='date, close'
                    )
                    if r_day.error() == 0:
                        ddf = r_day.dataframe()
                        if len(ddf) >= 2:
                            prev_close = float(ddf.iloc[-2].get('close') or 0.0)
                            if prev_close > 0 and price > 0:
                                change_pct = (price / prev_close - 1.0) * 100.0
                except Exception:
                    pass

                quotes_map[code] = {
                    "code": code,
                    "price": round(price, 4),
                    "changePercent": round(change_pct, 4),
                    "volume": volume,
                    "turnover": turnover,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "source": "tinystock",
                }
                ai = ai_interest.get(code, {})
                quotes_map[code].update({
                    "aiHoldingCount": ai.get("holding_count", 0),
                    "aiTotalShares": ai.get("total_shares", 0),
                    "aiTradeVolume": ai.get("trade_volume", 0),
                    "aiTurnoverPercent": ai.get("turnover_percent", 0.0),
                    "aiAttentionScore": ai.get("attention_score", 0.0),
                })
        finally:
            try:
                client.logout()
            except Exception:
                pass

    fallback_codes = [
        code for code in code_list
        if code not in quotes_map or quotes_map[code]["price"] <= 0
    ]
    fallback_map: Dict[str, Dict[str, Any]] = {}
    if fallback_codes:
        fallback_map = _build_snapshot_quotes(fallback_codes)
        for code, entry in fallback_map.items():
            ai = ai_interest.get(code, {})
            entry.update({
                "aiHoldingCount": ai.get("holding_count", 0),
                "aiTotalShares": ai.get("total_shares", 0),
                "aiTradeVolume": ai.get("trade_volume", 0),
                "aiTurnoverPercent": ai.get("turnover_percent", 0.0),
                "aiAttentionScore": ai.get("attention_score", 0.0),
            })
            quotes_map[code] = entry

    ordered_quotes: List[Dict[str, Any]] = []
    for code in code_list:
        entry = quotes_map.get(code)
        if entry:
            ordered_quotes.append(entry)
        elif fallback_map:
            # attempt to fallback via normalized key
            fallback = fallback_map.get(code)
            if fallback:
                ordered_quotes.append(fallback)

    source = "tinystock"
    if fallback_map and client:
        source = "mixed"
    elif fallback_map:
        source = "shared-snapshot"

    return {
        "quotes": ordered_quotes,
        "source": source,
        "note": ts_error,
    }


# =============== Optional hourly scheduler ===============
_SCHEDULER_ENABLED = os.getenv("ENABLE_HOURLY_TRADING", "false").lower() in ("1", "true", "yes")
_SCHEDULER_TASK = None

async def _sleep_until_next_hour():
    import asyncio
    from datetime import datetime, timedelta
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    await asyncio.sleep((next_hour - now).total_seconds())

async def _hourly_trading_loop():
    import asyncio
    while True:
        await _sleep_until_next_hour()
        try:
            # start a trading job using the same subprocess mechanism
            cmd = [sys.executable, str(Path(__file__).parent / "main.py")]
            cfg_path = os.getenv("TRADING_CONFIG_PATH")
            if cfg_path:
                cmd.append(cfg_path)
            job_id = f"hourly-{datetime.utcnow().strftime('%Y%m%d%H')}-{uuid.uuid4().hex[:8]}"
            log_file = LOG_DIR / f"{job_id}.log"
            lf = open(log_file, "wb")
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                stdout=lf,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            JOBS[job_id] = {
                "id": job_id,
                "pid": proc.pid,
                "started_at": datetime.utcnow().isoformat() + "Z",
                "status": "running",
                "returncode": None,
                "log_file": str(log_file),
                "process": proc,
            }
        except Exception:
            # swallow exceptions to keep loop alive
            pass

@app.on_event("startup")
async def _maybe_start_scheduler():
    global _SCHEDULER_TASK
    if _SCHEDULER_ENABLED and _SCHEDULER_TASK is None:
        import asyncio
        _SCHEDULER_TASK = asyncio.create_task(_hourly_trading_loop())

@app.get("/api/scheduler/status")
async def scheduler_status():
    return {
        "enabled": _SCHEDULER_ENABLED,
        "running": _SCHEDULER_TASK is not None,
    }


@app.post("/api/run-trading")
async def run_trading(config_path: str | None = None):
    """Start the trading script (main.py) in a subprocess and return a job id.

    Request body (JSON): {"config_path": "configs/my_config.json"} (optional)
    """
    job_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat() + "Z"
    log_file = LOG_DIR / f"{job_id}.log"

    if not _truthy_env("SKIP_API_BACKUP"):
        ok = run_backup_snapshot(reason="api_run_trading")
        if not ok:
            print("⚠️ Pre-run backup failed (API). Continuing without blocking request.")

    # Build the command
    cmd = [sys.executable, str(Path(__file__).parent / "main.py")]
    if config_path:
        cmd.append(config_path)

    # Open log file and start subprocess
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

    JOBS[job_id] = {
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
        raise HTTPException(status_code=500, detail="无法完成备份，请稍后重试。")
    return {"status": "ok", "retain": retain}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if process still exists
    proc: subprocess.Popen = job.get("process")
    if proc is not None:
        # Poll the process to update status
        rc = proc.poll()
        if rc is None:
            status = "running"
        else:
            status = "finished" if rc == 0 else "failed"
            job["returncode"] = rc
            job["status"] = status
            # Close process handle
            job.pop("process", None)
    else:
        # Process already completed, use stored status
        status = job.get("status", "unknown")

    job["status"] = status

    # Read last part of log file (if exists)
    log_text = None
    try:
        lfpath = Path(job["log_file"])
        if lfpath.exists():
            with open(lfpath, "r", encoding="utf-8", errors="ignore") as f:
                # Return last ~2000 chars
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
    for j in JOBS.values():
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
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    proc: subprocess.Popen = job.get("process")
    if not proc:
        raise HTTPException(status_code=400, detail="Process handle not available (already finished)")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    job["status"] = "terminated"
    job.pop("process", None)
    return {"id": job_id, "status": job["status"]}
