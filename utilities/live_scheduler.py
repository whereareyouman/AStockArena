#!/usr/bin/env python3
"""Long-running live scheduler for real-time paper benchmark runs.

This module intentionally stays above the trading workflow: it waits for
decision windows, prepares shared inputs once, launches isolated model workers,
and writes operational evidence. AgenticWorkflow/main.py still own all trading
decisions and paper-position updates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional for smoke tests
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as trading_main
from agent_engine.agent.agent import AgenticWorkflow
from agent_engine.shared_prefetch import _FileLock
from utils.news_cache_guard import NewsCacheIntegrityError, validate_news_cache_integrity
from utils.position_manager import normalize_symbol


DEFAULT_DECISION_TIMES = ("10:30:00", "11:30:00", "14:00:00")
DEFAULT_RETRY_BACKOFF = (5.0, 30.0, 120.0)
TERMINAL_EVENT_STATUSES = {"succeeded", "partial_failed", "failed", "skipped"}
STATE_DIR = PROJECT_ROOT / "jobs" / "live_scheduler"
STATE_PATH = STATE_DIR / "state.json"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
RUNTIME_DIR = PROJECT_ROOT / "settings" / "runtime" / "live_scheduler"
ERROR_FIELDS = [
    "date",
    "decision_time",
    "decision_count",
    "stage",
    "model_signature",
    "symbol",
    "attempts",
    "error_type",
    "error_message",
    "human_message",
    "final_action",
    "log_path",
    "snapshot_path",
    "created_at",
]


@dataclass(frozen=True)
class DecisionEvent:
    date: str
    decision_time: str
    decision_count: int
    target_dt: datetime

    @property
    def key(self) -> str:
        return f"{self.date}#{self.decision_count}#{self.decision_time}"

    @property
    def safe_time(self) -> str:
        return self.decision_time.replace(":", "-").replace(" ", "_")

    @property
    def clock_dir(self) -> str:
        return self.decision_time.split()[-1].replace(":", "-")


class SnapshotValidationError(RuntimeError):
    """Raised when the shared snapshot is incomplete for live paper runs."""


def _now_cn() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _json_safe(value: Any, *, _seen: Optional[set[int]] = None) -> Any:
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value_id = id(value)
    if value_id in _seen:
        return "<circular>"
    if isinstance(value, dict):
        _seen.add(value_id)
        try:
            return {str(k): _json_safe(v, _seen=_seen) for k, v in value.items()}
        finally:
            _seen.discard(value_id)
    if isinstance(value, (list, tuple, set)):
        _seen.add(value_id)
        try:
            return [_json_safe(v, _seen=_seen) for v in value]
        finally:
            _seen.discard(value_id)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)


def _load_state(path: Path = STATE_PATH) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("events", {})
                return data
    except Exception as exc:
        try:
            corrupt_path = path.with_name(f"{path.name}.corrupt.{os.getpid()}.{time.time_ns()}")
            path.replace(corrupt_path)
            print(f"State file was unreadable and was moved to {corrupt_path}: {exc}", flush=True)
        except Exception:
            pass
    return {"events": {}}


def _save_state(state: Dict[str, Any], path: Path = STATE_PATH) -> None:
    _atomic_write_json(path, state)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _int_config(config: Dict[str, Any], key: str, default: int, *, min_value: Optional[int] = None) -> int:
    try:
        value = int(float((config.get("run_config") or {}).get(key, default)))
    except Exception:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


def _float_config(config: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float((config.get("run_config") or {}).get(key, default))
    except Exception:
        return default


def _path_config(config: Dict[str, Any], key: str, default: Path) -> Path:
    raw = (config.get("run_config") or {}).get(key)
    path = Path(str(raw)).expanduser() if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def retry_attempts(config: Dict[str, Any]) -> int:
    return _int_config(config, "live_retry_attempts", 3, min_value=3)


def model_parallelism(config: Dict[str, Any]) -> int:
    return _int_config(config, "live_scheduler_model_parallelism", 1, min_value=1)


def model_start_delay_seconds(config: Dict[str, Any]) -> float:
    run_config = config.get("run_config") or {}
    if "live_scheduler_model_start_delay_seconds" in run_config:
        return max(0.0, _float_config(config, "live_scheduler_model_start_delay_seconds", 0.0))
    return max(0.0, _float_config(config, "parallel_spawn_delay_seconds", 0.0))


def retry_backoffs(config: Dict[str, Any]) -> List[float]:
    raw = (config.get("run_config") or {}).get("live_retry_backoff_seconds")
    if isinstance(raw, list) and raw:
        values: List[float] = []
        for item in raw:
            try:
                values.append(max(0.0, float(item)))
            except Exception:
                continue
        if values:
            return values
    return list(DEFAULT_RETRY_BACKOFF)


def _backoff_for_attempt(config: Dict[str, Any], attempt: int) -> float:
    backoffs = retry_backoffs(config)
    return backoffs[min(max(attempt - 1, 0), len(backoffs) - 1)]


def _decision_times(config: Dict[str, Any]) -> List[str]:
    raw = (config.get("run_config") or {}).get("live_decision_times") or DEFAULT_DECISION_TIMES
    times: List[str] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        if len(text) == 5:
            text = f"{text}:00"
        datetime.strptime(text, "%H:%M:%S")
        times.append(text)
    return times or list(DEFAULT_DECISION_TIMES)


def _enabled_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [model for model in config.get("models", []) if model.get("enabled", True)]


def _calendar_agent() -> AgenticWorkflow:
    return AgenticWorkflow.__new__(AgenticWorkflow)


def is_trading_day(date_str: str) -> bool:
    return bool(_calendar_agent()._is_trading_day(date_str))


def _event_for(date_str: str, decision_clock: str, decision_count: int) -> DecisionEvent:
    target_dt = datetime.strptime(f"{date_str} {decision_clock}", "%Y-%m-%d %H:%M:%S")
    return DecisionEvent(
        date=date_str,
        decision_time=f"{date_str} {decision_clock}",
        decision_count=decision_count,
        target_dt=target_dt,
    )


def iter_future_events(config: Dict[str, Any], now: Optional[datetime] = None, max_days: int = 35) -> Iterable[DecisionEvent]:
    now = now or _now_cn()
    for offset in range(max_days + 1):
        day = now.date() + timedelta(days=offset)
        date_str = day.strftime("%Y-%m-%d")
        if not is_trading_day(date_str):
            continue
        for idx, decision_clock in enumerate(_decision_times(config), 1):
            yield _event_for(date_str, decision_clock, idx)


def mark_missed_events(config: Dict[str, Any], state: Dict[str, Any], now: Optional[datetime] = None) -> int:
    now = now or _now_cn()
    catchup_minutes = _float_config(config, "live_scheduler_catchup_minutes", 20.0)
    events = state.setdefault("events", {})
    marked = 0
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for event in iter_future_events(config, start_of_day, max_days=0):
        if event.target_dt + timedelta(minutes=catchup_minutes) >= now:
            continue
        if events.get(event.key, {}).get("status") in TERMINAL_EVENT_STATUSES:
            continue
        events[event.key] = {
            **events.get(event.key, {}),
            "date": event.date,
            "decision_time": event.decision_time,
            "decision_count": event.decision_count,
            "status": "skipped",
            "finished_at": _now_cn().isoformat(timespec="seconds"),
            "reason": "missed_catchup_window",
        }
        marked += 1
    return marked


def next_pending_event(config: Dict[str, Any], state: Dict[str, Any], now: Optional[datetime] = None) -> Optional[DecisionEvent]:
    now = now or _now_cn()
    catchup_minutes = _float_config(config, "live_scheduler_catchup_minutes", 20.0)
    events = state.setdefault("events", {})
    for event in iter_future_events(config, now, max_days=35):
        status = events.get(event.key, {}).get("status")
        if status in TERMINAL_EVENT_STATUSES:
            continue
        if event.target_dt + timedelta(minutes=catchup_minutes) < now:
            continue
        return event
    return None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in str(value)) or "model"


def _tail_text(path: Optional[Path], max_chars: int = 12000) -> str:
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-max_chars:]


def _error_dir(config: Dict[str, Any], event: DecisionEvent) -> Path:
    root = _path_config(config, "live_error_dir", STATE_DIR / "errors")
    path = root / event.date / event.clock_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _human_error_message(
    *,
    event: DecisionEvent,
    stage: str,
    symbol: str,
    attempts: int,
    error_type: str,
    error_message: str,
) -> str:
    return (
        f"{symbol} 在 {event.decision_time} 第{event.decision_count}次操盘的 "
        f"{stage} 阶段发生 {error_type}：{error_message}，已重试{attempts}次。"
    )


def write_error_artifacts(
    config: Dict[str, Any],
    event: DecisionEvent,
    *,
    stage: str,
    model_signature: str = "ALL",
    symbol: str = "ALL",
    attempts: int = 1,
    error: Optional[BaseException] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    traceback_text: Optional[str] = None,
    final_action: str = "continue",
    log_path: Optional[Path] = None,
    snapshot_path: Optional[str] = None,
    stdout_tail: Optional[str] = None,
    stderr_tail: Optional[str] = None,
    returncode: Optional[int] = None,
) -> Path:
    err_type = error_type or (type(error).__name__ if error else "Error")
    err_msg = error_message if error_message is not None else (str(error) if error else "")
    if traceback_text is None and error is not None:
        traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    traceback_text = traceback_text or ""
    safe_stage = _safe_name(stage)
    safe_model = _safe_name(model_signature)
    created_at = _now_cn().isoformat(timespec="seconds")
    human = _human_error_message(
        event=event,
        stage=stage,
        symbol=symbol or "UNKNOWN",
        attempts=attempts,
        error_type=err_type,
        error_message=err_msg,
    )

    directory = _error_dir(config, event)
    summary_path = directory / "error_summary.csv"
    row = {
        "date": event.date,
        "decision_time": event.decision_time,
        "decision_count": event.decision_count,
        "stage": stage,
        "model_signature": model_signature,
        "symbol": symbol or "UNKNOWN",
        "attempts": attempts,
        "error_type": err_type,
        "error_message": err_msg,
        "human_message": human,
        "final_action": final_action,
        "log_path": str(log_path) if log_path else "",
        "snapshot_path": str(snapshot_path or ""),
        "created_at": created_at,
    }
    lock = _FileLock(str(directory / ".error_artifacts.lock"), timeout=30.0)
    with lock:
        write_header = not summary_path.exists()
        with summary_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        details = {
            **row,
            "returncode": returncode,
            "traceback_path": "",
            "stdout_tail_path": "",
            "stderr_tail_path": "",
        }
        suffix = f"{safe_stage}_{safe_model}_{int(time.time())}_{os.getpid()}"
        if traceback_text:
            trace_path = directory / f"traceback_{suffix}.txt"
            trace_path.write_text(traceback_text, encoding="utf-8")
            details["traceback_path"] = str(trace_path)
        if stdout_tail:
            stdout_path = directory / f"stdout_tail_{suffix}.txt"
            stdout_path.write_text(stdout_tail, encoding="utf-8")
            details["stdout_tail_path"] = str(stdout_path)
        if stderr_tail:
            stderr_path = directory / f"stderr_tail_{suffix}.txt"
            stderr_path.write_text(stderr_tail, encoding="utf-8")
            details["stderr_tail_path"] = str(stderr_path)

        with (directory / "error_details.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(details, ensure_ascii=False) + "\n")
    return directory


def _write_heartbeat(
    config: Dict[str, Any],
    *,
    status: str,
    event: Optional[DecisionEvent] = None,
    state: Optional[Dict[str, Any]] = None,
    current_processes: Optional[List[Dict[str, Any]]] = None,
    last_error_dir: Optional[Path] = None,
) -> None:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        events = (state or {}).get("events", {}) if isinstance(state, dict) else {}
        succeeded = [item for item in events.values() if isinstance(item, dict) and item.get("status") == "succeeded"]
        succeeded.sort(key=lambda item: str(item.get("finished_at") or ""))
        payload = {
            "updated_at": _now_cn().isoformat(timespec="seconds"),
            "status": status,
            "current_event": {
                "date": event.date,
                "decision_time": event.decision_time,
                "decision_count": event.decision_count,
            } if event else None,
            "last_success": _json_safe(succeeded[-1]) if succeeded else None,
            "current_processes": _json_safe(current_processes or []),
            "disk": {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
            },
            "last_error_dir": str(last_error_dir) if last_error_dir else None,
            "retry_attempts": retry_attempts(config),
        }
        _atomic_write_json(HEARTBEAT_PATH, payload)
    except Exception as exc:  # pragma: no cover - heartbeat must not block runs
        print(f"Failed to write heartbeat: {exc}", flush=True)


def _run_with_retries(
    config: Dict[str, Any],
    event: DecisionEvent,
    *,
    stage: str,
    symbol: str,
    final_action_on_failure: str,
    func: Callable[[int], Any],
) -> Tuple[bool, Any, Optional[Path]]:
    attempts = retry_attempts(config)
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return True, func(attempt), None
        except Exception as exc:
            last_error = exc
            print(f"{stage} attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(_backoff_for_attempt(config, attempt))
    error_dir = write_error_artifacts(
        config,
        event,
        stage=stage,
        symbol=symbol,
        attempts=attempts,
        error=last_error,
        final_action=final_action_on_failure,
    )
    return False, None, error_dir


def build_tick_config(
    base_config: Dict[str, Any],
    event: DecisionEvent,
    model_config: Dict[str, Any],
    *,
    live_backtest_mode: bool,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["date_range"] = {"init_date": event.date, "end_date": event.date}
    run_config = config.setdefault("run_config", {})
    run_config["realtime_mode"] = "stop"
    run_config["parallel_run"] = False
    run_config["prefetch_news_before_run"] = False
    run_config["backtest_mode"] = bool(live_backtest_mode)
    config["models"] = [{**copy.deepcopy(model_config), "enabled": True}]
    return config


def _write_tick_config(config: Dict[str, Any], event: DecisionEvent, signature: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNTIME_DIR / f"config_{event.date}_{event.safe_time}_{_safe_name(signature)}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prefetch_news_for_event(config: Dict[str, Any], event: DecisionEvent) -> bool:
    run_config = config.get("run_config", {})
    if _truthy(os.getenv("ASTOCK_SKIP_EVENT_NEWS_PREFETCH"), default=False):
        print("News prefetch skipped by ASTOCK_SKIP_EVENT_NEWS_PREFETCH=1", flush=True)
        return True
    if not _truthy(run_config.get("live_prefetch_news_before_decision"), default=True):
        print("News prefetch skipped by live_prefetch_news_before_decision=false", flush=True)
        return True
    tick_config = copy.deepcopy(config)
    tick_config["date_range"] = {"init_date": event.date, "end_date": event.date}
    tick_run_config = tick_config.setdefault("run_config", {})
    tick_run_config["prefetch_news_before_run"] = True
    tick_run_config.setdefault("prefetch_news_respect_init_date", False)
    trading_main.prefetch_configured_news_before_run(tick_config, AgenticWorkflow.DEFAULT_STOCK_SYMBOLS)
    return True


def _snapshot_agent(config: Dict[str, Any], event: DecisionEvent) -> AgenticWorkflow:
    agent_config = config.get("agent_config", {})
    data_config = config.get("data_config", {})
    log_config = config.get("log_config", {})
    trading_rules = config.get("trading_rules", {})
    risk_management = config.get("risk_management", {})
    run_config = config.get("run_config", {})
    if "snapshot_hourly_cache_days" in run_config:
        os.environ["SNAPSHOT_HOURLY_CACHE_DAYS"] = str(run_config.get("snapshot_hourly_cache_days"))
    enabled_model = next(iter(_enabled_models(config)), {})
    agent = AgenticWorkflow(
        signature="snapshot-preparer",
        basemodel=enabled_model.get("basemodel", "snapshot-preparer"),
        stock_symbols=AgenticWorkflow.DEFAULT_STOCK_SYMBOLS,
        stock_json_path=data_config.get("stock_json_path", "./data_flow/ai_stock_data.json"),
        news_csv_path=data_config.get("news_csv_path", "./data_flow/news.csv"),
        macro_csv_path=None,
        log_path=log_config.get("log_path", "./data_flow/trading_summary_each_agent"),
        openai_base_url=enabled_model.get("openai_base_url"),
        openai_api_key="snapshot-preparer",
        max_steps=agent_config.get("max_steps", 10),
        max_retries=agent_config.get("max_retries", 3),
        base_delay=agent_config.get("base_delay", 0.5),
        initial_cash=agent_config.get("initial_cash", 1000000.0),
        init_date=event.date,
        trading_rules=trading_rules,
        risk_management=risk_management,
        force_replay=False,
    )
    agent._persist_stock_data_during_snapshot = bool(run_config.get("persist_stock_data_during_snapshot", False))
    return agent


def _validate_snapshot_payload(agent: AgenticWorkflow, payload: Dict[str, Any]) -> None:
    missing_prices: List[str] = []
    missing_indicators: List[str] = []
    prices = payload.get("prices") if isinstance(payload, dict) else {}
    indicators = payload.get("indicators") if isinstance(payload, dict) else {}
    for symbol in sorted(agent.allowed_symbols):
        price_payload = prices.get(symbol) if isinstance(prices, dict) else None
        summary = price_payload.get("summary") if isinstance(price_payload, dict) else None
        if not isinstance(summary, dict) or summary.get("close") in (None, "", 0):
            missing_prices.append(symbol)
        indicator_payload = indicators.get(symbol) if isinstance(indicators, dict) else None
        indicator_values = indicator_payload.get("indicators") if isinstance(indicator_payload, dict) else None
        if not isinstance(indicator_values, dict) or not indicator_values:
            missing_indicators.append(symbol)
    if missing_prices or missing_indicators:
        raise SnapshotValidationError(
            "shared snapshot incomplete: "
            f"missing_prices={missing_prices}; missing_indicators={missing_indicators}"
        )


def prepare_snapshot_for_event(config: Dict[str, Any], event: DecisionEvent) -> Dict[str, Any]:
    if not _truthy((config.get("run_config") or {}).get("live_snapshot_before_models"), default=True):
        return {"skipped": True}
    news_path = Path((config.get("data_config") or {}).get("news_csv_path", "./data_flow/news.csv"))
    if not news_path.is_absolute():
        news_path = PROJECT_ROOT / news_path
    validate_news_cache_integrity(news_path, strict=True)
    agent = _snapshot_agent(config, event)
    try:
        symbols_signature = agent._symbols_signature()

        def builder() -> Dict[str, Any]:
            bundle = agent._collect_prefetch_bundle(event.date, event.decision_time, event.decision_count)
            bundle.pop("observation_summary", None)
            return bundle

        result = agent.prefetch_coordinator.ensure_snapshot(
            today_date=event.date,
            current_time=event.decision_time,
            symbols_signature=symbols_signature,
            builder=builder,
        )
        _validate_snapshot_payload(agent, result.data)
        return {
            "path": result.path,
            "created": result.created,
            "snapshot_id": result.data.get("snapshot_id"),
            "symbols_signature": symbols_signature,
        }
    finally:
        if getattr(agent, "dm", None):
            try:
                agent.dm.close_ts_client(force=True)
            except Exception:
                pass


def _process_env(event: DecisionEvent, signature: str, config: Dict[str, Any], log_path: Path, live_backtest_mode: bool) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BACKTEST_MODE"] = "true" if live_backtest_mode else "false"
    env["SKIP_SNAPSHOT_PREPARE"] = "1"
    env["NEWS_ALREADY_PREFETCHED"] = "1"
    env["SNAPSHOTS_ALREADY_PREPARED"] = "1"
    env["ONLY_DECISION_COUNT"] = str(event.decision_count)
    env["ASTOCK_JOB_LOG_PATH"] = str(log_path)
    env["RUNTIME_ENV_PATH"] = str(RUNTIME_DIR / f"runtime_{event.date}_{event.decision_count}_{_safe_name(signature)}.json")
    if "snapshot_hourly_cache_days" in (config.get("run_config") or {}):
        env["SNAPSHOT_HOURLY_CACHE_DAYS"] = str(config["run_config"].get("snapshot_hourly_cache_days"))
    return env


def _run_model_process_attempt(
    config: Dict[str, Any],
    event: DecisionEvent,
    model: Dict[str, Any],
    *,
    live_backtest_mode: bool,
    attempt: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    signature = str(model.get("signature") or model.get("name") or "model")
    jobs_dir = STATE_DIR / event.date
    jobs_dir.mkdir(parents=True, exist_ok=True)
    tick_config = build_tick_config(config, event, model, live_backtest_mode=live_backtest_mode)
    tick_config_path = _write_tick_config(tick_config, event, signature)
    log_path = jobs_dir / f"{event.safe_time}_{_safe_name(signature)}_attempt{attempt}.log"
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "main.py"), str(tick_config_path)]
    env = _process_env(event, signature, config, log_path, live_backtest_mode)
    started_at = _now_cn().isoformat(timespec="seconds")
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            returncode = proc.wait(timeout=max(timeout_seconds, 1))
            timeout = False
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                returncode = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait(timeout=20)
            timeout = True
    result = {
        "signature": signature,
        "attempt": attempt,
        "pid": proc.pid,
        "returncode": returncode,
        "timeout": timeout,
        "log_path": str(log_path),
        "config_path": str(tick_config_path),
        "started_at": started_at,
        "finished_at": _now_cn().isoformat(timespec="seconds"),
        "cmd": cmd,
    }
    if timeout:
        raise TimeoutError(f"{signature} timed out after {timeout_seconds}s")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return result


def run_model_with_retries(
    config: Dict[str, Any],
    event: DecisionEvent,
    model: Dict[str, Any],
    *,
    live_backtest_mode: bool,
    timeout_seconds: int,
    snapshot_path: Optional[str],
) -> Dict[str, Any]:
    attempts = retry_attempts(config)
    signature = str(model.get("signature") or model.get("name") or "model")
    attempt_results: List[Dict[str, Any]] = []
    last_error: Optional[BaseException] = None
    last_log_path: Optional[Path] = None
    for attempt in range(1, attempts + 1):
        try:
            result = _run_model_process_attempt(
                config,
                event,
                model,
                live_backtest_mode=live_backtest_mode,
                attempt=attempt,
                timeout_seconds=timeout_seconds,
            )
            attempt_summary = dict(result)
            result["attempts"] = attempt
            result["status"] = "succeeded"
            result["attempt_results"] = attempt_results + [attempt_summary]
            return result
        except Exception as exc:
            last_error = exc
            maybe_log = STATE_DIR / event.date / f"{event.safe_time}_{_safe_name(signature)}_attempt{attempt}.log"
            last_log_path = maybe_log if maybe_log.exists() else last_log_path
            attempt_results.append({
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "log_path": str(maybe_log) if maybe_log.exists() else "",
            })
            print(f"model {signature} attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(_backoff_for_attempt(config, attempt))
    stdout_tail = _tail_text(last_log_path)
    error_dir = write_error_artifacts(
        config,
        event,
        stage="model_api",
        model_signature=signature,
        symbol="ALL",
        attempts=attempts,
        error=last_error,
        final_action="model_failed_continue_others",
        log_path=last_log_path,
        snapshot_path=snapshot_path,
        stdout_tail=stdout_tail,
        stderr_tail="",
        returncode=getattr(last_error, "returncode", None),
    )
    return {
        "signature": signature,
        "status": "failed",
        "attempts": attempts,
        "attempt_results": attempt_results,
        "error_dir": str(error_dir),
        "error_type": type(last_error).__name__ if last_error else "Error",
        "error_message": str(last_error) if last_error else "",
        "log_path": str(last_log_path) if last_log_path else "",
    }


def run_benchmark_for_event(config: Dict[str, Any], event: DecisionEvent) -> Dict[str, Any]:
    from benchmark import benchmark_evaluator

    output_dir = PROJECT_ROOT / "data_flow" / "benchmark_reports"
    report = benchmark_evaluator.run(output_dir=output_dir, archive_date=event.date)
    return {
        "status": "succeeded",
        "output_dir": str(output_dir),
        "model_count": report.get("model_count"),
        "snapshot_count": report.get("snapshot_count"),
    }


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _latest_agent_run(signature: str, event: DecisionEvent) -> Optional[Path]:
    run_dir = PROJECT_ROOT / "data_flow" / "trading_summary_each_agent" / signature / "runs" / event.date / event.clock_dir
    if run_dir.exists():
        return run_dir
    return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _position_records(signature: str) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "data_flow" / "trading_summary_each_agent" / signature / "position" / "position.jsonl"
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if isinstance(record, dict):
                records.append(record)
    except Exception:
        return []
    return records


def _latest_position_record(signature: str, event: DecisionEvent) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for record in _position_records(signature):
            if record.get("date") != event.date:
                continue
            if record.get("decision_time") != event.decision_time:
                continue
            if record.get("decision_count") not in (None, event.decision_count):
                continue
            selected = record
    return selected


def _snapshot_prices(event_state: Dict[str, Any]) -> Dict[str, float]:
    snapshot_info = event_state.get("snapshot") if isinstance(event_state, dict) else {}
    snapshot_path = ""
    if isinstance(snapshot_info, dict):
        snapshot_path = str(snapshot_info.get("path") or "")
    payload = _read_json(Path(snapshot_path), {}) if snapshot_path else {}
    raw_prices = payload.get("prices") if isinstance(payload, dict) else {}
    prices: Dict[str, float] = {}
    if not isinstance(raw_prices, dict):
        return prices
    for symbol, value in raw_prices.items():
        normalized = normalize_symbol(symbol)
        price: Optional[float] = None
        if isinstance(value, dict):
            for key in ("current_price", "price", "last_price", "close", "latest_price"):
                price = _float_or_none(value.get(key))
                if price is not None:
                    break
        else:
            price = _float_or_none(value)
        if normalized and price is not None:
            prices[normalized] = price
    return prices


def _position_equity_fields(position_record: Dict[str, Any], prices: Dict[str, float]) -> Dict[str, float]:
    positions = position_record.get("positions") if isinstance(position_record, dict) else {}
    if not isinstance(positions, dict):
        return {}
    cash = _float_or_none(positions.get("CASH")) or 0.0
    cost_value = 0.0
    market_value = 0.0
    for symbol, payload in positions.items():
        if symbol == "CASH" or not isinstance(payload, dict):
            continue
        shares = _float_or_none(payload.get("shares")) or 0.0
        avg_price = _float_or_none(payload.get("avg_price")) or 0.0
        normalized = normalize_symbol(symbol)
        mark_price = prices.get(normalized) if normalized else None
        if mark_price is None:
            mark_price = avg_price
        cost_value += shares * avg_price
        market_value += shares * mark_price
    return {
        "cash": round(cash, 4),
        "realized": round(cash + cost_value, 4),
        "unrealized": round(cash + market_value, 4),
    }


def _share_map(position_record: Dict[str, Any]) -> Dict[str, float]:
    positions = position_record.get("positions") if isinstance(position_record, dict) else {}
    if not isinstance(positions, dict):
        return {}
    shares: Dict[str, float] = {}
    for symbol, payload in positions.items():
        if symbol == "CASH" or not isinstance(payload, dict):
            continue
        normalized = normalize_symbol(symbol)
        amount = _float_or_none(payload.get("shares"))
        if normalized and amount is not None:
            shares[normalized] = amount
    return shares


def _actual_action_text(signature: str, event: DecisionEvent, selected: Dict[str, Any]) -> str:
    if not selected:
        return ""
    records = _position_records(signature)
    previous: Dict[str, Any] = {}
    for record in records:
        if record is selected:
            break
        if record.get("date") == selected.get("date") and record.get("decision_time") == selected.get("decision_time") and record.get("id") == selected.get("id"):
            break
        previous = record
    before = _share_map(previous)
    after = _share_map(selected)
    actions: List[str] = []
    for symbol in sorted(set(before) | set(after)):
        delta = (after.get(symbol) or 0.0) - (before.get(symbol) or 0.0)
        if abs(delta) < 1e-9:
            continue
        verb = "buy" if delta > 0 else "sell"
        amount = abs(delta)
        amount_text = str(int(amount)) if float(amount).is_integer() else str(round(amount, 4))
        actions.append(f"{verb} {symbol} {amount_text}")
    if actions:
        return "; ".join(actions)
    action = selected.get("this_action") if isinstance(selected, dict) else {}
    if isinstance(action, dict):
        verb = str(action.get("action") or "").strip()
        symbol = str(action.get("symbol") or "").strip()
        amount = action.get("amount")
        if verb and verb != "seed":
            return f"{verb} {symbol} {amount or ''}".strip()
    return "no_trade"


def write_daily_summary(config: Dict[str, Any], date_str: str, state: Dict[str, Any]) -> Path:
    out_dir = STATE_DIR / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[str] = [
        f"# Live Paper Summary | {date_str}",
        "",
        "This is a readable summary. Raw evidence is stored in each model run directory and scheduler logs.",
        "",
        "| Decision Time | Model | Status | Attempts | Action | Cash | Realized | Unrealized | Schema | Error |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    events = state.get("events", {}) if isinstance(state, dict) else {}
    for key, event_state in sorted(events.items()):
        if not isinstance(event_state, dict) or event_state.get("date") != date_str:
            continue
        decision_time = str(event_state.get("decision_time") or "")
        event = DecisionEvent(date_str, decision_time, int(event_state.get("decision_count") or 0), _now_cn())
        prices = _snapshot_prices(event_state)
        for signature, model_state in sorted((event_state.get("models") or {}).items()):
            attempts = model_state.get("attempts") or model_state.get("attempt") or ""
            status = model_state.get("status") or ("succeeded" if model_state.get("returncode") == 0 else event_state.get("status", ""))
            error = model_state.get("error_message") or model_state.get("error_dir") or ""
            action_text = ""
            cash = realized = unrealized = schema = ""
            run_dir = _latest_agent_run(signature, event)
            if run_dir:
                execution = _read_json(run_dir / "execution.json", {})
                input_payload = _read_json(run_dir / "input.json", {})
                actions = execution.get("actions_planned_or_taken") or []
                if actions:
                    action_text = "; ".join(
                        f"{a.get('action')} {a.get('symbol') or ''} {a.get('amount') or ''}".strip()
                        for a in actions if isinstance(a, dict)
                    )
                position_record = _latest_position_record(signature, event)
                actual_action = _actual_action_text(signature, event, position_record)
                if actual_action:
                    action_text = actual_action
                position_fields = _position_equity_fields(position_record, prices)
                cash = position_fields.get("cash", input_payload.get("cash", ""))
                realized = position_fields.get("realized", input_payload.get("realized_equity", ""))
                unrealized = position_fields.get("unrealized", input_payload.get("unrealized_equity", input_payload.get("total_equity", "")))
                schema = f"parse={execution.get('parse_success')}, schema={execution.get('schema_valid')}"
            rows.append(
                f"| {decision_time} | {signature} | {status} | {attempts} | "
                f"{str(action_text)[:160]} | {cash} | {realized} | {unrealized} | {schema} | {str(error)[:160]} |"
            )
    path = out_dir / "daily_summary.md"
    path.write_text("\n".join(rows).rstrip() + "\n", encoding="utf-8")
    return path


def run_event(config: Dict[str, Any], event: DecisionEvent, *, dry_run: bool = False) -> Dict[str, Any]:
    run_config = config.get("run_config", {})
    live_backtest_mode = _truthy(run_config.get("live_backtest_mode"), default=False)
    timeout_seconds = _int_config(config, "live_scheduler_model_timeout_seconds", 7200, min_value=1)
    abort_on_news_failure = _truthy(run_config.get("live_abort_on_news_prefetch_failure"), default=False)
    abort_on_snapshot_failure = _truthy(run_config.get("live_abort_on_snapshot_failure"), default=True)
    models = _enabled_models(config)
    max_model_workers = min(model_parallelism(config), max(len(models), 1))
    start_delay_seconds = model_start_delay_seconds(config) if max_model_workers > 1 else 0.0
    result: Dict[str, Any] = {
        "date": event.date,
        "decision_time": event.decision_time,
        "decision_count": event.decision_count,
        "started_at": _now_cn().isoformat(timespec="seconds"),
        "status": "running",
        "model_parallelism": max_model_workers,
        "model_start_delay_seconds": start_delay_seconds,
        "models": {},
        "error_dirs": [],
    }
    _write_heartbeat(config, status="running_event", event=event, current_processes=[], state={"events": {event.key: result}})

    if not models:
        error_dir = write_error_artifacts(
            config,
            event,
            stage="scheduler",
            attempts=1,
            error_type="NoEnabledModels",
            error_message="No enabled models configured",
            final_action="event_failed",
        )
        result.update({"status": "failed", "error": "no_enabled_models", "finished_at": _now_cn().isoformat(timespec="seconds")})
        result["error_dirs"].append(str(error_dir))
        return result

    if dry_run:
        result["status"] = "succeeded"
        result["dry_run"] = True
        result["news_prefetch_ok"] = None
        result["snapshot"] = {"dry_run": True}
        for model in models:
            signature = str(model.get("signature") or model.get("name") or "model")
            result["models"][signature] = {"status": "succeeded", "returncode": 0, "dry_run": True, "attempts": 0}
        result["finished_at"] = _now_cn().isoformat(timespec="seconds")
        return result

    news_ok, _, news_error_dir = _run_with_retries(
        config,
        event,
        stage="news_prefetch",
        symbol="ALL",
        final_action_on_failure="continue_with_existing_news_cache" if not abort_on_news_failure else "event_failed",
        func=lambda _attempt: prefetch_news_for_event(config, event),
    )
    result["news_prefetch_ok"] = news_ok
    if news_error_dir:
        result["error_dirs"].append(str(news_error_dir))
    if not news_ok and abort_on_news_failure:
        result.update({"status": "failed", "error": "news_prefetch_failed", "finished_at": _now_cn().isoformat(timespec="seconds")})
        return result

    snapshot_ok, snapshot_info, snapshot_error_dir = _run_with_retries(
        config,
        event,
        stage="snapshot_prepare",
        symbol="ALL",
        final_action_on_failure="event_failed" if abort_on_snapshot_failure else "continue_without_prepared_snapshot",
        func=lambda _attempt: prepare_snapshot_for_event(config, event),
    )
    result["snapshot"] = snapshot_info
    snapshot_path = str((snapshot_info or {}).get("path") or "")
    if snapshot_error_dir:
        result["error_dirs"].append(str(snapshot_error_dir))
    if not snapshot_ok and abort_on_snapshot_failure:
        result.update({"status": "failed", "error": "snapshot_prepare_failed", "finished_at": _now_cn().isoformat(timespec="seconds")})
        return result

    all_models_ok = True

    def record_model_result(signature: str, model_result: Dict[str, Any]) -> None:
        nonlocal all_models_ok
        result["models"][signature] = model_result
        if model_result.get("status") == "failed":
            all_models_ok = False
            if model_result.get("error_dir"):
                result["error_dirs"].append(str(model_result["error_dir"]))
        _write_heartbeat(
            config,
            status="running_event",
            event=event,
            current_processes=[
                {"signature": sig, **payload}
                for sig, payload in result.get("models", {}).items()
                if isinstance(payload, dict)
            ],
            state={"events": {event.key: result}},
            last_error_dir=Path(result["error_dirs"][-1]) if result.get("error_dirs") else None,
        )

    def run_one_model(model: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        signature = str(model.get("signature") or model.get("name") or "model")
        try:
            model_result = run_model_with_retries(
                config,
                event,
                model,
                live_backtest_mode=live_backtest_mode,
                timeout_seconds=timeout_seconds,
                snapshot_path=snapshot_path,
            )
        except Exception as exc:
            error_dir = write_error_artifacts(
                config,
                event,
                stage="model_api",
                model_signature=signature,
                symbol="ALL",
                attempts=1,
                error=exc,
                final_action="model_failed_continue_others",
                snapshot_path=snapshot_path,
            )
            model_result = {
                "signature": signature,
                "status": "failed",
                "attempts": 1,
                "error_dir": str(error_dir),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        return signature, model_result

    if max_model_workers <= 1:
        for model in models:
            signature, model_result = run_one_model(model)
            record_model_result(signature, model_result)
    else:
        print(
            f"Running {len(models)} model worker(s) with "
            f"parallelism={max_model_workers}, start_delay={start_delay_seconds:.1f}s",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_model_workers) as executor:
            futures = {}
            for model_idx, model in enumerate(models):
                future = executor.submit(run_one_model, model)
                futures[future] = model
                if start_delay_seconds > 0 and model_idx < len(models) - 1:
                    time.sleep(start_delay_seconds)
            for future in as_completed(futures):
                model = futures[future]
                fallback_signature = str(model.get("signature") or model.get("name") or "model")
                try:
                    signature, model_result = future.result()
                except Exception as exc:
                    error_dir = write_error_artifacts(
                        config,
                        event,
                        stage="model_api",
                        model_signature=fallback_signature,
                        symbol="ALL",
                        attempts=1,
                        error=exc,
                        final_action="model_failed_continue_others",
                        snapshot_path=snapshot_path,
                    )
                    signature = fallback_signature
                    model_result = {
                        "signature": signature,
                        "status": "failed",
                        "attempts": 1,
                        "error_dir": str(error_dir),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                record_model_result(signature, model_result)

    benchmark_ok, benchmark_result, benchmark_error_dir = _run_with_retries(
        config,
        event,
        stage="benchmark_summary",
        symbol="ALL",
        final_action_on_failure="continue_next_event",
        func=lambda _attempt: run_benchmark_for_event(config, event),
    )
    result["benchmark"] = benchmark_result if benchmark_ok else {"status": "failed"}
    if benchmark_error_dir:
        result["error_dirs"].append(str(benchmark_error_dir))

    result["status"] = "succeeded" if all_models_ok else "partial_failed"
    result["finished_at"] = _now_cn().isoformat(timespec="seconds")
    return result


def _record_result(config: Dict[str, Any], state: Dict[str, Any], event: DecisionEvent, result: Dict[str, Any]) -> None:
    state.setdefault("events", {})[event.key] = result
    _save_state(state)
    write_daily_summary(config, event.date, state)
    _write_heartbeat(
        config,
        status=result.get("status", "unknown"),
        event=event,
        state=state,
        last_error_dir=Path(result["error_dirs"][-1]) if result.get("error_dirs") else None,
    )


def run_doctor(config_path: Optional[str] = None) -> int:
    config = trading_main.load_config(config_path)
    checks: List[Tuple[str, bool, str]] = []

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    checks.append((".venv", venv_python.exists(), str(venv_python)))
    for module_name in ("pandas", "requests", "langchain", "langgraph", "tushare"):
        checks.append((f"python module {module_name}", importlib.util.find_spec(module_name) is not None, module_name))
    enabled = _enabled_models(config)
    checks.append(("enabled models", bool(enabled), ", ".join(str(m.get("signature")) for m in enabled)))
    needs_openrouter = any("openrouter.ai" in str(m.get("openai_base_url", "")) for m in enabled)
    checks.append(("OPENROUTER_API_KEY", (not needs_openrouter) or bool(os.getenv("OPENROUTER_API_KEY")), "required for OpenRouter models"))
    checks.append(("TinySoft credentials", bool(os.getenv("TSL_USER") or os.getenv("TSL_USERNAME")), "TSL_USER/TSL_USERNAME"))
    checks.append(("timezone Asia/Shanghai", _now_cn().tzinfo is None, _now_cn().isoformat(timespec="seconds")))
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        checks.append(("disk free > 2GB", usage.free > 2 * 1024**3, f"free={usage.free}"))
    except Exception as exc:
        checks.append(("disk free > 2GB", False, str(exc)))
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        probe = STATE_DIR / f".write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(("write permission", True, str(STATE_DIR)))
    except Exception as exc:
        checks.append(("write permission", False, str(exc)))
    news_path = Path((config.get("data_config") or {}).get("news_csv_path", "./data_flow/news.csv"))
    if not news_path.is_absolute():
        news_path = PROJECT_ROOT / news_path
    checks.append(("news.csv exists", news_path.exists(), str(news_path)))
    try:
        news_integrity = validate_news_cache_integrity(news_path, strict=False)
        detail = (
            f"rows={news_integrity.get('rows')} size={news_integrity.get('size_bytes')} "
            f"manifest={news_integrity.get('manifest_exists')}"
        )
        if news_integrity.get("errors"):
            detail += f" errors={news_integrity.get('errors')}"
        checks.append(("news.csv manifest integrity", bool(news_integrity.get("ok")), detail))
    except NewsCacheIntegrityError as exc:
        checks.append(("news.csv manifest integrity", False, str(exc)))
    next_event = next_pending_event(config, {"events": {}}, _now_cn())
    checks.append(("next event", next_event is not None, next_event.decision_time if next_event else "none"))

    ok = True
    print("Live scheduler doctor:")
    for name, passed, detail in checks:
        ok = ok and passed
        mark = "OK" if passed else "FAIL"
        print(f"- [{mark}] {name}: {detail}")
    return 0 if ok else 1


def _parse_event_arg(config: Dict[str, Any], value: str) -> DecisionEvent:
    target = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    clock = target.strftime("%H:%M:%S")
    decision_times = _decision_times(config)
    if clock not in decision_times:
        raise ValueError(f"time must be one of {decision_times}")
    return _event_for(target.strftime("%Y-%m-%d"), clock, decision_times.index(clock) + 1)


def _run_scheduler_locked(
    config_path: Optional[str],
    *,
    once: bool = False,
    dry_run: bool = False,
    run_now: Optional[str] = None,
    force_event: Optional[str] = None,
) -> int:
    config = trading_main.load_config(config_path)
    state = _load_state()
    poll_seconds = max(5, _int_config(config, "live_scheduler_poll_seconds", 60))

    if run_now or force_event:
        event = _parse_event_arg(config, run_now or force_event or "")
        result = run_event(config, event, dry_run=dry_run)
        if force_event:
            result["forced"] = True
        _record_result(config, state, event, result)
        return 0 if result.get("status") in ("succeeded", "partial_failed") else 1

    while True:
        now = _now_cn()
        if mark_missed_events(config, state, now):
            _save_state(state)

        event = next_pending_event(config, state, now)
        if event is None:
            print("No pending decision event found in the next 35 days.", flush=True)
            _write_heartbeat(config, status="no_pending_event", state=state)
            return 1

        if now < event.target_dt:
            wait_seconds = min(poll_seconds, max(1, int((event.target_dt - now).total_seconds())))
            print(f"Waiting for {event.decision_time}; next check in {wait_seconds}s", flush=True)
            _write_heartbeat(config, status="waiting", event=event, state=state)
            if once:
                return 0
            time.sleep(wait_seconds)
            continue

        state.setdefault("events", {})[event.key] = {
            **state.get("events", {}).get(event.key, {}),
            "date": event.date,
            "decision_time": event.decision_time,
            "decision_count": event.decision_count,
            "status": "running",
            "started_at": _now_cn().isoformat(timespec="seconds"),
        }
        _save_state(state)
        result = run_event(config, event, dry_run=dry_run)
        _record_result(config, state, event, result)
        if once:
            return 0 if result.get("status") in ("succeeded", "partial_failed") else 1


def run_scheduler(
    config_path: Optional[str],
    *,
    once: bool = False,
    dry_run: bool = False,
    run_now: Optional[str] = None,
    force_event: Optional[str] = None,
) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / "scheduler.lock"
    lock = _FileLock(str(lock_path), timeout=1.0)
    if not lock.acquire(timeout=1.0):
        stale_after = _float_config(trading_main.load_config(config_path), "live_scheduler_stale_lock_seconds", 86400.0)
        try:
            age = time.time() - lock_path.stat().st_mtime
            suffix = " The lock file looks stale." if age > stale_after else ""
        except Exception:
            suffix = ""
        print(f"Another live scheduler instance is already running.{suffix}", flush=True)
        return 2
    try:
        lock_path.write_text(
            json.dumps({"pid": os.getpid(), "started_at": _now_cn().isoformat(timespec="seconds")}, ensure_ascii=False),
            encoding="utf-8",
        )
        return _run_scheduler_locked(config_path, once=once, dry_run=dry_run, run_now=run_now, force_event=force_event)
    finally:
        lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live paper trading scheduler.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to settings JSON; defaults to settings/default_config.json")
    parser.add_argument("--doctor", action="store_true", help="Check server readiness and exit")
    parser.add_argument("--once", action="store_true", help="Run one due event, or print the next wait target and exit")
    parser.add_argument("--dry-run", action="store_true", help="Exercise scheduling without network/model subprocesses")
    parser.add_argument("--run-now", help="Immediately run a configured decision time, e.g. '2026-06-18 10:30:00'")
    parser.add_argument("--force-event", help="Run a configured decision time even if state says it already succeeded")
    args = parser.parse_args()
    if args.doctor:
        raise SystemExit(run_doctor(args.config_path))
    raise SystemExit(
        run_scheduler(
            args.config_path,
            once=args.once,
            dry_run=args.dry_run,
            run_now=args.run_now,
            force_event=args.force_event,
        )
    )


if __name__ == "__main__":
    main()
