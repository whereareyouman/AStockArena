"""
Shared prefetch coordinator to avoid duplicated AkShare/TinySoft calls
when multiple agents (LLMs) need the same market snapshot.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from utils.json_file_manager import JsonFileManager

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


@dataclass
class SnapshotResult:
    data: Dict[str, Any]
    path: str
    created: bool


class _FileLock:
    """
    Simple cross-platform file lock (best-effort, mirrors json_file_manager approach).
    """

    def __init__(self, path: str, timeout: float = 15.0):
        self.path = path
        self.timeout = timeout
        self._handle = None

    def acquire(self, timeout: float = 15.0) -> bool:
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
                return True  # fallback: no real lock
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
            self._handle.close()
            self._handle = None

    def __enter__(self):
        if not self.acquire(timeout=self.timeout):
            raise TimeoutError(f"Timed out acquiring lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class SharedPrefetchCoordinator:
    """
    Coordinates shared market snapshots across multiple agents.
    """

    SNAPSHOT_SCHEMA_VERSION = 1

    def __init__(
        self,
        base_dir: Optional[str] = None,
        ttl_seconds: int = 600,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_dir = project_root / "data_flow" / "trading_summary_each_agent" / "shared"
        self.base_dir = Path(base_dir) if base_dir else default_dir
        self.snapshots_dir = self.base_dir / "snapshots"
        self.logs_dir = self.base_dir / "logs"
        self.lock_dir = self.base_dir / "locks"
        self.ttl_seconds = ttl_seconds
        self.json_manager = JsonFileManager()
        for path in (self.snapshots_dir, self.logs_dir, self.lock_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _log_snapshot(self, snapshot: Dict[str, Any], snapshot_path: Path, created: bool) -> None:
        try:
            today = snapshot.get("today_date") or "unknown"
            decision_time = snapshot.get("decision_time") or "unknown"
            sanitized_time = decision_time.replace(":", "-").replace(" ", "_")
            log_dir = self.logs_dir / str(today)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{sanitized_time}.jsonl"

            news_payload = snapshot.get("news") or {}
            price_payloads = snapshot.get("prices") or {}
            indicator_payloads = snapshot.get("indicators") or {}
            combined_symbols = sorted(
                set(news_payload.keys()) | set(price_payloads.keys()) | set(indicator_payloads.keys())
            )
            symbol_rows = []
            for symbol in combined_symbols:
                price_summary = {}
                price_entry = price_payloads.get(symbol)
                if isinstance(price_entry, dict):
                    price_summary = price_entry.get("summary") or {}
                indicators_entry = indicator_payloads.get(symbol) or {}
                indicators_payload = indicators_entry.get("indicators") if isinstance(indicators_entry, dict) else None
                symbol_rows.append(
                    {
                        "symbol": symbol,
                        "close": price_summary.get("close"),
                        "change_pct": price_summary.get("change_pct"),
                        "news_count": len((news_payload.get(symbol) or {}).get("news", [])),
                        "has_indicators": bool(indicators_payload),
                    }
                )

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": snapshot.get("snapshot_id"),
                "snapshot_path": str(snapshot_path),
                "created": created,
                "symbols": symbol_rows,
                "observation_preview": (snapshot.get("observation_summary") or "")[:800],
            }
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as err:  # pragma: no cover - logging failure shouldn't break trading
            print(f"⚠️ Failed to log shared snapshot: {err}")

    def _decision_key(self, today_date: str, current_time: str, symbols_signature: str) -> str:
        sanitized_time = current_time.replace(":", "-").replace(" ", "_")
        # Windows 文件名字符清理：替换无效字符（|, <, >, :, ", /, \, ?, *）为下划线
        safe_signature = symbols_signature.replace("|", "_").replace("<", "_").replace(">", "_").replace(":", "_").replace('"', "_").replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_")
        return f"{today_date}_{sanitized_time}_{safe_signature}"

    def _snapshot_path(self, today_date: str, current_time: str, symbols_signature: str) -> Path:
        sanitized_time = current_time.replace(":", "-").replace(" ", "_")
        # Windows 文件名字符清理：替换无效字符为下划线
        safe_signature = symbols_signature.replace("|", "_").replace("<", "_").replace(">", "_").replace(":", "_").replace('"', "_").replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_")
        return self.snapshots_dir / today_date / f"{sanitized_time}_{safe_signature}.json"

    def _lock_path(self, key: str) -> Path:
        # 确保 key 中不包含 Windows 无效字符
        safe_key = key.replace("|", "_").replace("<", "_").replace(">", "_").replace(":", "_").replace('"', "_").replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_")
        return self.lock_dir / f"{safe_key}.lock"

    def _is_fresh(self, payload: Dict[str, Any], today_date: str, current_time: str, symbols_signature: str) -> bool:
        if not payload:
            return False
        if payload.get("today_date") != today_date:
            return False
        if payload.get("decision_time") != current_time:
            return False
        if payload.get("symbols_signature") != symbols_signature:
            return False
        
        # 对于历史日期（过去的日期），不检查TTL，直接认为fresh，允许复用已有snapshots
        try:
            snapshot_date = datetime.fromisoformat(today_date).date()
            current_date = datetime.now(timezone.utc).date()
            is_historical_date = snapshot_date < current_date
            if is_historical_date:
                # 历史日期：不检查TTL，直接复用
                return True
        except Exception:
            # 如果日期解析失败，继续检查TTL
            pass
        
        # 对于今天或未来的日期，检查TTL
        generated_at = payload.get("generated_at")
        if not generated_at or not self.ttl_seconds:
            return True
        try:
            ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
            return age.total_seconds() <= self.ttl_seconds
        except Exception:
            return True

    def _load_snapshot(
        self,
        today_date: str,
        current_time: str,
        symbols_signature: str,
    ) -> Optional[SnapshotResult]:
        # 首先尝试使用替换后的文件名（新格式）
        snapshot_path = self._snapshot_path(today_date, current_time, symbols_signature)
        if not snapshot_path.exists():
            # 如果新格式不存在，尝试旧格式（使用原始 | 分隔符，兼容旧文件）
            sanitized_time = current_time.replace(":", "-").replace(" ", "_")
            old_format_path = self.snapshots_dir / today_date / f"{sanitized_time}_{symbols_signature}.json"
            if old_format_path.exists():
                snapshot_path = old_format_path
            else:
                return None
        
        payload = self.json_manager.safe_read_json(str(snapshot_path), default={})
        if not self._is_fresh(payload, today_date, current_time, symbols_signature):
            return None
        return SnapshotResult(payload, str(snapshot_path), created=False)

    def ensure_snapshot(
        self,
        today_date: str,
        current_time: str,
        symbols_signature: str,
        builder: Callable[[], Dict[str, Any]],
    ) -> SnapshotResult:
        """
        Load a fresh snapshot if it exists; otherwise build one using the builder callable.
        Returns the snapshot payload and whether it was freshly created.
        """
        existing = self._load_snapshot(today_date, current_time, symbols_signature)
        if existing:
            return existing

        key = self._decision_key(today_date, current_time, symbols_signature)
        lock_path = self._lock_path(key)
        # 默认：无限等待锁，确保所有进程必须复用 shared snapshot（不允许 15s 超时后 fallback）。
        lock_timeout_raw = os.getenv("SHARED_PREFETCH_LOCK_TIMEOUT")
        lock_timeout = 0.0
        if lock_timeout_raw is not None and str(lock_timeout_raw).strip() != "":
            try:
                lock_timeout = float(lock_timeout_raw)
            except Exception:
                lock_timeout = 0.0
        # 约定：<=0 表示无限等待（不超时）
        if lock_timeout <= 0:
            lock_timeout = 0.0

        with _FileLock(str(lock_path), timeout=lock_timeout):
            # Re-check after acquiring the lock
            existing_after_lock = self._load_snapshot(today_date, current_time, symbols_signature)
            if existing_after_lock:
                return existing_after_lock

            snapshot = builder()
            if not isinstance(snapshot, dict):
                raise ValueError("Snapshot builder must return a dict")
            snapshot.setdefault("schema_version", self.SNAPSHOT_SCHEMA_VERSION)
            snapshot.setdefault("symbols_signature", symbols_signature)
            snapshot.setdefault("today_date", today_date)
            snapshot.setdefault("decision_time", current_time)
            snapshot.setdefault(
                "generated_at",
                datetime.now(timezone.utc).isoformat(),
            )
            snapshot_path = self._snapshot_path(today_date, current_time, symbols_signature)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.json_manager.safe_write_json(str(snapshot_path), snapshot, backup=False)
            self._log_snapshot(snapshot, snapshot_path, created=True)
            return SnapshotResult(snapshot, str(snapshot_path), created=True)

