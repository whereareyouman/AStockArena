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

from tools.json_file_manager import JsonFileManager

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

    def __init__(self, path: str):
        self.path = path
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
        if not self.acquire():
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
        default_dir = project_root / "data" / "agent_data" / "shared"
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
        return f"{today_date}_{sanitized_time}_{symbols_signature}"

    def _snapshot_path(self, today_date: str, current_time: str, symbols_signature: str) -> Path:
        sanitized_time = current_time.replace(":", "-").replace(" ", "_")
        return self.snapshots_dir / today_date / f"{sanitized_time}_{symbols_signature}.json"

    def _lock_path(self, key: str) -> Path:
        return self.lock_dir / f"{key}.lock"

    def _is_fresh(self, payload: Dict[str, Any], today_date: str, current_time: str, symbols_signature: str) -> bool:
        if not payload:
            return False
        if payload.get("today_date") != today_date:
            return False
        if payload.get("decision_time") != current_time:
            return False
        if payload.get("symbols_signature") != symbols_signature:
            return False
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
        snapshot_path = self._snapshot_path(today_date, current_time, symbols_signature)
        if not snapshot_path.exists():
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
        with _FileLock(str(lock_path)):
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

