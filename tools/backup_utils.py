"""
Helper utilities to trigger the backup_data.py script from Python entrypoints.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = PROJECT_ROOT / "scripts" / "backup_data.py"


def run_backup_snapshot(reason: str = "manual", retain: Optional[int] = None) -> bool:
    """
    Execute the backup_data.py script.

    Args:
        reason: Human-readable context, stored in BACKUP_REASON env var for logs.
        retain: Optional override for --retain parameter.

    Returns:
        True if the backup script completed successfully, False otherwise.
    """
    if not BACKUP_SCRIPT.exists():
        print(f"⚠️ Backup script not found at {BACKUP_SCRIPT}")
        return False

    cmd = [sys.executable, str(BACKUP_SCRIPT)]
    if retain is not None:
        cmd.extend(["--retain", str(retain)])

    env = os.environ.copy()
    env["BACKUP_REASON"] = reason

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print("📦 Backup snapshot completed.")
        if completed.stdout:
            print(completed.stdout.strip())
        if completed.stderr:
            print(completed.stderr.strip())
        return True
    except subprocess.CalledProcessError as exc:
        print(f"⚠️ Backup script exited with non-zero status ({exc.returncode}).")
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())
        return False
    except Exception as exc:
        print(f"⚠️ Failed to run backup script: {exc}")
        return False

def save_pnl_snapshot(reason: str) -> bool:
    """Save PnL/equity curve snapshot to a separate file."""
    try:
        from api_server import _read_jsonl_tail, _load_initial_cash, _estimate_equity_for_positions
        
        pos_file = Path(PROJECT_ROOT) / "data" / "agent_data" / "default" / "position.jsonl"
        all_items = _read_jsonl_tail(pos_file, limit=100000)
        
        if not all_items:
            return False
    
        by_date = {}
        for it in all_items:
            d = it.get("date")
            if not d:
                continue
            prev = by_date.get(d)
            if prev is None or (it.get("id", -1) > prev.get("id", -1)):
                by_date[d] = it
        
        dates_sorted = sorted(by_date.keys())
        initial_cash = _load_initial_cash()
        
        pnl_data = []
        for d in dates_sorted:
            rec = by_date[d]
            _, equity_val, _, _ = _estimate_equity_for_positions(
                rec.get("positions", {}) or {}, rec.get("decision_time"), rec.get("date")
            )
            ret_pct = (equity_val / initial_cash - 1.0) * 100.0 if initial_cash > 0 else 0.0
            pnl_data.append({
                "date": d,
                "returnPct": ret_pct,
                "equity": equity_val,
                "id": rec.get("id"),
            })

        pnl_dir = Path(PROJECT_ROOT) / "data" / "pnl_snapshots"
        pnl_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pnl_file = pnl_dir / f"pnl_{reason}_{timestamp}.json"
        
        with open(pnl_file, 'w') as f:
            json.dump(pnl_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"⚠️ Error saving PnL snapshot: {e}")
        return False

