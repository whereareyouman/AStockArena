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


