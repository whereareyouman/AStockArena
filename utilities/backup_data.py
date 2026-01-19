#!/usr/bin/env python3
"""
Create a timestamped backup of all critical trading data_flow.

Backed-up assets:
  - data_flow/trading_summary_each_agent/**        (all model logs and positions)
  - data_flow/news.csv             (latest news cache)
  - data_flow/ai_stock_data.json   (hourly price cache)
  - settings/runtime/*.json    (per-model runtime state)
  - settings/*.json            (agentic workflow configuration)

The script produces a compressed tarball plus a manifest with SHA-256 hashes
so corrupted backups can be detected quickly. Old backups are rotated
according to the --retain flag (default: keep 5 most recent snapshots).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "backups"


def _iter_files(paths: Iterable[Path]) -> List[Path]:
    """Expand directories into individual files."""
    files: List[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    files.append(child)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for file in files:
        rel = file.relative_to(PROJECT_ROOT)
        if rel not in seen:
            unique.append(file)
            seen.add(rel)
    return unique


def _collect_targets() -> Tuple[List[Path], List[Path]]:
    """Return (files, missing_paths)."""
    include_dirs = [
        PROJECT_ROOT / "data_flow" / "trading_summary_each_agent",
        PROJECT_ROOT / "settings",
    ]
    include_files = [
        PROJECT_ROOT / "data_flow" / "news.csv",
        PROJECT_ROOT / "data_flow" / "ai_stock_data.json",
    ]

    # Runtime env files (runtime_env.json + runtime_env_*.json)
    runtime_dir = PROJECT_ROOT / "settings" / "runtime"
    if runtime_dir.exists():
        runtime_files = list(runtime_dir.glob("runtime_env*.json"))
        include_files.extend(runtime_files)

    missing: List[Path] = []
    resolved: List[Path] = []

    for path in include_dirs + include_files:
        if path.exists():
            resolved.append(path)
        else:
            missing.append(path)

    files = _iter_files(resolved)
    return files, missing


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(target_dir: Path, archive_name: str, files: List[Path]) -> None:
    entries = []
    for file in files:
        rel = file.relative_to(PROJECT_ROOT)
        entries.append({
            "path": str(rel),
            "size": file.stat().st_size,
            "sha256": _sha256(file),
        })

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_name,
        "total_files": len(entries),
        "source_root": str(PROJECT_ROOT),
        "files": entries,
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _create_archive(target_dir: Path, files: List[Path]) -> Path:
    archive_path = target_dir / "backup.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file in files:
            arcname = file.relative_to(PROJECT_ROOT)
            tar.add(file, arcname=str(arcname))
    return archive_path


def _enforce_retention(backup_dir: Path, retain: int) -> None:
    if retain <= 0 or not backup_dir.exists():
        return
    entries = sorted(
        [d for d in backup_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    excess = len(entries) - retain
    for entry in entries[:max(0, excess)]:
        shutil.rmtree(entry, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data_flow backups.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BACKUP_DIR,
        help="Directory to store backups (default: %(default)s)",
    )
    parser.add_argument(
        "--retain",
        type=int,
        default=5,
        help="Maximum number of backup snapshots to keep (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be backed up without creating archives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files, missing = _collect_targets()

    if not files:
        try:
            print("[WARNING] No files found to back up. Nothing to do.")
        except UnicodeEncodeError:
            print("WARNING: No files found to back up. Nothing to do.")
        return 0

    if args.dry_run:
        try:
            print("[DRY RUN] The following files would be included:")
        except UnicodeEncodeError:
            print("DRY RUN: The following files would be included:")
        for file in files:
            print(f" - {file.relative_to(PROJECT_ROOT)}")
        if missing:
            print("\nMissing (skipped):")
            for path in missing:
                print(f" - {path.relative_to(PROJECT_ROOT)}")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_dir = args.output_dir / timestamp
    if target_dir.exists():
        # Another process may be writing the same timestamp. Retry a few times
        # before creating a suffixed directory.
        import time

        MAX_RETRIES = 5
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[WAIT] Backup directory {target_dir} exists. Waiting before retry {attempt}/{MAX_RETRIES}...")
            except UnicodeEncodeError:
                print(f"WAIT: Backup directory {target_dir} exists. Waiting before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(attempt)  # incremental backoff
            if not target_dir.exists():
                break
        else:
            suffix = datetime.now(timezone.utc).strftime("%f")
            target_dir = args.output_dir / f"{timestamp}_{suffix}"
            try:
                print(f"[WARNING] Using alternative backup directory: {target_dir}")
            except UnicodeEncodeError:
                print(f"WARNING: Using alternative backup directory: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=False)

    archive_path = _create_archive(target_dir, files)
    _write_manifest(target_dir, archive_path.name, files)

    # Windows GBK 编码兼容：使用 ASCII 字符或 UTF-8 安全处理
    try:
        print(f"[OK] Backup created at {archive_path}")
    except UnicodeEncodeError:
        # Fallback for Windows GBK console
        print(f"Backup created at {archive_path}")
    
    if missing:
        try:
            print("[WARNING] Some paths were missing and skipped:")
        except UnicodeEncodeError:
            print("WARNING: Some paths were missing and skipped:")
        for path in missing:
            print(f"   - {path.relative_to(PROJECT_ROOT)}")

    _enforce_retention(args.output_dir, args.retain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


