#!/usr/bin/env python3
"""
Clean up future date data from data_flow directory.

This script removes all data with dates after 2026-01-15, including:
- Date-organized folders (log, snapshots, logs)
- Lock files with future dates in filenames
- Future date records in position.jsonl files
- Future date records in pnl_snapshots JSON files

Before cleanup, automatically creates a backup using backup_data.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFF_DATE = "2026-01-15"
DATA_FLOW_DIR = PROJECT_ROOT / "data_flow"
BACKUP_SCRIPT = PROJECT_ROOT / "utilities" / "backup_data.py"


def create_backup() -> bool:
    """Create a backup before cleanup."""
    print(f"[INFO] Creating backup before cleanup...")
    if not BACKUP_SCRIPT.exists():
        print(f"[WARNING] Backup script not found at {BACKUP_SCRIPT}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(BACKUP_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[OK] Backup created successfully")
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Backup failed: {e}")
        if e.stdout:
            print(e.stdout.strip())
        if e.stderr:
            print(e.stderr.strip())
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run backup: {e}")
        return False


def is_future_date(date_str: str) -> bool:
    """Check if date string is after cutoff date."""
    if not date_str or len(date_str) < 10:
        return False
    try:
        # Extract YYYY-MM-DD part if longer
        date_part = date_str[:10]
        return date_part > CUTOFF_DATE
    except Exception:
        return False


def cleanup_date_folders(base_dir: Path, subdir: str, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Clean up date-organized folders.
    
    Args:
        base_dir: Base directory (e.g., agent_data or trading_summary_each_agent)
        subdir: Subdirectory name (log, snapshots, or logs)
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Tuple of (deleted_count, deleted_paths)
    """
    deleted_count = 0
    deleted_paths = []
    
    # Handle model-specific log folders
    if subdir == "log":
        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "shared":
                continue
            log_dir = model_dir / "log"
            if not log_dir.exists():
                continue
            
            for date_folder in log_dir.iterdir():
                if date_folder.is_dir() and is_future_date(date_folder.name):
                    deleted_count += 1
                    deleted_paths.append(str(date_folder.relative_to(PROJECT_ROOT)))
                    if dry_run:
                        print(f"[DELETE] Would remove folder: {date_folder.relative_to(PROJECT_ROOT)}")
                    else:
                        try:
                            shutil.rmtree(date_folder)
                            print(f"[DELETE] Removed folder: {date_folder.relative_to(PROJECT_ROOT)}")
                        except Exception as e:
                            print(f"[WARNING] Failed to delete {date_folder}: {e}")
    
    # Handle shared folders (snapshots, logs)
    shared_dir = base_dir / "shared" / subdir
    if shared_dir.exists():
        for date_folder in shared_dir.iterdir():
            if date_folder.is_dir() and is_future_date(date_folder.name):
                deleted_count += 1
                deleted_paths.append(str(date_folder.relative_to(PROJECT_ROOT)))
                if dry_run:
                    print(f"[DELETE] Would remove folder: {date_folder.relative_to(PROJECT_ROOT)}")
                else:
                    try:
                        shutil.rmtree(date_folder)
                        print(f"[DELETE] Removed folder: {date_folder.relative_to(PROJECT_ROOT)}")
                    except Exception as e:
                        print(f"[WARNING] Failed to delete {date_folder}: {e}")
    
    return deleted_count, deleted_paths


def cleanup_lock_files(locks_dir: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Clean up lock files with future dates in filenames.
    
    Lock file format: {date}_{time}_{symbols}.lock or prefetch_{date}_{time}_{symbols}.lock
    
    Args:
        locks_dir: Directory containing lock files
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Tuple of (deleted_count, deleted_paths)
    """
    deleted_count = 0
    deleted_paths = []
    
    if not locks_dir.exists():
        return deleted_count, deleted_paths
    
    for lock_file in locks_dir.glob("*.lock"):
        filename = lock_file.name
        # Extract date from filename
        # Format: YYYY-MM-DD_HH-MM-SS_... or prefetch_YYYY-MM-DD_HH-MM-SS_...
        parts = filename.split("_")
        date_str = None
        
        # Try to find date pattern YYYY-MM-DD
        for i, part in enumerate(parts):
            if len(part) == 10 and part.count("-") == 2:
                try:
                    datetime.strptime(part, "%Y-%m-%d")
                    date_str = part
                    break
                except ValueError:
                    continue
        
        if date_str and is_future_date(date_str):
            deleted_count += 1
            deleted_paths.append(str(lock_file.relative_to(PROJECT_ROOT)))
            if dry_run:
                print(f"[DELETE] Would remove lock file: {lock_file.relative_to(PROJECT_ROOT)}")
            else:
                try:
                    lock_file.unlink()
                    print(f"[DELETE] Removed lock file: {lock_file.relative_to(PROJECT_ROOT)}")
                except Exception as e:
                    print(f"[WARNING] Failed to delete {lock_file}: {e}")
    
    return deleted_count, deleted_paths


def truncate_position_file(position_file: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Truncate position.jsonl file to only keep records with date <= CUTOFF_DATE.
    
    Args:
        position_file: Path to position.jsonl file
        dry_run: If True, only show what would be removed without actually modifying the file
    
    Returns:
        Tuple of (original_count, kept_count)
    """
    if not position_file.exists():
        return 0, 0
    
    original_count = 0
    kept_count = 0
    kept_lines = []
    removed_dates = []
    
    try:
        with open(position_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                original_count += 1
                try:
                    record = json.loads(line)
                    date_str = record.get("date", "")
                    if not is_future_date(date_str):
                        kept_lines.append(line)
                        kept_count += 1
                    else:
                        removed_dates.append(date_str)
                        if dry_run:
                            print(f"[TRUNCATE] Would remove record from {position_file.name}: date={date_str}")
                except json.JSONDecodeError:
                    # Keep malformed lines to avoid data loss
                    kept_lines.append(line)
                    kept_count += 1
        
        # Rewrite file with kept records (only if not dry run and there are changes)
        if not dry_run and kept_count < original_count:
            with open(position_file, "w", encoding="utf-8") as f:
                for line in kept_lines:
                    f.write(line + "\n")
            print(f"[TRUNCATE] {position_file.relative_to(PROJECT_ROOT)}: {original_count} -> {kept_count} records")
        elif dry_run and kept_count < original_count:
            print(f"[TRUNCATE] Would truncate {position_file.relative_to(PROJECT_ROOT)}: {original_count} -> {kept_count} records")
    
    except Exception as e:
        print(f"[ERROR] Failed to truncate {position_file}: {e}")
    
    return original_count, kept_count


def cleanup_pnl_snapshots(pnl_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Clean up pnl snapshot files by removing future date records.
    
    Args:
        pnl_dir: Directory containing pnl snapshot JSON files
        dry_run: If True, only show what would be removed without actually modifying files
    
    Returns:
        Tuple of (files_processed, records_removed_total)
    """
    if not pnl_dir.exists():
        return 0, 0
    
    files_processed = 0
    records_removed_total = 0
    
    for pnl_file in pnl_dir.glob("*.json"):
        try:
            with open(pnl_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
            
            original_count = len(data)
            filtered_data = [
                record for record in data
                if not is_future_date(record.get("date", ""))
            ]
            kept_count = len(filtered_data)
            removed_count = original_count - kept_count
            
            if removed_count > 0:
                records_removed_total += removed_count
                if dry_run:
                    print(f"[CLEAN] Would remove {removed_count} future date records from {pnl_file.name}")
                else:
                    with open(pnl_file, "w", encoding="utf-8") as f:
                        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                    print(f"[CLEAN] {pnl_file.name}: removed {removed_count} future date records")
            
            files_processed += 1
        
        except Exception as e:
            print(f"[WARNING] Failed to process {pnl_file}: {e}")
    
    return files_processed, records_removed_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up future date data from data_flow")
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip creating backup before cleanup (not recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    args = parser.parse_args()
    
    print(f"[INFO] Starting cleanup of data after {CUTOFF_DATE}")
    print(f"[INFO] Data flow directory: {DATA_FLOW_DIR}")
    
    if args.dry_run:
        print("[DRY RUN] No files will be deleted")
    
    # Create backup unless skipped
    if not args.skip_backup and not args.dry_run:
        if not create_backup():
            response = input("[WARNING] Backup failed. Continue anyway? (yes/no): ")
            if response.lower() != "yes":
                print("[INFO] Cleanup cancelled")
                return 1
    
    total_deleted_folders = 0
    total_deleted_locks = 0
    total_position_records_removed = 0
    total_pnl_records_removed = 0
    
    # Clean up agent_data directory
    agent_data_dir = DATA_FLOW_DIR / "agent_data"
    if agent_data_dir.exists():
        print(f"\n[INFO] Cleaning agent_data directory...")
        
        # Clean date folders
        for subdir in ["log", "snapshots", "logs"]:
            count, paths = cleanup_date_folders(agent_data_dir, subdir, dry_run=args.dry_run)
            total_deleted_folders += count
        
        # Clean lock files
        locks_dir = agent_data_dir / "shared" / "locks"
        count, paths = cleanup_lock_files(locks_dir, dry_run=args.dry_run)
        total_deleted_locks += count
        
        # Truncate position files
        for model_dir in agent_data_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "shared":
                continue
            position_file = model_dir / "position" / "position.jsonl"
            if position_file.exists():
                orig, kept = truncate_position_file(position_file, dry_run=args.dry_run)
                total_position_records_removed += (orig - kept)
    
    # Clean up trading_summary_each_agent directory
    trading_summary_dir = DATA_FLOW_DIR / "trading_summary_each_agent"
    if trading_summary_dir.exists():
        print(f"\n[INFO] Cleaning trading_summary_each_agent directory...")
        
        # Clean date folders
        for subdir in ["log", "snapshots", "logs"]:
            count, paths = cleanup_date_folders(trading_summary_dir, subdir, dry_run=args.dry_run)
            total_deleted_folders += count
        
        # Clean lock files
        locks_dir = trading_summary_dir / "shared" / "locks"
        count, paths = cleanup_lock_files(locks_dir, dry_run=args.dry_run)
        total_deleted_locks += count
        
        # Truncate position files
        for model_dir in trading_summary_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "shared":
                continue
            position_file = model_dir / "position" / "position.jsonl"
            if position_file.exists():
                orig, kept = truncate_position_file(position_file, dry_run=args.dry_run)
                total_position_records_removed += (orig - kept)
    
    # Clean up pnl_snapshots
    pnl_dir = DATA_FLOW_DIR / "pnl_snapshots"
    if pnl_dir.exists():
        print(f"\n[INFO] Cleaning pnl_snapshots directory...")
        files_processed, records_removed = cleanup_pnl_snapshots(pnl_dir, dry_run=args.dry_run)
        total_pnl_records_removed = records_removed
        print(f"[INFO] Processed {files_processed} pnl snapshot files")
    
    # Summary
    print(f"\n[SUMMARY] Cleanup completed:")
    print(f"  - Deleted folders: {total_deleted_folders}")
    print(f"  - Deleted lock files: {total_deleted_locks}")
    print(f"  - Removed position records: {total_position_records_removed}")
    print(f"  - Removed pnl records: {total_pnl_records_removed}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

