#!/usr/bin/env python3
"""
清理 agent 持仓历史：
1. 深度归并 positions，消除 SH/SZ 前缀不一致及共享引用。
2. 规范 this_action.symbol、decision_time 等字段格式。
3. 支持指定模型或扫描全部模型。
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.price_tools import normalize_decision_time, normalize_positions, normalize_symbol
AGENT_DATA_DIR = BASE_DIR / "data" / "agent_data"


def _sanitize_record(record: Dict[str, any]) -> Dict[str, any]:
    sanitized = copy.deepcopy(record)
    sanitized["positions"] = normalize_positions(sanitized.get("positions", {}))

    action = sanitized.get("this_action")
    if isinstance(action, dict):
        action_symbol = normalize_symbol(action.get("symbol"))
        action["symbol"] = action_symbol or ""

    sanitized["decision_time"] = normalize_decision_time(
        sanitized.get("date", ""), sanitized.get("decision_time")
    )

    decision_count = sanitized.get("decision_count")
    try:
        sanitized["decision_count"] = int(decision_count)
    except Exception:
        sanitized["decision_count"] = 0

    return sanitized


def _sort_key(item: Dict[str, any]) -> Tuple[str, str, int]:
    return (
        item.get("date", ""),
        item.get("decision_time", ""),
        item.get("id", 0),
    )


def migrate_position_file(position_file: Path) -> None:
    if not position_file.exists():
        return

    records: List[Dict[str, any]] = []
    with position_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                records.append(_sanitize_record(doc))
            except Exception:
                continue

    records.sort(key=_sort_key)

    with position_file.open("w", encoding="utf-8") as f:
        for doc in records:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✅ Migrated {position_file} ({len(records)} records)")


def main(models: Optional[List[str]] = None) -> None:
    targets: List[Tuple[str, Path]] = []

    if models:
        for model in models:
            path = AGENT_DATA_DIR / model / "position" / "position.jsonl"
            targets.append((model, path))
    else:
        for model_dir in AGENT_DATA_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            path = model_dir / "position" / "position.jsonl"
            targets.append((model_dir.name, path))

    for model, path in targets:
        if path.exists():
            print(f"🛠  Migrating {model} -> {path}")
            migrate_position_file(path)
        else:
            print(f"⚠️  Skip {model}, file not found: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize agent position history files.")
    parser.add_argument(
        "--model",
        dest="models",
        nargs="*",
        help="指定一个或多个模型名称（默认扫描全部）。",
    )
    args = parser.parse_args()
    main(args.models)

