"""Synchronize readable summaries with canonical position ledger fills.

This script only rewrites derived artifacts:
- data_flow/trading_summary_each_agent/<model>/runs/<date>/<time>/readable.md
- sibling execution.json

It does not modify raw model conversations, decision_report.json, position.jsonl,
PnL snapshots, or scheduler logs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_position_records(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            date = str(record.get("date") or "")
            decision_time = str(record.get("decision_time") or "")
            time_part = decision_time.split(" ")[-1].replace(":", "-")
            records[(date, time_part)] = record
    return records


def _format_fill_lines(record: Optional[Dict[str, Any]]) -> List[str]:
    if not record:
        return ["- (ledger record not found; no executed fill can be confirmed from position.jsonl)"]

    fills = record.get("fills")
    if not isinstance(fills, list):
        fills = []
    lines: List[str] = []
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        action = str(fill.get("action") or "unknown")
        symbol = str(fill.get("symbol") or "")
        amount = fill.get("amount", fill.get("shares", 0))
        price = fill.get("price")
        suffix = f" @ {price}" if price not in (None, "") else ""
        fees = []
        if fill.get("commission") not in (None, "", 0):
            fees.append(f"commission={fill.get('commission')}")
        if fill.get("stamp_duty") not in (None, "", 0):
            fees.append(f"stamp_duty={fill.get('stamp_duty')}")
        if fees:
            suffix += f" ({', '.join(fees)})"
        lines.append(f"- `{action}` {symbol} x{amount}{suffix}")
    if lines:
        return lines

    action = record.get("this_action") if isinstance(record, dict) else {}
    action_name = action.get("action", "no_trade") if isinstance(action, dict) else "no_trade"
    return [f"- `{action_name or 'no_trade'}`: no executed buy/sell fill recorded in position ledger"]


def _find_section(lines: List[str], header: str) -> Tuple[Optional[int], Optional[int]]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx
            break
    if start is None:
        return None, None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return start, end


def _section_body(lines: List[str], header: str) -> List[str]:
    start, end = _find_section(lines, header)
    if start is None or end is None:
        return []
    body = lines[start + 1 : end]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return body


def _replace_section(lines: List[str], header: str, body: Iterable[str]) -> List[str]:
    replacement = [header, *body, ""]
    start, end = _find_section(lines, header)
    if start is None or end is None:
        return [*lines, "", *replacement]
    return [*lines[:start], *replacement, *lines[end:]]


def _decision_tuples(lines: Iterable[str]) -> List[Tuple[str, str, int]]:
    tuples: List[Tuple[str, str, int]] = []
    pattern = re.compile(r"`(buy|sell)`\s+(?:SH)?(\d{6})\s+x\s*([0-9]+)", re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match:
            tuples.append((match.group(1).lower(), match.group(2), int(match.group(3))))
    return sorted(tuples)


def _fill_tuples(record: Optional[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    if not record:
        return []
    tuples: List[Tuple[str, str, int]] = []
    fills = record.get("fills")
    if not isinstance(fills, list):
        return tuples
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        action = str(fill.get("action") or "").lower()
        if action not in {"buy", "sell"}:
            continue
        symbol = str(fill.get("symbol") or "").replace("SH", "").replace("SZ", "")
        amount = int(float(fill.get("amount") or fill.get("shares") or 0))
        tuples.append((action, symbol, amount))
    return sorted(tuples)


def _sync_readable(readable_path: Path, record: Optional[Dict[str, Any]]) -> Tuple[bool, bool]:
    text = readable_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    old_decision = _section_body(lines, "## Decision")
    model_stated = _section_body(lines, "## Model-Stated Decision") or old_decision
    ledger_lines = _format_fill_lines(record)
    mismatch = _decision_tuples(model_stated) != _fill_tuples(record)

    lines = _replace_section(lines, "## Decision", ledger_lines)
    lines = _replace_section(lines, "## Model-Stated Decision", model_stated or ["- (no structured model action captured)"])
    consistency_lines = [
        f"- Ledger record found: {record is not None}",
        f"- Model-stated trades match ledger fills: {not mismatch}",
        "- Canonical execution source: `position/position.jsonl` `fills`",
    ]
    lines = _replace_section(lines, "## Ledger Consistency", consistency_lines)

    new_text = "\n".join(lines).rstrip() + "\n"
    changed = new_text != text
    if changed:
        readable_path.write_text(new_text, encoding="utf-8")
    return changed, mismatch


def _sync_execution(execution_path: Path, record: Optional[Dict[str, Any]], mismatch: bool) -> bool:
    if execution_path.exists():
        try:
            payload = json.loads(execution_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    old = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    fills = record.get("fills") if isinstance(record, dict) and isinstance(record.get("fills"), list) else []
    payload.update(
        {
            "ledger_record_found": record is not None,
            "ledger_position_id": record.get("id") if isinstance(record, dict) else None,
            "ledger_decision_time": record.get("decision_time") if isinstance(record, dict) else None,
            "ledger_action": record.get("this_action") if isinstance(record, dict) else None,
            "executed_fills": fills,
            "model_stated_trades_match_ledger_fills": not mismatch,
            "canonical_execution_source": "position/position.jsonl:fills",
        }
    )
    new = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    changed = old != new
    if changed:
        execution_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def sync(root: Path, since: str, apply: bool) -> Dict[str, int]:
    agent_root = root / "data_flow" / "trading_summary_each_agent"
    stats = {
        "run_dirs": 0,
        "readable_changed": 0,
        "execution_changed": 0,
        "mismatches_preserved": 0,
        "missing_ledger": 0,
    }
    for model_dir in sorted(agent_root.iterdir() if agent_root.exists() else []):
        if not model_dir.is_dir():
            continue
        records = _load_position_records(model_dir / "position" / "position.jsonl")
        runs_dir = model_dir / "runs"
        if not runs_dir.exists():
            continue
        for readable in sorted(runs_dir.glob("2026-*/*/readable.md")):
            date = readable.parts[-3]
            time_part = readable.parts[-2]
            if date < since:
                continue
            stats["run_dirs"] += 1
            record = records.get((date, time_part))
            if record is None:
                stats["missing_ledger"] += 1
            if not apply:
                old_decision = _section_body(readable.read_text(encoding="utf-8").splitlines(), "## Decision")
                if _decision_tuples(old_decision) != _fill_tuples(record):
                    stats["mismatches_preserved"] += 1
                continue
            readable_changed, mismatch = _sync_readable(readable, record)
            if mismatch:
                stats["mismatches_preserved"] += 1
            if readable_changed:
                stats["readable_changed"] += 1
            if _sync_execution(readable.with_name("execution.json"), record, mismatch):
                stats["execution_changed"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--since", default="2026-04-01", help="Only sync run dates >= this date")
    parser.add_argument("--apply", action="store_true", help="Rewrite readable.md/execution.json")
    args = parser.parse_args()
    stats = sync(Path(args.root).resolve(), args.since, args.apply)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
