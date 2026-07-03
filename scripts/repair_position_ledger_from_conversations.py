#!/usr/bin/env python3
"""Repair position.jsonl action metadata from saved raw conversation tool calls.

The trading tools already return the authoritative post-trade ``new_position``.
This utility reads every model run's ``conversation.jsonl``, extracts successful
tool calls, and patches the per-model ``position/position.jsonl`` records with:

- ``actions``: all successful trade/no-trade tool results for the decision slot
- ``fills``: successful buy/sell fills only, including price and fees
- ``this_action``: a backwards-compatible summary that avoids labelling a
  multi-action decision as a single misleading buy/sell/no_trade

It intentionally does not infer trades from natural-language model output.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRADE_ACTIONS = {"buy", "sell"}
KNOWN_ACTIONS = {"buy", "sell", "no_trade"}


@dataclass
class ToolAction:
    action: str
    symbol: str
    amount: float
    payload: Dict[str, Any]
    sequence: int

    @property
    def is_fill(self) -> bool:
        return self.action in TRADE_ACTIONS


def _load_json_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(line)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _load_tool_payload(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    message_type = message.get("type") or message.get("message_class")
    if message_type not in {"tool", "ToolMessage"} and message.get("message_class") != "ToolMessage":
        return None

    content = message.get("content")
    if isinstance(content, dict):
        payload = content
    elif isinstance(content, str):
        try:
            payload = json.loads(content)
        except Exception:
            return None
    else:
        return None

    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None
    action = str(payload.get("action") or "").strip()
    if action not in KNOWN_ACTIONS:
        return None
    return payload


def _to_number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _compact_action(payload: Dict[str, Any], sequence: int) -> ToolAction:
    action = str(payload.get("action") or "").strip()
    symbol = str(payload.get("symbol") or "").strip()
    amount = _to_number(payload.get("amount"))
    return ToolAction(action=action, symbol=symbol, amount=amount, payload=payload, sequence=sequence)


def extract_actions(conversation_path: Path) -> List[ToolAction]:
    actions: List[ToolAction] = []
    if not conversation_path.exists():
        return actions
    for line in conversation_path.read_text(encoding="utf-8", errors="replace").splitlines():
        message = _load_json_line(line)
        if not message:
            continue
        payload = _load_tool_payload(message)
        if not payload:
            continue
        actions.append(_compact_action(payload, len(actions)))
    return actions


def _action_dict(action: ToolAction) -> Dict[str, Any]:
    payload = action.payload
    item: Dict[str, Any] = {
        "action": action.action,
        "symbol": action.symbol,
        "amount": int(action.amount) if float(action.amount).is_integer() else action.amount,
        "sequence": action.sequence,
        "decision_time": payload.get("decision_time"),
        "decision_count": payload.get("decision_count"),
    }
    for key in ("price", "cost", "revenue", "commission", "stamp_duty", "net_revenue"):
        if key in payload:
            item[key] = payload.get(key)
    return item


def _summary_action(actions: List[ToolAction]) -> Dict[str, Any]:
    fills = [a for a in actions if a.is_fill]
    if len(fills) == 1:
        fill = fills[0]
        return {
            "action": fill.action,
            "symbol": fill.symbol,
            "amount": int(fill.amount) if float(fill.amount).is_integer() else fill.amount,
        }
    if len(fills) > 1:
        return {
            "action": "multi_trade",
            "symbol": "",
            "amount": 0,
            "actions_count": len(fills),
        }
    if actions:
        return {"action": "no_trade", "symbol": "", "amount": 0}
    return {"action": "unknown", "symbol": "", "amount": 0}


def _run_key_from_path(run_dir: Path) -> Tuple[str, str]:
    date_str = run_dir.parent.name
    time_str = run_dir.name.replace("-", ":")
    return date_str, f"{date_str} {time_str}"


def collect_run_actions(model_dir: Path) -> Dict[Tuple[str, str], List[ToolAction]]:
    out: Dict[Tuple[str, str], List[ToolAction]] = {}
    runs_dir = model_dir / "runs"
    if not runs_dir.exists():
        return out
    for conversation in sorted(runs_dir.glob("2026-*/*/conversation.jsonl")):
        run_dir = conversation.parent
        actions = extract_actions(conversation)
        if not actions:
            continue
        out[_run_key_from_path(run_dir)] = actions
    return out


def _load_position_records(position_file: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not position_file.exists():
        return records
    for line in position_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _write_position_records(position_file: Path, records: Iterable[Dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    position_file.write_text(text, encoding="utf-8")


def _last_fill_position(actions: List[ToolAction]) -> Optional[Dict[str, Any]]:
    for action in reversed(actions):
        if action.is_fill and isinstance(action.payload.get("new_position"), dict):
            return action.payload["new_position"]
    return None


def _normalize_lot_number(value: float) -> Any:
    return int(value) if float(value).is_integer() else value


def _clean_positions(positions: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {"CASH": _to_number(positions.get("CASH"))}
    for symbol, entry in positions.items():
        if symbol == "CASH" or not isinstance(entry, dict):
            continue
        shares = _to_number(entry.get("shares"))
        if shares <= 1e-9:
            continue
        lots = []
        for lot in entry.get("lots") or []:
            if not isinstance(lot, dict):
                continue
            lot_shares = _to_number(lot.get("shares"))
            if lot_shares <= 1e-9:
                continue
            lots.append(
                {
                    "shares": _normalize_lot_number(lot_shares),
                    "purchase_date": lot.get("purchase_date") or entry.get("purchase_date"),
                    "avg_price": _to_number(lot.get("avg_price", entry.get("avg_price"))),
                }
            )
        if not lots:
            lots = [
                {
                    "shares": _normalize_lot_number(shares),
                    "purchase_date": entry.get("purchase_date"),
                    "avg_price": _to_number(entry.get("avg_price")),
                }
            ]
        total_cost = sum(_to_number(lot.get("shares")) * _to_number(lot.get("avg_price")) for lot in lots)
        total_shares = sum(_to_number(lot.get("shares")) for lot in lots)
        avg_price = total_cost / total_shares if total_shares > 0 else _to_number(entry.get("avg_price"))
        purchase_dates = [str(lot.get("purchase_date")) for lot in lots if lot.get("purchase_date")]
        cleaned[symbol] = {
            "shares": _normalize_lot_number(total_shares),
            "purchase_date": min(purchase_dates) if purchase_dates else entry.get("purchase_date"),
            "avg_price": avg_price,
            "lots": lots,
        }
    return cleaned


def _remove_from_lots(lots: List[Dict[str, Any]], amount: float) -> Tuple[List[Dict[str, Any]], float]:
    remaining_amount = amount
    remaining_lots: List[Dict[str, Any]] = []
    for lot in lots:
        lot_shares = _to_number(lot.get("shares"))
        if remaining_amount <= 1e-9:
            remaining_lots.append(copy.deepcopy(lot))
            continue
        if lot_shares <= remaining_amount + 1e-9:
            remaining_amount -= lot_shares
            continue
        new_lot = copy.deepcopy(lot)
        new_lot["shares"] = _normalize_lot_number(lot_shares - remaining_amount)
        remaining_amount = 0.0
        remaining_lots.append(new_lot)
    return remaining_lots, remaining_amount


def _apply_fill(positions: Dict[str, Any], fill: ToolAction, date_str: str) -> Tuple[Dict[str, Any], Optional[str]]:
    new_positions = _clean_positions(copy.deepcopy(positions))
    payload = fill.payload
    symbol = fill.symbol
    amount = _to_number(payload.get("amount"))
    price = _to_number(payload.get("price"))
    warning: Optional[str] = None

    if fill.action == "buy":
        raw_cost = _to_number(payload.get("cost")) or price * amount
        commission = _to_number(payload.get("commission"))
        new_positions["CASH"] = _to_number(new_positions.get("CASH")) - raw_cost - commission
        entry = copy.deepcopy(new_positions.get(symbol) or {})
        old_shares = _to_number(entry.get("shares"))
        old_avg = _to_number(entry.get("avg_price")) or price
        lots = copy.deepcopy(entry.get("lots") or [])
        lots.append({"shares": _normalize_lot_number(amount), "purchase_date": date_str, "avg_price": price})
        total_shares = old_shares + amount
        avg_price = ((old_avg * old_shares) + (price * amount)) / total_shares if total_shares else price
        entry.update(
            {
                "shares": _normalize_lot_number(total_shares),
                "purchase_date": entry.get("purchase_date") or date_str,
                "avg_price": avg_price,
                "lots": lots,
            }
        )
        new_positions[symbol] = entry
        return _clean_positions(new_positions), warning

    if fill.action == "sell":
        entry = copy.deepcopy(new_positions.get(symbol) or {})
        old_shares = _to_number(entry.get("shares"))
        if old_shares + 1e-9 < amount:
            warning = f"insufficient_shares:{symbol}:have={old_shares}:sell={amount}"
            amount = old_shares
        if amount <= 1e-9:
            return _clean_positions(new_positions), warning

        lots = copy.deepcopy(entry.get("lots") or [])
        if not lots:
            lots = [{"shares": _normalize_lot_number(old_shares), "purchase_date": entry.get("purchase_date"), "avg_price": entry.get("avg_price")}]
        remaining_lots, unfilled = _remove_from_lots(lots, amount)
        if unfilled > 1e-9:
            warning = f"insufficient_lots:{symbol}:unfilled={unfilled}"

        revenue = _to_number(payload.get("revenue")) or price * amount
        commission = _to_number(payload.get("commission"))
        stamp_duty = _to_number(payload.get("stamp_duty"))
        net_revenue = _to_number(payload.get("net_revenue")) or (revenue - commission - stamp_duty)
        new_positions["CASH"] = _to_number(new_positions.get("CASH")) + net_revenue

        remaining_shares = sum(_to_number(lot.get("shares")) for lot in remaining_lots)
        if remaining_shares <= 1e-9:
            new_positions.pop(symbol, None)
        else:
            total_cost = sum(_to_number(lot.get("shares")) * _to_number(lot.get("avg_price")) for lot in remaining_lots)
            purchase_dates = [str(lot.get("purchase_date")) for lot in remaining_lots if lot.get("purchase_date")]
            new_positions[symbol] = {
                "shares": _normalize_lot_number(remaining_shares),
                "purchase_date": min(purchase_dates) if purchase_dates else entry.get("purchase_date"),
                "avg_price": total_cost / remaining_shares,
                "lots": remaining_lots,
            }
        return _clean_positions(new_positions), warning

    return new_positions, f"unsupported_fill:{fill.action}"


def repair_model_with_tool_positions(model_dir: Path, apply: bool) -> Dict[str, Any]:
    model = model_dir.name
    position_file = model_dir / "position" / "position.jsonl"
    records = _load_position_records(position_file)
    actions_by_key = collect_run_actions(model_dir)
    repaired = 0
    position_changed = 0
    missing_records = 0
    no_actions = 0
    action_counts = Counter()

    records_by_key = {
        (str(record.get("date") or ""), str(record.get("decision_time") or "")): record for record in records
    }

    for key, actions in actions_by_key.items():
        record = records_by_key.get(key)
        if record is None:
            missing_records += 1
            continue
        action_dicts = [_action_dict(action) for action in actions]
        fill_dicts = [_action_dict(action) for action in actions if action.is_fill]
        action_counts[len(fill_dicts)] += 1

        new_position = _last_fill_position(actions)
        old_position = record.get("positions")
        if new_position is not None and new_position != old_position:
            record["positions"] = new_position
            position_changed += 1

        old_this_action = record.get("this_action")
        record["actions"] = action_dicts
        record["fills"] = fill_dicts
        record["this_action"] = _summary_action(actions)
        record["ledger_repair"] = {
            "source": "conversation_tool_calls",
            "tool_action_count": len(action_dicts),
            "fill_count": len(fill_dicts),
            "previous_this_action": old_this_action,
        }
        repaired += 1

    no_actions = max(0, len(records) - repaired)
    if apply:
        records.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("decision_time") or ""), int(item.get("id") or 0)))
        _write_position_records(position_file, records)

    return {
        "model": model,
        "mode": "tool_positions",
        "records": len(records),
        "runs_with_actions": len(actions_by_key),
        "repaired_records": repaired,
        "position_changed": position_changed,
        "missing_position_records": missing_records,
        "records_without_actions": no_actions,
        "fill_count_distribution": dict(sorted(action_counts.items())),
    }


def repair_model_by_replay(model_dir: Path, apply: bool) -> Dict[str, Any]:
    model = model_dir.name
    position_file = model_dir / "position" / "position.jsonl"
    records = _load_position_records(position_file)
    records.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("decision_time") or ""), int(item.get("id") or 0)))
    actions_by_key = collect_run_actions(model_dir)

    repaired = 0
    warnings: List[Dict[str, Any]] = []
    action_counts = Counter()
    current_positions: Optional[Dict[str, Any]] = None

    for record in records:
        date_str = str(record.get("date") or "")
        decision_time = str(record.get("decision_time") or "")
        key = (date_str, decision_time)
        original_action = record.get("this_action")
        actions = actions_by_key.get(key, [])
        fills = [action for action in actions if action.is_fill]
        action_counts[len(fills)] += 1

        if current_positions is None:
            current_positions = _clean_positions(copy.deepcopy(record.get("positions") or {"CASH": 1000000.0}))
        elif not fills:
            # No successful buy/sell fill: carry the latest repaired position forward.
            current_positions = _clean_positions(current_positions)
        else:
            current_positions = _clean_positions(current_positions)

        for fill in fills:
            current_positions, warning = _apply_fill(current_positions, fill, date_str)
            if warning:
                warnings.append({"time": decision_time, "action": _action_dict(fill), "warning": warning})

        if actions:
            record["actions"] = [_action_dict(action) for action in actions]
            record["fills"] = [_action_dict(action) for action in fills]
            record["this_action"] = _summary_action(actions)
            repaired += 1
        else:
            record.setdefault("actions", [])
            record.setdefault("fills", [])

        record["positions"] = copy.deepcopy(current_positions)
        record["ledger_repair"] = {
            "source": "replayed_conversation_fills",
            "tool_action_count": len(actions),
            "fill_count": len(fills),
            "previous_this_action": original_action,
        }

    if apply:
        _write_position_records(position_file, records)

    return {
        "model": model,
        "mode": "replay",
        "records": len(records),
        "runs_with_actions": len(actions_by_key),
        "repaired_records": repaired,
        "warnings": len(warnings),
        "warning_samples": warnings[:10],
        "fill_count_distribution": dict(sorted(action_counts.items())),
    }


def audit_repaired(root: Path) -> Dict[str, Any]:
    base = root / "data_flow" / "trading_summary_each_agent"
    label_anomalies: List[Dict[str, Any]] = []
    reset_issues: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for position_file in sorted(base.glob("*/position/position.jsonl")):
        model = position_file.parts[-3]
        records = _load_position_records(position_file)
        records.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("decision_time") or ""), int(item.get("id") or 0)))
        previous: Optional[Dict[str, Any]] = None
        for record in records:
            if previous is not None:
                previous_id = int(previous.get("id") or -1)
                current_id = int(record.get("id") or -1)
                if current_id <= previous_id:
                    reset_issues.append({
                        "model": model,
                        "previous_time": previous.get("decision_time"),
                        "previous_id": previous.get("id"),
                        "time": record.get("decision_time"),
                        "id": record.get("id"),
                        "reason": "non_increasing_id",
                    })
                prev_pos = previous.get("positions") or {}
                cur_pos = record.get("positions") or {}
                prev_cash = _to_number(prev_pos.get("CASH"))
                cur_cash = _to_number(cur_pos.get("CASH"))
                action = (record.get("this_action") or {}).get("action")
                if action == "buy" and cur_cash > prev_cash + 1e-6:
                    label_anomalies.append({"model": model, "time": record.get("decision_time"), "kind": "buy_cash_increased"})
                if action == "sell" and cur_cash < prev_cash - 1e-6:
                    label_anomalies.append({"model": model, "time": record.get("decision_time"), "kind": "sell_cash_decreased"})
                if action == "no_trade":
                    prev_non_cash = {k: v for k, v in prev_pos.items() if k != "CASH"}
                    cur_non_cash = {k: v for k, v in cur_pos.items() if k != "CASH"}
                    if abs(cur_cash - prev_cash) > 1e-6 or prev_non_cash != cur_non_cash:
                        label_anomalies.append({"model": model, "time": record.get("decision_time"), "kind": "no_trade_position_changed"})
            previous = record
        if records:
            latest = records[-1]
            summaries.append({
                "model": model,
                "records": len(records),
                "last_time": latest.get("decision_time"),
                "last_id": latest.get("id"),
                "cash": (latest.get("positions") or {}).get("CASH"),
                "this_action": latest.get("this_action"),
                "actions": len(latest.get("actions") or []),
                "fills": len(latest.get("fills") or []),
            })

    return {
        "reset_issues": reset_issues,
        "label_anomalies": label_anomalies,
        "model_summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Extracted run root containing data_flow/")
    parser.add_argument("--apply", action="store_true", help="Modify position.jsonl files in place")
    parser.add_argument(
        "--mode",
        choices=("replay", "tool-positions"),
        default="replay",
        help="replay fills chronologically, or copy each tool result's new_position",
    )
    parser.add_argument("--report", type=Path, help="Write JSON repair report")
    args = parser.parse_args()

    base = args.root / "data_flow" / "trading_summary_each_agent"
    results = []
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and (p / "position" / "position.jsonl").exists()):
        if args.mode == "tool-positions":
            results.append(repair_model_with_tool_positions(model_dir, apply=args.apply))
        else:
            results.append(repair_model_by_replay(model_dir, apply=args.apply))

    audit = audit_repaired(args.root)
    report = {
        "root": str(args.root),
        "applied": args.apply,
        "mode": args.mode,
        "models": results,
        "audit": audit,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
