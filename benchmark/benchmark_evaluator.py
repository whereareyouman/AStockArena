from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FLOW = PROJECT_ROOT / "data_flow"
SNAPSHOT_ROOT = DATA_FLOW / "agent_data" / "shared" / "snapshots"
AGENT_ROOT = DATA_FLOW / "trading_summary_each_agent"
PNL_ROOT = DATA_FLOW / "pnl_snapshots"
REPORT_ROOT = DATA_FLOW / "benchmark_reports"
INITIAL_CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class SnapshotPoint:
    decision_time: str
    dt: datetime
    path: Path
    prices: Dict[str, float]
    indicators: Dict[str, Any]
    news: Dict[str, Any]


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d":
                return dt.replace(hour=0, minute=0, second=0)
            return dt
        except ValueError:
            continue
    return None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _price_from_payload(payload: Any) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if isinstance(summary, dict):
        price = _safe_float(summary.get("close"))
        if price and price > 0:
            return price
    points = payload.get("prices_3d")
    if isinstance(points, list):
        for point in reversed(points):
            if isinstance(point, dict):
                price = _safe_float(point.get("close"))
                if price and price > 0:
                    return price
    return None


def load_snapshots(snapshot_root: Path = SNAPSHOT_ROOT) -> List[SnapshotPoint]:
    points: Dict[str, SnapshotPoint] = {}
    for path in sorted(snapshot_root.glob("*/*.json")):
        payload = _read_json(path, {})
        if not isinstance(payload, dict):
            continue
        decision_time = str(payload.get("decision_time") or "").strip()
        dt = _parse_dt(decision_time)
        if not decision_time or dt is None:
            continue
        prices: Dict[str, float] = {}
        for symbol, price_payload in (payload.get("prices") or {}).items():
            price = _price_from_payload(price_payload)
            if price is not None:
                prices[str(symbol)] = price
        points[decision_time] = SnapshotPoint(
            decision_time=decision_time,
            dt=dt,
            path=path,
            prices=prices,
            indicators=payload.get("indicators") or {},
            news=payload.get("news") or {},
        )
    return sorted(points.values(), key=lambda item: item.dt)


def _next_snapshot_index(points: List[SnapshotPoint]) -> Dict[str, Optional[SnapshotPoint]]:
    out: Dict[str, Optional[SnapshotPoint]] = {}
    for idx, point in enumerate(points):
        out[point.decision_time] = points[idx + 1] if idx + 1 < len(points) else None
    return out


def _snapshot_by_time(points: List[SnapshotPoint]) -> Dict[str, SnapshotPoint]:
    return {point.decision_time: point for point in points}


def _top_symbols_by_next_return(current: SnapshotPoint, nxt: SnapshotPoint, top_n: int = 3) -> List[str]:
    returns: List[Tuple[str, float]] = []
    for symbol, price in current.prices.items():
        next_price = nxt.prices.get(symbol)
        if price > 0 and next_price is not None:
            returns.append((symbol, (next_price / price - 1.0) * 100.0))
    returns.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in returns[:top_n]]


def load_position_records(agent_root: Path = AGENT_ROOT) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(agent_root.glob("*/position/position.jsonl")):
        signature = path.parents[1].name
        rows = [row for row in _read_jsonl(path) if not row.get("seed")]
        rows.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("decision_time") or ""), int(row.get("id") or 0)))
        out[signature] = rows
    return out


def load_decision_reports(agent_root: Path = AGENT_ROOT) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in sorted(agent_root.glob("*/log/*/*.jsonl")):
        signature = path.parts[-4]
        for row in _read_jsonl(path):
            if row.get("event_type") != "decision_evidence_report":
                continue
            decision_time = str(row.get("decision_time") or "").strip()
            if not decision_time:
                continue
            out.setdefault(signature, {})[decision_time] = row
    return out


def load_pnl_snapshots(pnl_root: Path = PNL_ROOT) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(pnl_root.glob("pnl_*.json")):
        signature = path.stem.removeprefix("pnl_")
        rows = _read_json(path, [])
        if isinstance(rows, list):
            rows = [row for row in rows if isinstance(row, dict)]
            rows.sort(key=lambda row: str(row.get("decision_time") or row.get("date") or ""))
            out[signature] = rows
    return out


def _candidate_for_symbol(report: Optional[Dict[str, Any]], symbol: str) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None
    candidates = report.get("candidate_review")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("symbol") or "").upper() == symbol.upper():
            return candidate
    return None


def _news_direction_flags(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    flags = {
        "positive_count": 0,
        "negative_count": 0,
        "mixed_count": 0,
        "has_positive": False,
        "has_negative_or_mixed": False,
        "directions": [],
    }
    if not isinstance(candidate, dict):
        return flags
    news_items = candidate.get("news_evidence_used")
    if not isinstance(news_items, list):
        return flags
    directions: List[str] = []
    for item in news_items:
        if not isinstance(item, dict):
            continue
        direction = str(item.get("claimed_direction") or "unknown").lower()
        directions.append(direction)
        if direction == "positive":
            flags["positive_count"] += 1
        elif direction == "negative":
            flags["negative_count"] += 1
        elif direction == "mixed":
            flags["mixed_count"] += 1
    flags["has_positive"] = flags["positive_count"] > 0 or flags["mixed_count"] > 0
    flags["has_negative_or_mixed"] = flags["negative_count"] > 0 or flags["mixed_count"] > 0
    flags["directions"] = directions
    return flags


def _price_evidence(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    payload = candidate.get("price_evidence_used")
    return payload if isinstance(payload, dict) else {}


def _has_reason(candidate: Optional[Dict[str, Any]], action: Dict[str, Any]) -> bool:
    texts = [
        action.get("reason_text"),
        candidate.get("buy_reason_text") if isinstance(candidate, dict) else None,
        candidate.get("reject_or_hold_reason_text") if isinstance(candidate, dict) else None,
    ]
    return any(str(text or "").strip() for text in texts)


def _buy_price_from_positions(record: Dict[str, Any], symbol: str) -> Optional[float]:
    positions = record.get("positions") or {}
    entry = positions.get(symbol)
    if isinstance(entry, dict):
        lots = entry.get("lots")
        if isinstance(lots, list) and lots:
            last = lots[-1]
            if isinstance(last, dict):
                price = _safe_float(last.get("avg_price"))
                if price and price > 0:
                    return price
        price = _safe_float(entry.get("avg_price"))
        if price and price > 0:
            return price
    return None


def _iter_executed_buys(position_records: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for record in position_records:
        fills = record.get("fills")
        if isinstance(fills, list):
            for fill in fills:
                if not isinstance(fill, dict):
                    continue
                if str(fill.get("action") or "").lower() != "buy":
                    continue
                symbol = str(fill.get("symbol") or "").upper()
                amount = _safe_float(fill.get("amount"), 0.0) or 0.0
                if not symbol or amount <= 0:
                    continue
                yield record, fill
            continue

        action = record.get("this_action")
        if not isinstance(action, dict):
            continue
        if str(action.get("action") or "").lower() != "buy":
            continue
        symbol = str(action.get("symbol") or "").upper()
        amount = _safe_float(action.get("amount"), 0.0) or 0.0
        if not symbol or amount <= 0:
            continue
        yield record, action


def _basis_outcome_metrics(equities: List[Optional[float]]) -> Dict[str, Any]:
    values = [value for value in equities if value is not None and value > 0]
    if not values:
        return {
            "points": 0,
            "total_return_pct": None,
            "max_drawdown_pct": None,
            "sharpe": None,
            "calmar": None,
        }
    final = values[-1]
    total_return_pct = (final / INITIAL_CAPITAL - 1.0) * 100.0

    peak = values[0]
    max_dd = 0.0
    for equity in values:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity / peak - 1.0) * 100.0)

    period_returns = []
    for prev, cur in zip(values, values[1:]):
        if prev > 0:
            period_returns.append(cur / prev - 1.0)
    sharpe = None
    if len(period_returns) >= 2:
        mean = sum(period_returns) / len(period_returns)
        variance = sum((value - mean) ** 2 for value in period_returns) / (len(period_returns) - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = (mean / std) * math.sqrt(len(period_returns))
    calmar = None
    if max_dd < 0:
        calmar = total_return_pct / abs(max_dd)

    return {
        "points": len(values),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def _outcome_metrics(pnl_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not pnl_rows:
        empty = _basis_outcome_metrics([])
        return {
            **empty,
            "realized_return_pct": None,
            "realized": empty,
            "unrealized": empty,
        }

    unrealized_equities = [
        _safe_float(row.get("unrealized_equity"), _safe_float(row.get("equity")))
        for row in pnl_rows
    ]
    realized_equities = [
        _safe_float(row.get("realized_equity"))
        for row in pnl_rows
    ]
    if not any(value is not None and value > 0 for value in realized_equities):
        realized_equities = [_safe_float(row.get("equity")) for row in pnl_rows]

    unrealized = _basis_outcome_metrics(unrealized_equities)
    realized = _basis_outcome_metrics(realized_equities)
    return {
        **unrealized,
        "realized_return_pct": realized.get("total_return_pct"),
        "realized": realized,
        "unrealized": unrealized,
    }


def _workflow_metrics(report_rows: Dict[str, Dict[str, Any]], decisions: int) -> Dict[str, Any]:
    total_reports = len(report_rows)
    parse_success = sum(1 for row in report_rows.values() if row.get("parse_success"))
    schema_valid = sum(1 for row in report_rows.values() if row.get("schema_valid"))
    candidate_nonempty = 0
    action_nonempty = 0
    news_evidence = 0
    price_evidence = 0
    for row in report_rows.values():
        report = row.get("report") if isinstance(row.get("report"), dict) else {}
        candidates = report.get("candidate_review") if isinstance(report, dict) else []
        actions = report.get("actions_planned_or_taken") if isinstance(report, dict) else []
        if isinstance(candidates, list) and candidates:
            candidate_nonempty += 1
            if any(isinstance(c, dict) and c.get("news_evidence_used") for c in candidates):
                news_evidence += 1
            if any(isinstance(c, dict) and isinstance(c.get("price_evidence_used"), dict) for c in candidates):
                price_evidence += 1
        if isinstance(actions, list) and actions:
            action_nonempty += 1
    denom = decisions or total_reports or 1
    return {
        "decisions": decisions,
        "reports": total_reports,
        "parse_success_rate": parse_success / denom,
        "schema_valid_rate": schema_valid / denom,
        "candidate_review_rate": candidate_nonempty / denom,
        "action_reason_rate": action_nonempty / denom,
        "news_evidence_rate": news_evidence / denom,
        "price_evidence_rate": price_evidence / denom,
    }


def evaluate(
    snapshot_points: List[SnapshotPoint],
    position_records: Dict[str, List[Dict[str, Any]]],
    report_rows: Dict[str, Dict[str, Dict[str, Any]]],
    pnl_rows: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    snapshots_by_time = _snapshot_by_time(snapshot_points)
    next_by_time = _next_snapshot_index(snapshot_points)
    model_metrics: Dict[str, Any] = {}
    buy_rows: List[Dict[str, Any]] = []
    failure_modes = {
        "fin_snr_failure_count": 0,
        "fin_snr_loss": 0.0,
        "news_conflict_failure_count": 0,
        "news_conflict_loss": 0.0,
        "overheated_positive_news_failure_count": 0,
        "overheated_positive_news_loss": 0.0,
        "price_confirmation_breach_count": 0,
        "price_confirmation_breach_loss": 0.0,
        "weak_rationale_loss_count": 0,
        "weak_rationale_loss": 0.0,
    }

    signatures = sorted(set(position_records) | set(report_rows) | set(pnl_rows))
    for signature in signatures:
        records = position_records.get(signature, [])
        reports_for_model = report_rows.get(signature, {})
        buys = list(_iter_executed_buys(records))
        buy_count = 0
        hit_count = 0
        top3_count = 0
        next_returns: List[float] = []
        model_failure = {
            key: (0.0 if key.endswith("_loss") else 0)
            for key in failure_modes
        }
        buy_times = set()

        for record, action in buys:
            decision_time = str(record.get("decision_time") or "").strip()
            buy_times.add(decision_time)
            symbol = str(action.get("symbol") or "").upper()
            amount = _safe_float(action.get("amount"), 0.0) or 0.0
            current = snapshots_by_time.get(decision_time)
            nxt = next_by_time.get(decision_time)
            if current is None or nxt is None:
                continue
            entry_price = current.prices.get(symbol) or _buy_price_from_positions(record, symbol)
            next_price = nxt.prices.get(symbol)
            if not entry_price or not next_price:
                continue

            buy_count += 1
            next_return_pct = (next_price / entry_price - 1.0) * 100.0
            next_returns.append(next_return_pct)
            hit = next_return_pct > 0
            if hit:
                hit_count += 1
            top3_symbols = _top_symbols_by_next_return(current, nxt, top_n=3)
            top3 = symbol in top3_symbols
            if top3:
                top3_count += 1

            report_entry = reports_for_model.get(decision_time) or {}
            report = report_entry.get("report") if isinstance(report_entry.get("report"), dict) else None
            candidate = _candidate_for_symbol(report, symbol)
            news_flags = _news_direction_flags(candidate)
            price_payload = _price_evidence(candidate)
            rsi = _safe_float(price_payload.get("rsi_3") or price_payload.get("RSI_3"))
            recent_change_pct = _safe_float(price_payload.get("recent_change_pct"))
            loss_amount = max(0.0, -next_return_pct / 100.0 * amount * entry_price)
            is_loss = next_return_pct < 0
            has_positive = bool(news_flags["has_positive"])
            has_conflict = has_positive and bool(news_flags["has_negative_or_mixed"])
            overheated = has_positive and (
                (rsi is not None and rsi >= 70.0)
                or (recent_change_pct is not None and recent_change_pct >= 2.0)
            )
            weak_reason = not _has_reason(candidate, action)

            if is_loss and has_positive:
                model_failure["fin_snr_failure_count"] += 1
                model_failure["fin_snr_loss"] += loss_amount
                failure_modes["fin_snr_failure_count"] += 1
                failure_modes["fin_snr_loss"] += loss_amount
            if is_loss and has_conflict:
                model_failure["news_conflict_failure_count"] += 1
                model_failure["news_conflict_loss"] += loss_amount
                failure_modes["news_conflict_failure_count"] += 1
                failure_modes["news_conflict_loss"] += loss_amount
            if is_loss and overheated:
                model_failure["overheated_positive_news_failure_count"] += 1
                model_failure["overheated_positive_news_loss"] += loss_amount
                model_failure["price_confirmation_breach_count"] += 1
                model_failure["price_confirmation_breach_loss"] += loss_amount
                failure_modes["overheated_positive_news_failure_count"] += 1
                failure_modes["overheated_positive_news_loss"] += loss_amount
                failure_modes["price_confirmation_breach_count"] += 1
                failure_modes["price_confirmation_breach_loss"] += loss_amount
            if is_loss and weak_reason:
                model_failure["weak_rationale_loss_count"] += 1
                model_failure["weak_rationale_loss"] += loss_amount
                failure_modes["weak_rationale_loss_count"] += 1
                failure_modes["weak_rationale_loss"] += loss_amount

            buy_rows.append({
                "signature": signature,
                "decision_time": decision_time,
                "symbol": symbol,
                "amount": amount,
                "entry_price": entry_price,
                "next_price": next_price,
                "next_return_pct": next_return_pct,
                "hit": hit,
                "top3_capture": top3,
                "top3_symbols": "|".join(top3_symbols),
                "positive_news_count": news_flags["positive_count"],
                "negative_news_count": news_flags["negative_count"],
                "mixed_news_count": news_flags["mixed_count"],
                "fin_snr_failure": bool(is_loss and has_positive),
                "news_conflict_failure": bool(is_loss and has_conflict),
                "overheated_positive_news_failure": bool(is_loss and overheated),
                "price_confirmation_breach": bool(is_loss and overheated),
                "weak_rationale_loss": bool(is_loss and weak_reason),
                "estimated_loss": loss_amount,
                "reason_text": action.get("reason_text") or (candidate or {}).get("buy_reason_text") or "",
            })

        avg_next_return = sum(next_returns) / len(next_returns) if next_returns else None
        missed_opportunity_count = 0
        for record in records:
            decision_time = str(record.get("decision_time") or "").strip()
            if not decision_time or decision_time in buy_times:
                continue
            current = snapshots_by_time.get(decision_time)
            nxt = next_by_time.get(decision_time)
            if current is None or nxt is None:
                continue
            top_symbols = _top_symbols_by_next_return(current, nxt, top_n=3)
            has_positive_top = False
            for top_symbol in top_symbols:
                price = current.prices.get(top_symbol)
                next_price = nxt.prices.get(top_symbol)
                if price and next_price and next_price > price:
                    has_positive_top = True
                    break
            if has_positive_top:
                missed_opportunity_count += 1
        model_metrics[signature] = {
            "outcome": _outcome_metrics(pnl_rows.get(signature, [])),
            "action_quality": {
                "buy_count": buy_count,
                "buy_hit_rate": hit_count / buy_count if buy_count else None,
                "buy_avg_next_return_pct": avg_next_return,
                "top3_capture_rate": top3_count / buy_count if buy_count else None,
                "missed_opportunity_count": missed_opportunity_count,
            },
            "fin_snr": model_failure,
            "workflow": _workflow_metrics(reports_for_model, len(records)),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_count": len(snapshot_points),
        "model_count": len(model_metrics),
        "models": model_metrics,
        "failure_modes": failure_modes,
        "buy_trades": buy_rows,
    }


def _round_for_output(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write_summary_csv(report: Dict[str, Any], path: Path) -> None:
    rows = []
    for signature, metrics in sorted((report.get("models") or {}).items()):
        outcome = metrics.get("outcome") or {}
        action = metrics.get("action_quality") or {}
        fin = metrics.get("fin_snr") or {}
        workflow = metrics.get("workflow") or {}
        for basis in ("realized", "unrealized"):
            basis_outcome = outcome.get(basis) if isinstance(outcome.get(basis), dict) else outcome
            rows.append({
                "signature": signature,
                "equity_basis": basis,
                "total_return_pct": basis_outcome.get("total_return_pct"),
                "max_drawdown_pct": basis_outcome.get("max_drawdown_pct"),
                "sharpe": basis_outcome.get("sharpe"),
                "calmar": basis_outcome.get("calmar"),
                "buy_count": action.get("buy_count"),
                "buy_hit_rate": action.get("buy_hit_rate"),
                "buy_avg_next_return_pct": action.get("buy_avg_next_return_pct"),
                "top3_capture_rate": action.get("top3_capture_rate"),
                "missed_opportunity_count": action.get("missed_opportunity_count"),
                "fin_snr_failure_count": fin.get("fin_snr_failure_count"),
                "fin_snr_loss": fin.get("fin_snr_loss"),
                "news_conflict_failure_count": fin.get("news_conflict_failure_count"),
                "news_conflict_loss": fin.get("news_conflict_loss"),
                "price_confirmation_breach_count": fin.get("price_confirmation_breach_count"),
                "price_confirmation_breach_loss": fin.get("price_confirmation_breach_loss"),
                "weak_rationale_loss_count": fin.get("weak_rationale_loss_count"),
                "weak_rationale_loss": fin.get("weak_rationale_loss"),
                "workflow_parse_success_rate": workflow.get("parse_success_rate"),
                "workflow_schema_valid_rate": workflow.get("schema_valid_rate"),
                "workflow_candidate_review_rate": workflow.get("candidate_review_rate"),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["signature"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round_for_output(value) for key, value in row.items()})


def _write_buy_trades_csv(report: Dict[str, Any], path: Path) -> None:
    rows = report.get("buy_trades") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "signature",
        "decision_time",
        "symbol",
        "amount",
        "entry_price",
        "next_price",
        "next_return_pct",
        "hit",
        "top3_capture",
        "top3_symbols",
        "positive_news_count",
        "negative_news_count",
        "mixed_news_count",
        "fin_snr_failure",
        "news_conflict_failure",
        "overheated_positive_news_failure",
        "price_confirmation_breach",
        "weak_rationale_loss",
        "estimated_loss",
        "reason_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round_for_output(row.get(key)) for key in fieldnames})


def _write_latest_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Benchmark Latest",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Snapshots: {report.get('snapshot_count')}",
        f"- Models: {report.get('model_count')}",
        "",
        "| Model | Basis | Return % | MDD % | Sharpe | Buy Count | Hit Rate | Fin-SNR Loss | Workflow Schema |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for signature, metrics in sorted((report.get("models") or {}).items()):
        outcome = metrics.get("outcome") or {}
        action = metrics.get("action_quality") or {}
        fin = metrics.get("fin_snr") or {}
        workflow = metrics.get("workflow") or {}
        for basis in ("realized", "unrealized"):
            basis_outcome = outcome.get(basis) if isinstance(outcome.get(basis), dict) else outcome
            lines.append(
                "| {sig} | {basis} | {ret} | {dd} | {sharpe} | {buys} | {hit} | {loss} | {schema} |".format(
                    sig=signature,
                    basis=basis,
                    ret=_round_for_output(basis_outcome.get("total_return_pct")),
                    dd=_round_for_output(basis_outcome.get("max_drawdown_pct")),
                    sharpe=_round_for_output(basis_outcome.get("sharpe")),
                    buys=action.get("buy_count"),
                    hit=_round_for_output(action.get("buy_hit_rate")),
                    loss=_round_for_output(fin.get("fin_snr_loss")),
                    schema=_round_for_output(workflow.get("schema_valid_rate")),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _latest_snapshot_date(report: Dict[str, Any]) -> Optional[str]:
    dates = []
    for trade in report.get("buy_trades") or []:
        dt = _parse_dt(trade.get("decision_time"))
        if dt:
            dates.append(dt.strftime("%Y-%m-%d"))
    return max(dates) if dates else None


def _archive_outputs(output_dir: Path, archive_date: Optional[str]) -> None:
    if not archive_date:
        return
    archive_dir = output_dir / archive_date
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "buy_trades.csv",
        "latest.json",
        "latest.csv",
        "latest.md",
    ):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)


def run(output_dir: Path = REPORT_ROOT, archive_date: Optional[str] = None) -> Dict[str, Any]:
    snapshots = load_snapshots()
    positions = load_position_records()
    reports = load_decision_reports()
    pnl = load_pnl_snapshots()
    report = evaluate(snapshots, positions, reports, pnl)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_summary_csv(report, output_dir / "benchmark_summary.csv")
    _write_buy_trades_csv(report, output_dir / "buy_trades.csv")
    with (output_dir / "latest.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_summary_csv(report, output_dir / "latest.csv")
    _write_latest_markdown(report, output_dir / "latest.md")
    _archive_outputs(output_dir, archive_date or _latest_snapshot_date(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark metrics from snapshots, positions, and decision reports.")
    parser.add_argument("--output-dir", default=str(REPORT_ROOT))
    args = parser.parse_args()
    report = run(Path(args.output_dir))
    print(f"models={report['model_count']} snapshots={report['snapshot_count']}")
    print(f"wrote={Path(args.output_dir)}")


if __name__ == "__main__":
    main()
