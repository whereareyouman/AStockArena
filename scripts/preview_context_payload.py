#!/usr/bin/env python3
"""
Generate the exact user-context message (including observation snapshot)
that BaseAgent feeds to the LLM, so you can inspect content and estimate tokens
without launching the full trading loop.
"""

import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.base_agent.base_agent import BaseAgent  # pylint: disable=wrong-import-position
from tools.price_tools import get_latest_position  # pylint: disable=wrong-import-position


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "false").lower() in ("1", "true", "yes")


def estimate_tokens(text: str) -> int:
    # Rough heuristic: 1 token ≈ 4 characters for mixed CN/EN text
    return max(1, round(len(text) / 4))


def main() -> None:
    signature = os.environ.get("AGENT_SIGNATURE", "prefetch-tester")
    today = os.environ.get("TODAY_DATE") or datetime.now().strftime("%Y-%m-%d")
    current_time = os.environ.get("CURRENT_TIME") or f"{today} 09:30:00"
    decision_count = int(os.environ.get("DECISION_COUNT", "1"))
    init_date = os.environ.get("INIT_DATE") or today

    agent = BaseAgent(
        signature=signature,
        basemodel="noop",
        news_csv_path="./data/news.csv",
        log_path="./data/agent_data",
        init_date=init_date,
        force_replay=_truthy_env("FORCE_REPLAY"),
    )
    agent.runtime_context.update(
        {"TODAY_DATE": today, "CURRENT_TIME": current_time, "DECISION_COUNT": decision_count}
    )

    agent._prefetched_news.clear()
    agent._prefetched_prices.clear()
    agent._prefetched_indicators.clear()

    agent._prefetch_all_news(today, current_time, max_retries=1)
    agent._prefetch_all_prices(current_time)
    agent._prefetch_all_indicators(today, current_time)

    try:
        latest_positions, _, latest_record = get_latest_position(today, signature)
    except Exception as exc:
        print(f"⚠️ Failed to load latest positions: {exc}")
        latest_positions = {}
        latest_record = None

    metrics = agent._compute_portfolio_metrics(latest_positions, today, current_time)
    holdings_lines = []
    for holding in metrics.get("holdings", []):
        sym = holding["symbol"]
        shares = holding["shares"]
        avg_price = holding.get("avg_price")
        current_price = holding.get("current_price")
        market_value = holding.get("market_value", 0.0)
        unrealized = holding.get("unrealized")
        line = f"  • {sym}: {shares} shares"
        if current_price is not None:
            line += f", Px ¥{current_price:,.2f}"
        if avg_price is not None:
            line += f", Avg ¥{avg_price:,.2f}"
        line += f", MV ¥{market_value:,.2f}"
        if unrealized is not None:
            line += f", PnL ¥{unrealized:,.2f}"
        holdings_lines.append(line)
    if not holdings_lines:
        holdings_lines.append("  • (no equity positions)")

    stage = f"Decision {decision_count}"

    observation_summary = agent._build_observation_summary()
    positions_json = json.dumps(latest_positions, ensure_ascii=False, indent=2)
    last_action = latest_record.get("this_action") if latest_record else None

    context_message = (
        f"Trading context for {signature}:\n"
        f"- Date: {today}\n"
        f"- Time: {current_time}\n"
        f"- Decision index: {decision_count}/3 — stage: {stage}\n"
        f"- Latest recorded action: {json.dumps(last_action, ensure_ascii=False) if last_action else 'N/A'}\n"
        f"- Cash: ¥{metrics['cash']:,.2f}\n"
        f"- Position value: ¥{metrics['position_value']:,.2f}\n"
        f"- Total equity: ¥{metrics['total_equity']:,.2f}\n"
        f"- Unrealized PnL: ¥{metrics['unrealized_total']:,.2f}\n"
        f"- Holdings detail:\n" + "\n".join(holdings_lines) + "\n"
        f"- Observation snapshot (last 3 days news + latest hourly data):\n{observation_summary}\n"
        f"- Full positions JSON (from position file, do not trim):\n{positions_json}\n\n"
        "Please analyze current signals, adjust positions if needed, obey all rules, "
        "and provide a concise plan before calling tools or trades. Conclude with the required stop signal."
    )

    print(context_message)
    print("\n--- payload stats ---")
    print(f"Characters : {len(context_message):,}")
    print(f"Token est.  : ~{estimate_tokens(context_message):,} tokens")


if __name__ == "__main__":
    main()

