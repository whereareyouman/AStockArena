#!/usr/bin/env python3
"""
Standalone prefetch test.
Runs the BaseAgent prefetch helpers without starting main.py to verify
that every whitelisted symbol has news, price, and indicator data.
"""

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.base_agent.base_agent import BaseAgent  # pylint: disable=wrong-import-position
from tools.price_tools import normalize_symbol  # pylint: disable=wrong-import-position


def main() -> None:
    today = os.environ.get("TODAY_DATE") or datetime.now().strftime("%Y-%m-%d")
    current_time = os.environ.get("CURRENT_TIME") or f"{today} 10:30:00"
    decision_count = int(os.environ.get("DECISION_COUNT", "1"))

    agent = BaseAgent(
        signature="prefetch-tester",
        basemodel="noop",
        news_csv_path="./data/news.csv",
        log_path="./data/agent_data",
        init_date=today,
    )
    agent.runtime_context["TODAY_DATE"] = today
    agent.runtime_context["CURRENT_TIME"] = current_time
    agent.runtime_context["DECISION_COUNT"] = decision_count

    def build_snapshot():
        return agent._collect_prefetch_bundle(today, current_time, decision_count)

    snapshot_result = agent.prefetch_coordinator.ensure_snapshot(
        today_date=today,
        current_time=current_time,
        symbols_signature=agent._symbols_signature(),
        builder=build_snapshot,
    )
    bundle = snapshot_result.data
    agent._apply_prefetch_bundle(bundle)

    print(f"Prefetch audit for {today} {current_time}")
    print("-" * 60)
    print(
        f"Snapshot id={bundle.get('snapshot_id')} created_now={snapshot_result.created} "
        f"path={snapshot_result.path}"
    )
    for sym in agent.stock_symbols:
        normalized = normalize_symbol(sym)
        news_count = len(bundle.get("news", {}).get(normalized, {}).get("news", []))
        price_ready = "Yes" if bundle.get("prices", {}).get(normalized) else "No"
        indicator_ready = "Yes" if bundle.get("indicators", {}).get(normalized) else "No"
        print(
            f"{normalized}: news={news_count} (last 3d titles) | "
            f"price={price_ready} | indicators={indicator_ready}"
        )


if __name__ == "__main__":
    main()

