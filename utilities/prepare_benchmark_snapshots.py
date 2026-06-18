import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from agent_engine.agent.agent import AgenticWorkflow


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").lower() in ("1", "true", "yes")


async def prepare_benchmark_snapshots(
    config: Dict[str, Any],
    agent_class: Type[AgenticWorkflow] = AgenticWorkflow,
    stock_symbols: Optional[List[str]] = None,
) -> int:
    """Build all shared market snapshots before benchmark agents start."""
    date_range = config.get("date_range", {})
    init_date = date_range.get("init_date")
    end_date = date_range.get("end_date")
    if not init_date or not end_date:
        raise ValueError("date_range.init_date/end_date must be configured")

    agent_config = config.get("agent_config", {})
    data_config = config.get("data_config", {})
    log_config = config.get("log_config", {})
    trading_rules = config.get("trading_rules", {})
    risk_management = config.get("risk_management", {})
    run_config = config.get("run_config", {})
    if "snapshot_hourly_cache_days" in run_config:
        import os

        os.environ["SNAPSHOT_HOURLY_CACHE_DAYS"] = str(run_config.get("snapshot_hourly_cache_days"))
    enabled_model = next((m for m in config.get("models", []) if m.get("enabled", True)), {})

    agent = agent_class(
        signature="snapshot-preparer",
        basemodel=enabled_model.get("basemodel", "snapshot-preparer"),
        stock_symbols=stock_symbols or AgenticWorkflow.DEFAULT_STOCK_SYMBOLS,
        stock_json_path=data_config.get("stock_json_path", "./data_flow/ai_stock_data.json"),
        news_csv_path=data_config.get("news_csv_path", "./data_flow/news.csv"),
        macro_csv_path=None,
        log_path=log_config.get("log_path", "./data_flow/trading_summary_each_agent"),
        openai_base_url=enabled_model.get("openai_base_url"),
        openai_api_key="snapshot-preparer",
        max_steps=agent_config.get("max_steps", 10),
        max_retries=agent_config.get("max_retries", 3),
        base_delay=agent_config.get("base_delay", 0.5),
        initial_cash=agent_config.get("initial_cash", 1000000.0),
        init_date=init_date,
        trading_rules=trading_rules,
        risk_management=risk_management,
        force_replay=False,
    )
    agent._persist_stock_data_during_snapshot = bool(run_config.get("persist_stock_data_during_snapshot", False))

    prepared = 0
    current = datetime.strptime(init_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        today = current.strftime("%Y-%m-%d")
        if agent._is_trading_day(today):
            for idx, decision_time in enumerate(agent._get_trading_hours(today), 1):
                symbols_signature = agent._symbols_signature()

                def _builder(today=today, decision_time=decision_time, idx=idx) -> Dict[str, Any]:
                    bundle = agent._collect_prefetch_bundle(today, decision_time, idx)
                    bundle.pop("observation_summary", None)
                    return bundle

                result = agent.prefetch_coordinator.ensure_snapshot(
                    today_date=today,
                    current_time=decision_time,
                    symbols_signature=symbols_signature,
                    builder=_builder,
                )
                prepared += 1
                status = "created" if result.created else "cached"
                print(f"✅ snapshot {status}: {today} {decision_time.split()[-1]}")
        current += timedelta(days=1)

    return prepared


def should_prepare_snapshots() -> bool:
    """Default on for benchmark runs; can be disabled for debugging."""
    import os

    return not _truthy(os.getenv("SKIP_SNAPSHOT_PREPARE"))
