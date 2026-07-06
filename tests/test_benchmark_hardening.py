import json
import csv
import asyncio
import threading
import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.tools import tool

from benchmark.benchmark_evaluator import (
    SnapshotPoint,
    evaluate,
    _iter_executed_buys,
    _write_buy_trades_csv,
    _write_summary_csv,
)
import main
from agent_engine.agent.agent import AgenticWorkflow, OrderedTradingToolNode, serialized_position_tool
from agent_engine.shared_prefetch import SharedPrefetchCoordinator
from prompt_templates.prompts import STOP_SIGNAL, get_agent_system_prompt
import utilities.live_scheduler as live_scheduler
import utilities.prefetch_historical_news as prefetch_historical_news
import utils.backup_utils as backup_utils
from utils.json_file_manager import JsonFileManager
from utils.news_cache_guard import (
    NewsCacheIntegrityError,
    validate_news_cache_integrity,
    write_news_manifest,
)
from utilities.live_scheduler import _event_for, build_tick_config, next_pending_event, retry_attempts, write_error_artifacts
from utils.position_manager import (
    get_available_sell_shares,
    get_current_position,
    get_position_file_path,
    normalize_positions,
    remove_shares_from_lots,
    summarize_lots,
    upsert_position_record,
)


def test_lots_enforce_t_plus_one_partial_sell():
    entry = {
        "shares": 300,
        "purchase_date": "2026-01-10",
        "avg_price": 10.0,
        "lots": [
            {"shares": 100, "purchase_date": "2026-01-10", "avg_price": 9.0},
            {"shares": 200, "purchase_date": "2026-01-12", "avg_price": 11.0},
        ],
    }

    assert get_available_sell_shares(entry, "2026-01-12") == 100
    remaining = remove_shares_from_lots(entry, 100, "2026-01-12")
    shares, purchase_date, avg_price = summarize_lots(remaining)

    assert shares == 200
    assert purchase_date == "2026-01-12"
    assert avg_price == 11.0


def test_normalize_positions_adds_legacy_lots():
    positions = {"600519": {"shares": 100, "purchase_date": "2026-01-09", "avg_price": 100.0}, "CASH": 1}
    normalized = normalize_positions(positions)

    assert normalized["SH600519"]["lots"] == [
        {"shares": 100, "purchase_date": "2026-01-09", "avg_price": 100.0}
    ]


def test_position_file_uses_legacy_read_fallback(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    legacy = project_root / "trading_summary_each_agent" / "model-a" / "position"
    legacy.mkdir(parents=True)
    (legacy / "position.jsonl").write_text(
        json.dumps({"date": "2026-01-09", "id": 1, "positions": {"CASH": 9}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    settings = project_root / "settings"
    settings.mkdir()
    (settings / "default_config.json").write_text(
        json.dumps({"log_config": {"log_path": "./data_flow/trading_summary_each_agent"}}),
        encoding="utf-8",
    )

    # Patch parents lookup by replacing module __file__ with a path inside the fake repo.
    monkeypatch.setattr("utils.position_manager.__file__", str(project_root / "utils" / "position_manager.py"))

    assert get_position_file_path("model-a") == legacy / "position.jsonl"
    positions, _, _ = get_current_position("2026-01-12", "model-a")
    assert positions["CASH"] == 9


def test_current_position_falls_back_across_market_holidays(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    position_dir = project_root / "data_flow" / "trading_summary_each_agent" / "model-a" / "position"
    position_dir.mkdir(parents=True)
    (position_dir / "position.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "date": "2026-04-30",
                        "decision_time": "2026-04-30 14:00:00",
                        "id": 53,
                        "positions": {
                            "CASH": 670413.1536,
                            "SH688111": {"shares": 1200, "purchase_date": "2026-04-17", "avg_price": 248.1},
                        },
                    },
                    ensure_ascii=False,
                ),
                ""
            ]
        ),
        encoding="utf-8",
    )
    settings = project_root / "settings"
    settings.mkdir()
    (settings / "default_config.json").write_text(
        json.dumps({"log_config": {"log_path": "./data_flow/trading_summary_each_agent"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr("utils.position_manager.__file__", str(project_root / "utils" / "position_manager.py"))

    positions, current_id, latest_record = get_current_position("2026-05-06", "model-a")

    assert current_id == 53
    assert latest_record["decision_time"] == "2026-04-30 14:00:00"
    assert positions["CASH"] == 670413.1536
    assert positions["SH688111"]["shares"] == 1200


def test_position_upsert_merges_actions_for_same_decision_time(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    settings = project_root / "settings"
    settings.mkdir(parents=True)
    (settings / "default_config.json").write_text(
        json.dumps({"log_config": {"log_path": "./data_flow/trading_summary_each_agent"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.position_manager.__file__", str(project_root / "utils" / "position_manager.py"))

    upsert_position_record(
        "model-a",
        {
            "date": "2026-03-02",
            "decision_time": "2026-03-02 10:30:00",
            "decision_count": 1,
            "this_action": {"action": "buy", "symbol": "SH688256", "amount": 100},
            "actions": [
                {
                    "action": "buy",
                    "symbol": "SH688256",
                    "amount": 100,
                    "price": 1000,
                    "cost": 100000,
                    "commission": 30,
                }
            ],
            "fills": [
                {
                    "action": "buy",
                    "symbol": "SH688256",
                    "amount": 100,
                    "price": 1000,
                    "cost": 100000,
                    "commission": 30,
                }
            ],
            "positions": {"CASH": 899970, "SH688256": {"shares": 100, "purchase_date": "2026-03-02", "avg_price": 1000}},
        },
    )
    upsert_position_record(
        "model-a",
        {
            "date": "2026-03-02",
            "decision_time": "2026-03-02 10:30:00",
            "decision_count": 1,
            "this_action": {"action": "no_trade", "symbol": "", "amount": 0},
            "actions": [{"action": "no_trade", "symbol": "", "amount": 0}],
            "fills": [],
            "positions": {"CASH": 899970, "SH688256": {"shares": 100, "purchase_date": "2026-03-02", "avg_price": 1000}},
        },
    )
    upsert_position_record(
        "model-a",
        {
            "date": "2026-03-02",
            "decision_time": "2026-03-02 10:30:00",
            "decision_count": 1,
            "this_action": {"action": "no_trade", "symbol": "", "amount": 0},
            "actions": [{"action": "no_trade", "symbol": "", "amount": 0}],
            "fills": [],
            "positions": {"CASH": 899970, "SH688256": {"shares": 100, "purchase_date": "2026-03-02", "avg_price": 1000}},
        },
    )

    position_file = project_root / "data_flow" / "trading_summary_each_agent" / "model-a" / "position" / "position.jsonl"
    rows = [json.loads(line) for line in position_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["this_action"] == {"action": "buy", "symbol": "SH688256", "amount": 100}
    assert [action["action"] for action in rows[0]["actions"]] == ["buy", "no_trade", "no_trade"]
    assert len(rows[0]["fills"]) == 1


def test_position_upsert_does_not_carry_fills_across_rerun_ids(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    settings = project_root / "settings"
    settings.mkdir(parents=True)
    (settings / "default_config.json").write_text(
        json.dumps({"log_config": {"log_path": "./data_flow/trading_summary_each_agent"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.position_manager.__file__", str(project_root / "utils" / "position_manager.py"))

    upsert_position_record(
        "model-a",
        {
            "date": "2026-04-07",
            "decision_time": "2026-04-07 11:30:00",
            "decision_count": 2,
            "ledger_run_id": "old-run",
            "this_action": {"action": "buy", "symbol": "SH688256", "amount": 100},
            "actions": [{"action": "buy", "symbol": "SH688256", "amount": 100}],
            "fills": [{"action": "buy", "symbol": "SH688256", "amount": 100}],
            "positions": {"CASH": 880000, "SH688256": {"shares": 100, "purchase_date": "2026-04-07", "avg_price": 1200}},
        },
    )
    upsert_position_record(
        "model-a",
        {
            "date": "2026-04-07",
            "decision_time": "2026-04-07 11:30:00",
            "decision_count": 2,
            "ledger_run_id": "new-run",
            "this_action": {"action": "no_trade", "symbol": "", "amount": 0},
            "actions": [{"action": "no_trade", "symbol": "", "amount": 0}],
            "fills": [],
            "positions": {"CASH": 1000000},
        },
    )

    position_file = project_root / "data_flow" / "trading_summary_each_agent" / "model-a" / "position" / "position.jsonl"
    rows = [json.loads(line) for line in position_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["ledger_run_id"] == "new-run"
    assert rows[0]["this_action"] == {"action": "no_trade", "symbol": "", "amount": 0}
    assert rows[0]["fills"] == []
    assert rows[0]["positions"] == {"CASH": 1000000}


def test_serialized_position_tool_preserves_parallel_same_turn_trades(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    settings = project_root / "settings"
    settings.mkdir(parents=True)
    (settings / "default_config.json").write_text(
        json.dumps(
            {
                "agent_config": {"initial_cash": 1000000},
                "log_config": {"log_path": "./data_flow/trading_summary_each_agent"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.position_manager.__file__", str(project_root / "utils" / "position_manager.py"))

    class DummyAgent:
        signature = "model-a"

        def __init__(self):
            self._position_tool_lock = threading.RLock()

        @serialized_position_tool
        def trade(self, symbol, amount, cash_delta, delay):
            positions, current_id, latest_record = get_current_position("2026-03-02", self.signature)
            time.sleep(delay)
            positions = json.loads(json.dumps(positions))
            positions["CASH"] = float(positions.get("CASH", 0)) - cash_delta
            positions[symbol] = {
                "shares": int(positions.get(symbol, {}).get("shares", 0)) + amount,
                "purchase_date": "2026-03-02",
                "avg_price": cash_delta / amount,
                "lots": [{"shares": amount, "purchase_date": "2026-03-02", "avg_price": cash_delta / amount}],
            }
            fill = {
                "action": "buy",
                "symbol": symbol,
                "amount": amount,
                "cost": cash_delta,
                "decision_time": "2026-03-02 10:30:00",
                "decision_count": 1,
            }
            record = {
                "date": "2026-03-02",
                "decision_time": "2026-03-02 10:30:00",
                "decision_count": 1,
                "this_action": {"action": "buy", "symbol": symbol, "amount": amount},
                "actions": [fill],
                "fills": [fill],
                "positions": positions,
            }
            if latest_record and latest_record.get("decision_time") == "2026-03-02 10:30:00":
                record["id"] = latest_record.get("id")
            else:
                record["id"] = current_id + 1
            upsert_position_record(self.signature, record)

    agent = DummyAgent()
    barrier = threading.Barrier(2)
    errors = []

    def worker(symbol, amount, cash_delta, delay):
        try:
            barrier.wait()
            agent.trade(symbol, amount, cash_delta, delay)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("SH688271", 100, 10000, 0.05)),
        threading.Thread(target=worker, args=("SH688617", 200, 20000, 0.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    position_file = project_root / "data_flow" / "trading_summary_each_agent" / "model-a" / "position" / "position.jsonl"
    rows = [json.loads(line) for line in position_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["this_action"] == {"action": "multi_trade", "symbol": "", "amount": 0, "actions_count": 2}
    assert [fill["symbol"] for fill in row["fills"]] == ["SH688617", "SH688271"]
    assert row["positions"]["CASH"] == 970000
    assert row["positions"]["SH688271"]["shares"] == 100
    assert row["positions"]["SH688617"]["shares"] == 200


def test_ordered_trading_tool_node_runs_position_tools_in_call_order():
    events = []

    @tool
    def buy_stock(symbol: str) -> str:
        """Mock position-mutating buy tool."""
        events.append(f"start:{symbol}")
        if symbol == "A":
            time.sleep(0.05)
        events.append(f"end:{symbol}")
        return symbol

    node = OrderedTradingToolNode([buy_stock])
    runtime = SimpleNamespace(context=None, store=None, stream_writer=None, execution_info=None, server_info=None)

    result = node._func(
        [
            {"name": "buy_stock", "args": {"symbol": "A"}, "id": "call-1", "type": "tool_call"},
            {"name": "buy_stock", "args": {"symbol": "B"}, "id": "call-2", "type": "tool_call"},
        ],
        {},
        runtime,
    )

    assert events == ["start:A", "end:A", "start:B", "end:B"]
    assert [msg.tool_call_id for msg in result["messages"]] == ["call-1", "call-2"]


def test_ordered_trading_tool_node_async_runs_position_tools_in_call_order():
    events = []

    @tool
    async def sell_stock(symbol: str) -> str:
        """Mock position-mutating sell tool."""
        events.append(f"start:{symbol}")
        if symbol == "A":
            await asyncio.sleep(0.05)
        events.append(f"end:{symbol}")
        return symbol

    async def run_node():
        node = OrderedTradingToolNode([sell_stock])
        runtime = SimpleNamespace(context=None, store=None, stream_writer=None, execution_info=None, server_info=None)
        return await node._afunc(
            [
                {"name": "sell_stock", "args": {"symbol": "A"}, "id": "call-1", "type": "tool_call"},
                {"name": "sell_stock", "args": {"symbol": "B"}, "id": "call-2", "type": "tool_call"},
            ],
            {},
            runtime,
        )

    result = asyncio.run(run_node())

    assert events == ["start:A", "end:A", "start:B", "end:B"]
    assert [msg.tool_call_id for msg in result["messages"]] == ["call-1", "call-2"]


def test_startup_key_validation_routes_by_provider(monkeypatch, tmp_path):
    stock_path = tmp_path / "ai_stock_data.json"
    stock_path.write_text("{}", encoding="utf-8")
    news_path = tmp_path / "news.csv"
    news_path.write_text("symbol,title,content,publish_time,source,url,query,search_time\n", encoding="utf-8")
    config = {
        "date_range": {"init_date": "2026-01-12", "end_date": "2026-01-13"},
        "data_config": {"stock_json_path": str(stock_path), "news_csv_path": str(news_path)},
    }
    models = [
        {"basemodel": "deepseek/deepseek-v3.1-terminus", "openai_base_url": "https://openrouter.ai/api/v1"},
        {"basemodel": "openai/gpt-5.1", "openai_base_url": "https://openrouter.ai/api/v1"},
    ]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        main.validate_benchmark_startup(config, models)
        raised = False
    except SystemExit:
        raised = True
    assert raised

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter")
    main.validate_benchmark_startup(config, models)


def test_date_range_is_config_only(monkeypatch):
    config = {
        "date_range": {"init_date": "2026-06-15", "end_date": "2026-06-16"},
        "run_config": {"realtime_mode": "wait"},
    }

    monkeypatch.setenv("INIT_DATE", "2026-01-12")
    monkeypatch.setenv("END_DATE", "2026-02-12")

    assert config["date_range"]["init_date"] == "2026-06-15"
    assert config["date_range"]["end_date"] == "2026-06-16"
    assert main._configured_realtime_mode(config) == "wait"


def test_backtest_realtime_mode_normalizes_to_empty():
    assert main._configured_realtime_mode({"run_config": {"realtime_mode": "backtest"}}) == ""


def test_realtime_stop_mode_is_realtime_mode():
    assert main._configured_realtime_mode({"run_config": {"realtime_mode": "stop"}}) == "stop"
    assert main._allows_current_or_future_dates("stop") is True
    assert main._allows_current_or_future_dates("wait") is True
    assert main._allows_current_or_future_dates("") is False


def test_backtest_mode_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("BACKTEST_MODE", "true")

    assert main.configure_backtest_mode({"run_config": {"backtest_mode": False}}) == "false"
    assert main.configure_backtest_mode({"run_config": {"backtest_mode": True}}) == "true"


def test_news_prefetch_lookback_can_cross_init_date(monkeypatch, tmp_path):
    calls = []

    def fake_prefetch(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(main, "prefetch_historical_news", fake_prefetch)
    config = {
        "date_range": {"init_date": "2026-06-17", "end_date": "2026-06-17"},
        "run_config": {
            "prefetch_news_before_run": True,
            "prefetch_news_lookback_days": 3,
            "prefetch_news_respect_init_date": False,
        },
        "data_config": {"news_csv_path": str(tmp_path / "news.csv")},
    }

    main.prefetch_configured_news_before_run(config, ["SH688981"])

    assert calls[0]["start_date"] == "2026-06-15"
    assert calls[0]["end_date"] == "2026-06-17"


def test_news_prefetch_can_respect_init_date(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(main, "prefetch_historical_news", lambda **kwargs: calls.append(kwargs))
    config = {
        "date_range": {"init_date": "2026-06-17", "end_date": "2026-06-17"},
        "run_config": {
            "prefetch_news_before_run": True,
            "prefetch_news_lookback_days": 3,
            "prefetch_news_respect_init_date": True,
        },
        "data_config": {"news_csv_path": str(tmp_path / "news.csv")},
    }

    main.prefetch_configured_news_before_run(config, ["SH688981"])

    assert calls[0]["start_date"] == "2026-06-17"


def test_historical_news_prefetch_skips_weekend_calendar(monkeypatch, tmp_path):
    def fail_calendar(_date_value):
        raise AssertionError("calendar events should be skipped on weekends")

    monkeypatch.setattr(prefetch_historical_news, "fetch_akshare_calendar_events", fail_calendar)

    result = prefetch_historical_news.prefetch_historical_news(
        symbols=[],
        start_date="2026-06-06",
        end_date="2026-06-07",
        output_path=tmp_path / "news.csv",
        page_size=50,
        max_pages=0,
        timeout=1,
        sleep_seconds=0,
        use_akshare_stock_news=False,
        use_akshare_calendar_events=True,
    )

    assert result.empty


def test_historical_news_save_dedupes_cross_source_by_title_date(tmp_path):
    rows = [
        {
            "symbol": "SH688981",
            "title": "中芯国际 关于召开业绩说明会的公告",
            "content": "cninfo copy",
            "publish_time": "2026-04-30 09:00:00",
            "source": "cninfo_fulltext",
            "url": "http://cninfo",
            "query": "中芯国际",
            "search_time": "2026-05-01 00:00:00",
        },
        {
            "symbol": "SH688981",
            "title": "《中芯国际》关于召开业绩说明会的公告",
            "content": "sse copy",
            "publish_time": "2026-04-30 08:00:00",
            "source": "sse_announcement",
            "url": "http://sse",
            "query": "上交所公告",
            "search_time": "2026-05-01 00:00:00",
        },
        {
            "symbol": "SH688981",
            "title": "中芯国际 获得新订单",
            "content": "media",
            "publish_time": "2026-04-30 10:00:00",
            "source": "sina_deep",
            "url": "http://sina",
            "query": "688981",
            "search_time": "2026-05-01 00:00:00",
        },
    ]

    saved = prefetch_historical_news._save_news(
        tmp_path / "news.csv",
        pd.DataFrame(columns=prefetch_historical_news.NEWS_COLUMNS),
        rows,
    )

    assert len(saved) == 2
    manifest = json.loads((tmp_path / "news_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 2
    assert manifest["sha256"]
    duplicate = saved[saved["title"].str.contains("业绩说明会", na=False)].iloc[0]
    assert duplicate["source"] == "sse_announcement"
    assert duplicate["url"] == "http://sse"


def test_news_cache_guard_rejects_empty_csv_with_nonempty_manifest(tmp_path):
    news_path = tmp_path / "news.csv"
    news_path.write_text("symbol,title,content,publish_time,source,url,query,search_time\n", encoding="utf-8")
    (tmp_path / "news_manifest.json").write_text(
        json.dumps({"rows": 9176, "size_bytes": 4262708, "sha256": "expected"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = validate_news_cache_integrity(news_path)

    assert result["ok"] is False
    assert any("只有表头" in error for error in result["errors"])
    with pytest.raises(NewsCacheIntegrityError):
        validate_news_cache_integrity(news_path, strict=True)


def test_news_purge_refuses_to_overwrite_nonempty_cache_with_empty_filter(tmp_path):
    news_path = tmp_path / "news.csv"
    original = (
        "symbol,title,content,publish_time,source,url,query,search_time\n"
        "SH600000,other,body,2026-04-01 10:00:00,sina,,,\n"
    )
    news_path.write_text(original, encoding="utf-8")
    write_news_manifest(news_path)
    agent = object.__new__(AgenticWorkflow)
    agent.allowed_symbols = {"SH688981"}

    agent._purge_news_csv(str(news_path))

    assert news_path.read_text(encoding="utf-8") == original


def test_strict_snapshot_mode_accepts_legacy_env(monkeypatch):
    agent = object.__new__(AgenticWorkflow)
    monkeypatch.delenv("BACKTEST_MODE", raising=False)

    monkeypatch.setenv("STRICT_SNAPSHOT_MODE", "false")
    assert agent._strict_snapshot_mode() is False

    monkeypatch.setenv("BACKTEST_MODE", "true")
    assert agent._strict_snapshot_mode() is True


def test_decision_report_parser_handles_unescaped_news_quotes():
    agent = object.__new__(AgenticWorkflow)
    content = '''```json
{
  "decision_evidence_report": {
    "schema_version": 2,
    "observed_universe": ["SH688271"],
    "candidate_review": [
      {
        "symbol": "SH688271",
        "rank": 1,
        "selected_for_action": false,
        "news_evidence_used": [
          {
            "title": "8000万7T磁共振集采，因重大缺陷被"叫停"！",
            "model_interpretation": "集采叫停对医疗设备公司是短期利空",
            "claimed_direction": "negative"
          }
        ],
        "price_evidence_used": {
          "signal_evaluation": {
            "momentum_reading": "bearish",
            "trend_reading": "neutral",
            "risk_reading": "acceptable",
            "momentum_trend_conflict": false,
            "decision_implication": "supports_wait"
          }
        },
        "risk_checks_mentioned": ["news_conflict"],
        "reject_or_hold_reason_text": "等待"
      }
    ],
    "actions_planned_or_taken": [
      {"action": "no_trade", "reason_text": "等待", "risk_controls_cited": ["news_conflict"]}
    ],
    "workflow_trace": {
      "has_candidate_review": true,
      "has_news_evidence": true,
      "has_price_evidence": true,
      "has_risk_checks": true,
      "has_action_reason": true,
      "missing_required_sections": []
    }
  }
}
```'''

    report, error = agent._extract_benchmark_decision_report(content)

    assert error is None
    assert report["candidate_review"][0]["news_evidence_used"][0]["title"] == '8000万7T磁共振集采，因重大缺陷被"叫停"！'


def test_live_wait_defers_startup_data_prep_for_current_or_future_dates():
    assert main._defer_startup_data_prep("wait", date(2026, 6, 17), date(2026, 6, 17)) is True
    assert main._defer_startup_data_prep("wait", date(2026, 6, 18), date(2026, 6, 17)) is True
    assert main._defer_startup_data_prep("wait", date(2026, 6, 16), date(2026, 6, 17)) is False
    assert main._defer_startup_data_prep("stop", date(2026, 6, 18), date(2026, 6, 17)) is False


def test_live_scheduler_builds_single_decision_model_config():
    base_config = {
        "date_range": {"init_date": "2026-01-01", "end_date": "2026-01-02"},
        "run_config": {
            "parallel_run": True,
            "parallel_spawn_delay_seconds": 2,
            "realtime_mode": "wait",
            "prefetch_news_before_run": True,
        },
        "models": [
            {"name": "m1", "signature": "m1", "enabled": True},
            {"name": "m2", "signature": "m2", "enabled": True},
        ],
    }
    event = _event_for("2026-06-18", "10:30:00", 1)

    tick_config = build_tick_config(base_config, event, base_config["models"][1], live_backtest_mode=False)

    assert tick_config["date_range"] == {"init_date": "2026-06-18", "end_date": "2026-06-18"}
    assert tick_config["run_config"]["parallel_run"] is False
    assert tick_config["run_config"]["parallel_spawn_delay_seconds"] == 2
    assert tick_config["run_config"]["realtime_mode"] == "stop"
    assert tick_config["run_config"]["prefetch_news_before_run"] is False
    assert tick_config["run_config"]["backtest_mode"] is False
    assert [model["signature"] for model in tick_config["models"]] == ["m2"]


def test_live_scheduler_news_prefetch_overrides_child_skip(monkeypatch):
    calls = []

    def fake_prefetch(config, symbols):
        calls.append((config, symbols))

    monkeypatch.setattr(live_scheduler.trading_main, "prefetch_configured_news_before_run", fake_prefetch)
    config = {
        "run_config": {
            "live_prefetch_news_before_decision": True,
            "prefetch_news_before_run": False,
        }
    }
    event = _event_for("2026-06-18", "10:30:00", 1)

    assert live_scheduler.prefetch_news_for_event(config, event) is True

    assert calls
    assert calls[0][0]["run_config"]["prefetch_news_before_run"] is True
    assert calls[0][0]["date_range"] == {"init_date": "2026-06-18", "end_date": "2026-06-18"}


def test_live_scheduler_can_skip_news_prefetch_for_backfill(monkeypatch):
    monkeypatch.setenv("ASTOCK_SKIP_EVENT_NEWS_PREFETCH", "1")

    def fail_prefetch(_config, _symbols):
        raise AssertionError("backfill should use existing news cache")

    monkeypatch.setattr(live_scheduler.trading_main, "prefetch_configured_news_before_run", fail_prefetch)
    config = {"run_config": {"live_prefetch_news_before_decision": True}}
    event = _event_for("2026-06-18", "10:30:00", 1)

    assert live_scheduler.prefetch_news_for_event(config, event) is True


def test_live_scheduler_finds_next_pending_decision():
    config = {"run_config": {"live_decision_times": ["10:30:00"], "live_scheduler_catchup_minutes": 20}}
    event = next_pending_event(config, {"events": {}}, datetime(2026, 6, 17, 10, 31))

    assert event is not None
    assert event.decision_time == "2026-06-17 10:30:00"


def test_live_scheduler_terminal_event_statuses_are_not_rerun():
    config = {
        "run_config": {
            "live_decision_times": ["10:30:00", "11:30:00"],
            "live_scheduler_catchup_minutes": 20,
        }
    }
    state = {
        "events": {
            "2026-06-17#1#2026-06-17 10:30:00": {
                "status": "partial_failed",
                "date": "2026-06-17",
                "decision_time": "2026-06-17 10:30:00",
                "decision_count": 1,
            }
        }
    }

    event = next_pending_event(config, state, datetime(2026, 6, 17, 10, 31))

    assert event is not None
    assert event.decision_time == "2026-06-17 11:30:00"

    failed_state = {
        "events": {
            "2026-06-17#1#2026-06-17 10:30:00": {
                "status": "failed",
                "date": "2026-06-17",
                "decision_time": "2026-06-17 10:30:00",
                "decision_count": 1,
            }
        }
    }
    failed_event = next_pending_event(config, failed_state, datetime(2026, 6, 17, 10, 31))

    assert failed_event is not None
    assert failed_event.decision_time == "2026-06-17 11:30:00"


def test_live_scheduler_error_artifacts_include_human_csv(tmp_path):
    event = _event_for("2026-06-18", "10:30:00", 1)
    config = {"run_config": {"live_error_dir": str(tmp_path)}}

    error_dir = write_error_artifacts(
        config,
        event,
        stage="news_prefetch",
        model_signature="gemini-2.5-flash",
        symbol="SH688981",
        attempts=3,
        error=TimeoutError("新浪新闻请求超时"),
        final_action="continue_with_existing_news_cache",
    )

    rows = list(csv.DictReader((error_dir / "error_summary.csv").open(encoding="utf-8-sig")))
    assert rows[0]["date"] == "2026-06-18"
    assert rows[0]["decision_time"] == "2026-06-18 10:30:00"
    assert rows[0]["symbol"] == "SH688981"
    assert rows[0]["attempts"] == "3"
    assert "SH688981 在 2026-06-18 10:30:00 第1次操盘的 news_prefetch 阶段发生 TimeoutError" in rows[0]["human_message"]
    assert (error_dir / "error_details.jsonl").exists()
    assert any(path.name.startswith("traceback_") for path in error_dir.iterdir())


def test_live_scheduler_retries_at_least_three_and_writes_error(monkeypatch, tmp_path):
    event = _event_for("2026-06-18", "10:30:00", 1)
    config = {"run_config": {"live_error_dir": str(tmp_path), "live_retry_attempts": 1, "live_retry_backoff_seconds": [0]}}
    attempts = {"count": 0}
    monkeypatch.setattr("utilities.live_scheduler.time.sleep", lambda _seconds: None)

    def always_fails(_attempt):
        attempts["count"] += 1
        raise RuntimeError("boom")

    ok, _, error_dir = live_scheduler._run_with_retries(
        config,
        event,
        stage="snapshot_prepare",
        symbol="ALL",
        final_action_on_failure="event_failed",
        func=always_fails,
    )

    assert retry_attempts(config) == 3
    assert ok is False
    assert attempts["count"] == 3
    assert error_dir is not None
    assert (error_dir / "error_summary.csv").exists()


def test_live_scheduler_snapshot_failure_aborts_event(monkeypatch, tmp_path):
    event = _event_for("2026-06-18", "10:30:00", 1)
    config = {
        "run_config": {
            "live_error_dir": str(tmp_path),
            "live_retry_backoff_seconds": [0],
            "live_abort_on_snapshot_failure": True,
        },
        "models": [{"signature": "model-a", "enabled": True}],
    }
    monkeypatch.setattr("utilities.live_scheduler.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("utilities.live_scheduler.prefetch_news_for_event", lambda _config, _event: True)

    def fail_snapshot(_config, _event):
        raise live_scheduler.SnapshotValidationError("missing price")

    monkeypatch.setattr("utilities.live_scheduler.prepare_snapshot_for_event", fail_snapshot)

    result = live_scheduler.run_event(config, event)

    assert result["status"] == "failed"
    assert result["error"] == "snapshot_prepare_failed"
    assert result["models"] == {}
    error_dir = Path(result["error_dirs"][0])
    rows = list(csv.DictReader((error_dir / "error_summary.csv").open(encoding="utf-8-sig")))
    assert rows[-1]["stage"] == "snapshot_prepare"
    assert rows[-1]["final_action"] == "event_failed"


def test_live_scheduler_single_model_failure_keeps_other_models(monkeypatch, tmp_path):
    event = _event_for("2026-06-18", "10:30:00", 1)
    config = {
        "run_config": {"live_error_dir": str(tmp_path), "live_retry_backoff_seconds": [0]},
        "models": [
            {"signature": "good-model", "enabled": True},
            {"signature": "bad-model", "enabled": True},
        ],
    }
    monkeypatch.setattr("utilities.live_scheduler.prefetch_news_for_event", lambda _config, _event: True)
    monkeypatch.setattr("utilities.live_scheduler.prepare_snapshot_for_event", lambda _config, _event: {"path": "/tmp/snapshot.json"})
    monkeypatch.setattr("utilities.live_scheduler.run_benchmark_for_event", lambda _config, _event: {"status": "succeeded"})

    def fake_model_run(_config, _event, model, **_kwargs):
        signature = model["signature"]
        if signature == "bad-model":
            return {"signature": signature, "status": "failed", "error_dir": str(tmp_path / "bad"), "error_message": "api down"}
        return {"signature": signature, "status": "succeeded", "returncode": 0, "attempts": 1}

    monkeypatch.setattr("utilities.live_scheduler.run_model_with_retries", fake_model_run)

    result = live_scheduler.run_event(config, event)

    assert result["status"] == "partial_failed"
    assert result["models"]["good-model"]["status"] == "succeeded"
    assert result["models"]["bad-model"]["status"] == "failed"


def test_live_scheduler_model_parallelism_limits_workers(monkeypatch, tmp_path):
    event = _event_for("2026-06-18", "10:30:00", 1)
    config = {
        "run_config": {
            "live_error_dir": str(tmp_path),
            "live_retry_backoff_seconds": [0],
            "live_scheduler_model_parallelism": 2,
        },
        "models": [{"signature": f"model-{idx}", "enabled": True} for idx in range(4)],
    }
    monkeypatch.setattr("utilities.live_scheduler.prefetch_news_for_event", lambda _config, _event: True)
    monkeypatch.setattr("utilities.live_scheduler.prepare_snapshot_for_event", lambda _config, _event: {"path": "/tmp/snapshot.json"})
    monkeypatch.setattr("utilities.live_scheduler.run_benchmark_for_event", lambda _config, _event: {"status": "succeeded"})

    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_model_run(_config, _event, model, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"signature": model["signature"], "status": "succeeded", "returncode": 0, "attempts": 1}

    monkeypatch.setattr("utilities.live_scheduler.run_model_with_retries", fake_model_run)

    result = live_scheduler.run_event(config, event)

    assert result["status"] == "succeeded"
    assert result["model_parallelism"] == 2
    assert len(result["models"]) == 4
    assert max_active == 2


def test_live_scheduler_model_start_delay_uses_spawn_delay_fallback():
    config = {"run_config": {"parallel_spawn_delay_seconds": 2.5}}
    assert live_scheduler.model_start_delay_seconds(config) == 2.5

    override_config = {
        "run_config": {
            "parallel_spawn_delay_seconds": 2.5,
            "live_scheduler_model_start_delay_seconds": 0.75,
        }
    }
    assert live_scheduler.model_start_delay_seconds(override_config) == 0.75


def test_benchmark_evaluator_outputs_core_metrics(tmp_path):
    snapshots = [
        SnapshotPoint(
            decision_time="2026-01-12 10:30:00",
            dt=datetime(2026, 1, 12, 10, 30),
            path=tmp_path / "s1.json",
            prices={"SH600519": 10.0, "SH000001": 10.0},
            indicators={},
            news={},
        ),
        SnapshotPoint(
            decision_time="2026-01-12 11:30:00",
            dt=datetime(2026, 1, 12, 11, 30),
            path=tmp_path / "s2.json",
            prices={"SH600519": 9.0, "SH000001": 11.0},
            indicators={},
            news={},
        ),
    ]
    position_records = {
        "model-a": [
            {
                "decision_time": "2026-01-12 10:30:00",
                "this_action": {"action": "buy", "symbol": "SH600519", "amount": 100, "reason_text": "positive news"},
                "positions": {"SH600519": {"avg_price": 10.0, "lots": [{"avg_price": 10.0, "shares": 100}]}},
            }
        ]
    }
    report_rows = {
        "model-a": {
            "2026-01-12 10:30:00": {
                "parse_success": True,
                "schema_valid": True,
                "report": {
                    "candidate_review": [
                        {
                            "symbol": "SH600519",
                            "news_evidence_used": [
                                {"claimed_direction": "positive"},
                                {"claimed_direction": "mixed"},
                            ],
                            "price_evidence_used": {"rsi_3": 75.0, "recent_change_pct": 2.5},
                        }
                    ],
                    "actions_planned_or_taken": [{"action": "buy"}],
                },
            }
        }
    }
    pnl_rows = {
        "model-a": [
            {"decision_time": "2026-01-12 10:30:00", "unrealized_equity": 1_000_000.0},
            {"decision_time": "2026-01-12 11:30:00", "unrealized_equity": 999_000.0},
        ]
    }

    report = evaluate(snapshots, position_records, report_rows, pnl_rows)

    metrics = report["models"]["model-a"]
    assert metrics["outcome"]["total_return_pct"] == pytest.approx(-0.1)
    assert metrics["action_quality"]["buy_count"] == 1
    assert metrics["action_quality"]["buy_hit_rate"] == 0.0
    assert metrics["fin_snr"]["fin_snr_failure_count"] == 1
    assert metrics["fin_snr"]["news_conflict_failure_count"] == 1
    assert metrics["fin_snr"]["overheated_positive_news_failure_count"] == 1
    assert report["buy_trades"][0]["estimated_loss"] == pytest.approx(100.0)

    _write_summary_csv(report, tmp_path / "summary.csv")
    _write_buy_trades_csv(report, tmp_path / "buys.csv")
    assert "model-a" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "SH600519" in (tmp_path / "buys.csv").read_text(encoding="utf-8")
    summary_rows = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert [row["equity_basis"] for row in summary_rows] == ["realized", "unrealized"]
    assert summary_rows[0]["price_confirmation_breach_count"] == "1"
    buy_rows = list(csv.DictReader((tmp_path / "buys.csv").open(encoding="utf-8")))
    assert buy_rows[0]["price_confirmation_breach"] == "True"


def test_benchmark_evaluator_uses_fills_for_multi_trade_buys():
    records = [
        {
            "decision_time": "2026-01-12 10:30:00",
            "this_action": {"action": "multi_trade", "symbol": "", "amount": 0, "actions_count": 3},
            "fills": [
                {"action": "sell", "symbol": "SH688111", "amount": 100},
                {"action": "buy", "symbol": "SH688256", "amount": 200},
                {"action": "buy", "symbol": "SH688981", "amount": 300},
            ],
            "positions": {},
        }
    ]

    buys = list(_iter_executed_buys(records))

    assert [buy[1]["symbol"] for buy in buys] == ["SH688256", "SH688981"]
    assert [buy[1]["amount"] for buy in buys] == [200, 300]


def test_save_pnl_snapshot_is_transactional_under_parallel_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(backup_utils, "PROJECT_ROOT", tmp_path)
    settings = tmp_path / "settings"
    settings.mkdir(parents=True)
    (settings / "default_config.json").write_text(
        json.dumps({"agent_config": {"initial_cash": 1000000.0}}),
        encoding="utf-8",
    )
    data_flow = tmp_path / "data_flow"
    (data_flow / "trading_summary_each_agent" / "model-a" / "position").mkdir(parents=True)
    (data_flow / "trading_summary_each_agent" / "model-a" / "position" / "position.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "date": "2026-01-12",
                "decision_time": "2026-01-12 10:30:00",
                "decision_count": 1,
                "positions": {
                    "CASH": 990000.0,
                    "SH600519": {
                        "shares": 100,
                        "purchase_date": "2026-01-12",
                        "avg_price": 100.0,
                        "lots": [{"shares": 100, "purchase_date": "2026-01-12", "avg_price": 100.0}],
                    },
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_flow / "ai_stock_data.json").write_text(
        json.dumps(
            {
                "SH600519": {
                    "小时线行情": [
                        {"time": "2026-01-12 10:00:00", "close": 110.0},
                        {"time": "2026-01-12 11:00:00", "close": 120.0},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    errors = []

    def worker():
        try:
            assert backup_utils.save_pnl_snapshot("parallel-test") is True
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    pnl_file = data_flow / "pnl_snapshots" / "pnl_model-a.json"
    rows = json.loads(pnl_file.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["decision_time"] == "2026-01-12 10:30:00"
    assert rows[0]["realized_equity"] == pytest.approx(1000000.0)
    assert rows[0]["unrealized_equity"] == pytest.approx(1001000.0)


def test_json_file_manager_safe_write_json_is_concurrency_safe(tmp_path):
    manager = JsonFileManager(max_retries=20, retry_delay=0.01)
    target = tmp_path / "shared.json"
    errors = []

    def writer(idx):
        payload = {"writer": idx, "rows": list(range(100)), "text": "x" * 1000}
        try:
            assert manager.safe_write_json(str(target), payload, backup=False) is True
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(idx,)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["writer"] in range(8)
    assert data["rows"] == list(range(100))
    assert not list(tmp_path.glob("shared.json.tmp"))
    assert not list(tmp_path.glob("shared.json.tmp.*"))


def test_decision_run_artifacts_include_readable_markdown(tmp_path):
    agent = AgenticWorkflow.__new__(AgenticWorkflow)
    agent.signature = "test-model"
    agent.base_log_path = str(tmp_path)

    report_entry = {
        "parse_success": True,
        "schema_valid": True,
        "report": {
            "actions_planned_or_taken": [
                {
                    "action": "no_trade",
                    "symbol": "",
                    "amount": 0,
                    "reason_text": "No high-confidence setup.",
                }
            ]
        },
    }
    full_entry = {
        "messages": [
            {"type": "human", "content": "input"},
            {"type": "ai", "content": "Observation Summary:\n\n1. SH600519\n   - 技术指标: ok\n```json\n{}"},
        ]
    }
    tool_entry = {
        "by_tool": {"get_technical_indicators": {"count": 1, "success": 1, "failed": 0, "unknown": 0}},
        "calls": [{"name": "get_technical_indicators", "success": True}],
    }

    agent._write_decision_run_artifacts(
        today_date="2026-06-16",
        decision_time="2026-06-16 14:00:00",
        decision_count=3,
        input_payload={
            "cash": 1000,
            "total_equity": 1000,
            "snapshot": {"snapshot_path": "/tmp/snapshot.json"},
        },
        full_conversation_entry=full_entry,
        tool_metrics_entry=tool_entry,
        report_log_entry=report_entry,
        final_agent_summary=full_entry["messages"][1]["content"],
        tool_summary="ok",
        collected_tool_errors=[],
        handled_trading_result=True,
    )

    run_dir = tmp_path / "test-model" / "runs" / "2026-06-16" / "14-00-00"
    assert (run_dir / "input.json").exists()
    assert (run_dir / "conversation.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert json.loads((run_dir / "decision_report.json").read_text(encoding="utf-8"))["schema_valid"] is True
    readable = (run_dir / "readable.md").read_text(encoding="utf-8")
    assert "# test-model | 2026-06-16 14:00:00" in readable
    assert "`get_technical_indicators`" in readable
    assert "`no_trade`" in readable
    assert "摘要视图" in readable
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    by_name = {item["filename"]: item for item in manifest["artifacts"]}
    assert by_name["conversation.jsonl"]["complete_raw"] is True
    assert by_name["readable.md"]["summary_view"] is True


def test_decision_readable_uses_ledger_fills_for_executed_decision(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    settings = project_root / "settings"
    settings.mkdir(parents=True)
    (settings / "default_config.json").write_text(
        json.dumps({"log_config": {"log_path": "./data_flow/trading_summary_each_agent"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.position_manager.__file__", str(project_root / "utils" / "position_manager.py"))

    position_dir = project_root / "data_flow" / "trading_summary_each_agent" / "test-model" / "position"
    position_dir.mkdir(parents=True)
    (position_dir / "position.jsonl").write_text(
        json.dumps(
            {
                "id": 7,
                "date": "2026-06-16",
                "decision_time": "2026-06-16 14:00:00",
                "this_action": {"action": "sell", "symbol": "SH688981", "amount": 300},
                "actions": [{"action": "sell", "symbol": "SH688981", "amount": 300}],
                "fills": [
                    {
                        "action": "sell",
                        "symbol": "SH688981",
                        "amount": 300,
                        "price": 100,
                        "commission": 9,
                        "stamp_duty": 30,
                    }
                ],
                "positions": {"CASH": 1000},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    agent = AgenticWorkflow.__new__(AgenticWorkflow)
    agent.signature = "test-model"
    agent.base_log_path = str(tmp_path / "runs")

    agent._write_decision_run_artifacts(
        today_date="2026-06-16",
        decision_time="2026-06-16 14:00:00",
        decision_count=3,
        input_payload={"cash": 1000, "total_equity": 1000, "snapshot": {"snapshot_path": "/tmp/snapshot.json"}},
        full_conversation_entry={"messages": []},
        tool_metrics_entry={"by_tool": {}, "calls": []},
        report_log_entry={
            "parse_success": True,
            "schema_valid": True,
            "report": {
                "actions_planned_or_taken": [
                    {"action": "buy", "symbol": "SH688008", "amount": 100, "reason_text": "model plan"}
                ]
            },
        },
        final_agent_summary="Observation Summary:\n\nok",
        tool_summary="ok",
        collected_tool_errors=[],
        handled_trading_result=True,
    )

    readable = (
        tmp_path
        / "runs"
        / "test-model"
        / "runs"
        / "2026-06-16"
        / "14-00-00"
        / "readable.md"
    ).read_text(encoding="utf-8")
    assert "## Decision\n- `sell` SH688981 x300 @ 100" in readable
    assert "## Model-Stated Decision\n- `buy` SH688008 x100: model plan" in readable
    execution = json.loads(
        (
            tmp_path
            / "runs"
            / "test-model"
            / "runs"
            / "2026-06-16"
            / "14-00-00"
            / "execution.json"
        ).read_text(encoding="utf-8")
    )
    assert execution["ledger_record_found"] is True
    assert execution["executed_fills"][0]["symbol"] == "SH688981"


def test_snapshot_price_used_for_portfolio_metrics():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    agent._prefetched_prices = {"SH600519": {"summary": {"close": 12.5, "timestamp": "2026-01-12 10:30:00"}}}

    metrics = agent._compute_portfolio_metrics(
        {"CASH": 100.0, "SH600519": {"shares": 100, "avg_price": 10.0, "purchase_date": "2026-01-10"}},
        "2026-01-12",
        "2026-01-12 10:30:00",
    )

    assert metrics["position_value"] == 1250.0
    assert metrics["total_equity"] == 1350.0


def test_full_conversation_log_preserves_tool_messages(tmp_path):
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    log_file = tmp_path / "conversation.jsonl"
    long_tool_payload = json.dumps({"success": True, "rows": ["x" * 1000]}, ensure_ascii=False)

    agent._log_full_conversation(
        str(log_file),
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "buy_stock", "args": {"amount": 100}}]},
                {"role": "tool", "name": "buy_stock", "tool_call_id": "abc", "content": long_tool_payload},
            ]
        },
        decision_time="2026-01-12 10:30:00",
        decision_count=1,
    )

    entry = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert entry["event_type"] == "full_conversation"
    assert entry["messages"][2]["content"] == long_tool_payload


def test_report_content_found_before_final_stop_signal():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    report_text = """```json
{"decision_evidence_report":{"schema_version":2,"observed_universe":["SH600519"],"candidate_review":[],"actions_planned_or_taken":[],"workflow_trace":{}}}
```"""
    conversation = {
        "messages": [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": report_text, "tool_calls": [{"name": "buy_stock"}]},
            {"role": "tool", "name": "buy_stock", "content": json.dumps({"success": True})},
            {"role": "assistant", "content": "<FINISH_SIGNAL>"},
        ]
    }

    assert agent._latest_decision_report_content_from_conversation(conversation) == report_text


def test_snapshot_path_uses_short_signature_hash(tmp_path):
    coordinator = SharedPrefetchCoordinator(base_dir=str(tmp_path / "shared"))
    long_signature = "|".join([f"SH688{i:03d}" for i in range(100)])

    path = coordinator._snapshot_path("2026-06-08", "2026-06-08 10:30:00", long_signature)

    assert len(path.name) < 80
    assert "|" not in path.name
    assert path.name.startswith("2026-06-08_10-30-00_sig-")


def test_tool_call_metrics_jsonl(tmp_path):
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
        log_path=str(tmp_path),
    )
    conversation = {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "name": "buy_stock"}]},
            {"role": "tool", "name": "buy_stock", "tool_call_id": "call-1", "content": json.dumps({"success": True})},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2", "name": "sell_stock"}]},
            {"role": "tool", "name": "sell_stock", "tool_call_id": "call-2", "content": json.dumps({"error": "持股数量不足"})},
        ]
    }

    agent._log_tool_call_metrics(
        conversation,
        elapsed_seconds=1.2345,
        decision_time="2026-01-12 10:30:00",
        decision_count=1,
    )

    metrics_file = tmp_path / "test-agent" / "metrics" / "tool_call_metrics.jsonl"
    entry = json.loads(metrics_file.read_text(encoding="utf-8").strip())
    assert entry["total_tool_calls"] == 2
    assert entry["session_elapsed_seconds"] == 1.234
    assert entry["by_tool"]["buy_stock"]["success"] == 1
    assert entry["by_tool"]["sell_stock"]["failed"] == 1


def test_hourly_tool_returns_real_snapshot_candles():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    agent._prefetched_prices = {
        "SH600519": {
            "summary": {"close": 12.0, "timestamp": "2026-01-12 14:00:00"},
            "hourly_3d": {
                "candles": [
                    {"timestamp": "2026-01-12 10:30:00", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
                    {"timestamp": "2026-01-12 11:30:00", "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 120},
                    {"timestamp": "2026-01-12 14:00:00", "open": 11.5, "high": 12.2, "low": 11, "close": 12, "volume": 130},
                ]
            },
        }
    }

    payload = json.loads(agent.get_hourly_stock_data("SH600519", "2026-01-12 14:00:00", 72))

    assert payload["source"] == "snapshot_hourly_3d"
    assert payload["granularity"] == "60min"
    assert payload["total_candles_available"] == 3
    assert payload["candles"][0]["open"] == 10
    assert payload["summary"]["change_pct"] > 0


def test_compact_snapshot_omits_hourly_candles_from_prompt():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    compact = agent._compact_snapshot_for_llm({
        "snapshot_id": "snap-1",
        "schema_version": 1,
        "today_date": "2026-01-12",
        "decision_time": "2026-01-12 14:00:00",
        "prices": {
            "SH600519": {
                "summary": {"close": 12.0},
                "prices_3d": [{"date": "2026-01-12", "close": 12.0}],
                "hourly_3d": {
                    "candles": [
                        {"timestamp": "2026-01-12 10:30:00", "open": 10, "close": 11},
                        {"timestamp": "2026-01-12 14:00:00", "open": 11, "close": 12},
                    ]
                },
            }
        },
        "news": {"SH600519": {"news": [{"title": "n1", "publish_time": "t1"}], "count": 1}},
        "indicators": {
            "SH600519": {
                "indicators": {"RSI_3": 50},
                "price_indicators": {
                    "momentum": {"RSI_3": 50, "MACD_12_26_9": 0.1},
                    "trend": {"SMA_5_vs_20_pct": 1.2},
                    "risk": {"MAX_DRAWDOWN_5D": -2.4},
                    "microstructure": {"hit_limit_up": False, "hit_limit_down": False, "near_limit_up": False, "near_limit_down": False},
                },
            }
        },
    })

    assert compact["prices"]["SH600519"]["hourly_3d_available"] is True
    assert compact["prices"]["SH600519"]["hourly_3d_candle_count"] == 2
    assert "hourly_3d" not in compact["prices"]["SH600519"]
    assert compact["news"]["SH600519"]["news"][0]["title"] == "n1"
    assert compact["indicators"]["SH600519"]["price_indicators"]["trend"]["SMA_5_vs_20_pct"] == 1.2


def test_prefetch_all_prices_cuts_hourly_cache_at_decision_time():
    class FakeDataManager:
        def get_hourly_stock_data(self, symbol, end_date, lookback_hours):
            index = pd.to_datetime(
                [
                    "2026-01-11 10:00:00",
                    "2026-01-12 10:00:00",
                    "2026-01-12 11:00:00",
                    "2026-01-12 15:00:00",
                ]
            )
            return pd.DataFrame(
                {
                    "open": [9.5, 10.0, 11.0, 12.0],
                    "high": [10.0, 10.5, 11.5, 12.5],
                    "low": [9.0, 9.5, 10.5, 11.5],
                    "close": [9.8, 10.2, 11.2, 12.2],
                    "volume": [1000, 1200, 1300, 1400],
                },
                index=index,
            )

    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    agent.dm = FakeDataManager()

    agent._prefetch_all_prices("2026-01-12", "2026-01-12 10:30:00")

    payload = agent._prefetched_prices["SH600519"]
    assert payload["summary"]["timestamp"] == "2026-01-12 10:00:00"
    timestamps = [item["timestamp"] for item in payload["hourly_3d"]["candles"]]
    assert "2026-01-12 11:00:00" not in timestamps
    assert "2026-01-12 15:00:00" not in timestamps


def test_snapshot_price_indicator_helpers_are_objective():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH688008"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    idx = pd.date_range("2026-01-06 10:30:00", periods=24, freq="h")
    closes = [10, 10.2, 10.4, 10.1, 9.8, 9.5, 9.7, 9.9, 10.0, 10.4, 10.8, 11.0,
              10.7, 10.5, 10.2, 10.0, 9.6, 9.4, 9.8, 10.1, 10.4, 10.8, 11.2, 11.5]
    df = pd.DataFrame({"close": closes}, index=idx)

    sma_pct = agent._calculate_sma_5_vs_20_pct(df, pd.Timestamp("2026-01-07 09:30:00"))
    drawdown = agent._calculate_max_drawdown_pct(df, pd.Timestamp("2026-01-07").date(), pd.Timestamp("2026-01-07 09:30:00"))

    assert isinstance(sma_pct, float)
    assert drawdown <= 0


def test_limit_up_down_hard_blocks_execution():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH688008"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )

    buy_allowed, buy_reason = agent._passes_price_limit_liquidity("buy", 120.0, {"upper": 120.0, "lower": 80.0})
    sell_allowed, sell_reason = agent._passes_price_limit_liquidity("sell", 80.0, {"upper": 120.0, "lower": 80.0})

    assert buy_allowed is False
    assert "Trade_Failed: Limit_Up_Restriction" in buy_reason
    assert sell_allowed is False
    assert "Trade_Failed: Limit_Down_Restriction" in sell_reason


def test_indicator_tool_calculates_requested_snapshot_indicators():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    candles = []
    for idx in range(8):
        candles.append({
            "timestamp": f"2026-01-12 {9 + idx:02d}:30:00",
            "open": 10 + idx,
            "high": 10.5 + idx,
            "low": 9.5 + idx,
            "close": 10 + idx,
            "volume": 100 + idx,
        })
    agent._prefetched_prices = {
        "SH600519": {
            "summary": {"close": 17.0, "timestamp": "2026-01-12 16:30:00"},
            "hourly_3d": {"candles": candles},
        }
    }

    payload = json.loads(
        agent.get_technical_indicators(
            "SH600519",
            "2026-01-12 16:30:00",
            lookback_days=3,
            indicators=["SMA_3", "VOLATILITY_3", "CUM_RETURN"],
        )
    )

    assert payload["source"] == "snapshot_hourly_3d_calculated"
    assert payload["input_candles"] == 8
    assert payload["indicators"]["SMA_3"] is not None
    assert payload["indicators"]["VOLATILITY_3"] is not None
    assert payload["indicators"]["CUM_RETURN"] > 0


def test_indicator_tool_accepts_custom_macd_parameters_and_reports_unsupported():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    candles = []
    for idx in range(40):
        candles.append({
            "timestamp": f"2026-01-{10 + idx // 8:02d} {9 + idx % 8:02d}:30:00",
            "open": 10 + idx * 0.1,
            "high": 10.5 + idx * 0.1,
            "low": 9.5 + idx * 0.1,
            "close": 10 + idx * 0.1,
            "volume": 100 + idx,
        })
    agent._prefetched_prices = {
        "SH600519": {
            "summary": {"close": 14.0, "timestamp": "2026-01-14 16:30:00"},
            "hourly_3d": {"candles": candles},
        }
    }

    payload = json.loads(
        agent.get_technical_indicators(
            "SH600519",
            "2026-01-14 16:30:00",
            lookback_days=5,
            indicators=["MACD_6_13_5", "ATR_14", "NOT_A_REAL_INDICATOR"],
        )
    )

    assert payload["source"] == "snapshot_hourly_3d_calculated"
    assert "MACD_6_13_5" in payload["indicators"]
    assert "MACDh_6_13_5" in payload["indicators"]
    assert "ATRr_14" in payload["indicators"]
    assert "NOT_A_REAL_INDICATOR" in payload["unsupported_indicators"]


def test_search_stock_news_reads_requested_csv_range(tmp_path):
    news_csv = tmp_path / "news.csv"
    news_csv.write_text(
        "\n".join([
            "symbol,title,content,publish_time,source,url",
            "SH600519,old news,old,2026-01-01 09:00:00,src,http://old",
            "SH600519,recent news one,body1,2026-01-08 09:00:00,src,http://one",
            "SH600519,recent news two,body2,2026-01-12 10:00:00,src,http://two",
            "SH600519,future news,body4,2026-01-12 15:00:00,src,http://future",
            "SH600519,unknown time news,body5,not-a-time,src,http://unknown",
            "SH688008,other symbol,body3,2026-01-12 10:00:00,src,http://three",
        ]),
        encoding="utf-8",
    )
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path=str(news_csv),
        openai_api_key="test",
        init_date="2026-01-12",
    )

    payload = json.loads(
        agent.search_stock_news(
            "SH600519 近一周相关新闻",
            current_time="2026-01-12 14:00:00",
            lookback_days=7,
        )
    )

    assert payload["source"] == "news_csv"
    assert payload["symbol"] == "SH600519"
    assert payload["total_count"] == 2
    assert [item["title"] for item in payload["news"]] == ["recent news two", "recent news one"]


def test_prefetch_all_news_populates_snapshot_from_csv(tmp_path):
    news_csv = tmp_path / "news.csv"
    news_csv.write_text(
        "\n".join([
            "symbol,title,content,publish_time,source,url",
            "SH600519,old news,old,2025-12-01 09:00:00,src,http://old",
            "SH600519,recent news one,body1,2026-01-08 09:00:00,src,http://one",
            "SH600519,recent news two,body2,2026-01-12 10:00:00,src,http://two",
            "SH600519,future same day news,body4,2026-01-12 15:00:00,src,http://future",
            "SH600519,unknown time news,body5,not-a-time,src,http://unknown",
            "SH688008,other symbol,body3,2026-01-12 10:00:00,src,http://three",
        ]),
        encoding="utf-8",
    )
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path=str(news_csv),
        openai_api_key="test",
        init_date="2026-01-12",
    )

    agent._prefetch_all_news("2026-01-12", "2026-01-12 14:00:00")

    cached = agent._prefetched_news["SH600519"]
    assert cached["count"] == 2
    assert [item["title"] for item in cached["news"]] == ["recent news two", "recent news one"]

    compact = agent._compact_snapshot_for_llm({"news": agent._prefetched_news})
    shown = compact["news"]["SH600519"]
    assert shown["count"] == 2
    assert len(shown["news"]) == 2


def test_prefetch_fetch_with_retry_recovers_after_failures(monkeypatch):
    from utilities.prefetch_historical_news import _fetch_with_retry

    sleeps: list[float] = []
    monkeypatch.setattr(
        "utilities.prefetch_historical_news.time.sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )
    monkeypatch.setattr(
        "utilities.prefetch_historical_news.random.uniform",
        lambda _a, _b: 0.0,
    )

    attempts = {"count": 0}

    def flaky_fetcher():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("simulated timeout")
        return pd.DataFrame([{"title": "ok"}])

    result = _fetch_with_retry(
        "sina SH688271",
        flaky_fetcher,
        max_retries=3,
        retry_backoff_seconds=2.0,
    )

    assert attempts["count"] == 3
    assert len(result) == 1
    assert sleeps == [2.0, 4.0]


def test_prefetch_fetch_with_retry_returns_empty_after_exhausted_retries(monkeypatch):
    from utilities.prefetch_historical_news import _fetch_with_retry

    monkeypatch.setattr("utilities.prefetch_historical_news.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "utilities.prefetch_historical_news.random.uniform",
        lambda _a, _b: 0.0,
    )

    result = _fetch_with_retry(
        "sina SH688271",
        lambda: (_ for _ in ()).throw(ConnectionError("down")),
        max_retries=2,
        retry_backoff_seconds=1.0,
    )

    assert result.empty


def test_sina_deep_sleeps_between_pages(monkeypatch):
    class FakeResponse:
        def __init__(self, text):
            self.text = text
            self.encoding = "utf-8"

        def raise_for_status(self):
            return None

    calls = []
    sleeps = []

    def fake_get(_url, params, headers, timeout):
        calls.append(params["Page"])
        if params["Page"] == 1:
            return FakeResponse(
                '2026-06-30&nbsp;10:00 <a href="https://finance.sina.com.cn/a.html">新闻一</a>'
            )
        return FakeResponse(
            '2026-02-28&nbsp;10:00 <a href="https://finance.sina.com.cn/b.html">旧新闻</a>'
        )

    monkeypatch.setattr(prefetch_historical_news.requests, "get", fake_get)
    monkeypatch.setattr(prefetch_historical_news, "_polite_sleep", lambda seconds: sleeps.append(seconds))

    df = prefetch_historical_news.fetch_sina_stock_news_deep(
        "SH688981",
        timeout=1,
        max_pages=2,
        start_dt=pd.Timestamp("2026-03-01 00:00:00"),
        page_sleep_seconds=0.5,
    )

    assert calls == [1, 2]
    assert sleeps == [0.5]
    assert len(df) == 1


def test_trade_tool_validation_messages():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )

    buy_payload = json.loads(agent.buy_stock("SH600519", "abc"))
    sell_payload = json.loads(agent.sell_stock("SH600519", -100))

    assert buy_payload["error"] == "买入数量必须是正整数"
    assert sell_payload["error"] == "卖出数量必须大于0"


def test_no_trade_tool_returns_signature(monkeypatch):
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    agent.runtime_context["CURRENT_TIME"] = "2026-01-12 14:00:00"
    agent.runtime_context["DECISION_COUNT"] = 3
    monkeypatch.setattr("agent_engine.agent.agent.add_no_trade_record", lambda *args, **kwargs: None)

    payload = json.loads(agent.add_no_trade_record_tool("2026-01-12"))

    assert payload["signature"] == "test-agent"
    assert "agentic workflow" not in payload


def test_benchmark_decision_report_extraction_and_logging(tmp_path):
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    response_text = """
Observation Summary: ...
```json
{
  "decision_evidence_report": {
    "schema_version": 2,
    "signature": "test-agent",
    "date": "2026-01-12",
    "decision_time": "2026-01-12 14:00:00",
    "decision_count": 3,
    "observed_universe": ["SH600519"],
    "candidate_review": [
      {
        "symbol": "SH600519",
        "rank": 1,
        "selected_for_action": true,
        "news_evidence_used": [
          {
            "title": "positive news",
            "publish_time": "2026-01-12 09:30:00",
            "source": "snapshot",
            "model_interpretation": "company-specific positive catalyst",
            "claimed_direction": "positive",
            "specificity": "company",
            "freshness": "same_day"
          }
        ],
        "price_evidence_used": {
          "current_price": 12.5,
          "recent_change_pct": 1.2,
          "rsi_3": 55.0,
          "macd_12_26_9": 0.12,
          "price_indicators_used": {
            "momentum": {"RSI_3": 55.0, "MACD_12_26_9": 0.12},
            "trend": {"SMA_5_vs_20_pct": 1.5},
            "risk": {"MAX_DRAWDOWN_5D": -2.0},
            "microstructure": {"hit_limit_up": false, "hit_limit_down": false, "near_limit_up": false, "near_limit_down": false}
          },
          "signal_evaluation": {
            "momentum_reading": "neutral",
            "trend_reading": "bullish",
            "risk_reading": "acceptable",
            "momentum_trend_conflict": false,
            "decision_implication": "supports_entry"
          },
          "model_price_reading": "price confirmation is moderate"
        },
        "risk_checks_mentioned": ["cash", "position_limit"],
        "buy_reason_text": "buy based on positive news and moderate price confirmation",
        "reject_or_hold_reason_text": ""
      }
    ],
    "actions_planned_or_taken": [
      {
        "action": "buy",
        "symbol": "SH600519",
        "amount": 100,
        "reason_text": "test",
        "linked_candidate_rank": 1,
        "linked_evidence_titles": ["positive news"],
        "risk_controls_cited": ["cash", "position_limit"]
      }
    ],
    "workflow_trace": {
      "has_candidate_review": true,
      "has_news_evidence": true,
      "has_price_evidence": true,
      "has_risk_checks": true,
      "has_action_reason": true,
      "missing_required_sections": []
    }
  }
}
```
<FINISH_SIGNAL>
"""

    report, error = agent._extract_benchmark_decision_report(response_text)
    assert error is None
    assert report["actions_planned_or_taken"][0]["linked_evidence_titles"] == ["positive news"]

    log_file = tmp_path / "decision.jsonl"
    agent._log_benchmark_decision_report(str(log_file), response_text, "2026-01-12 14:00:00", 3)
    entry = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert entry["event_type"] == "decision_evidence_report"
    assert entry["parse_success"] is True
    assert entry["schema_version"] == 2
    assert entry["schema_valid"] is True
    assert entry["report"]["candidate_review"][0]["news_evidence_used"][0]["claimed_direction"] == "positive"


def test_benchmark_decision_report_schema_validation_flags_missing_sections():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )

    validation = agent._validate_benchmark_decision_report({
        "schema_version": 2,
        "observed_universe": ["SH600519"],
        "actions_planned_or_taken": [],
    })

    assert validation["schema_valid"] is False
    assert "candidate_review" in validation["missing_required_sections"]
    assert "workflow_trace" in validation["missing_required_sections"]
    assert "actions_planned_or_taken.non_empty" in validation["missing_required_sections"]


def test_decision_report_validation_defaults_optional_risk_check_lists():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )

    report = {
        "schema_version": 2,
        "observed_universe": ["SH600519"],
        "candidate_review": [
            {
                "symbol": "SH600519",
                "rank": 1,
                "selected_for_action": True,
                "news_evidence_used": [
                    {
                        "title": "positive news",
                        "model_interpretation": "positive",
                        "claimed_direction": "positive",
                    }
                ],
                "price_evidence_used": {
                    "signal_evaluation": {
                        "momentum_reading": "bullish",
                        "trend_reading": "bullish",
                        "risk_reading": "acceptable",
                        "momentum_trend_conflict": False,
                        "decision_implication": "supports_entry",
                    }
                },
                "buy_reason_text": "buy",
                "reject_or_hold_reason_text": "",
            }
        ],
        "actions_planned_or_taken": [
            {
                "action": "buy",
                "symbol": "SH600519",
                "reason_text": "buy",
                "risk_controls_cited": ["cash", "position_limit"],
            }
        ],
        "workflow_trace": {
            "has_candidate_review": True,
            "has_news_evidence": True,
            "has_price_evidence": True,
            "has_risk_checks": True,
            "has_action_reason": True,
        },
    }

    validation = agent._validate_benchmark_decision_report(report)

    assert validation["schema_valid"] is True
    assert report["candidate_review"][0]["risk_checks_mentioned"] == ["cash", "position_limit"]
    assert report["workflow_trace"]["missing_required_sections"] == []
    assert "candidate_review[0].risk_checks_mentioned_inferred_from_actions" in validation["schema_warnings"]
    assert "workflow_trace.missing_required_sections_defaulted_empty" in validation["schema_warnings"]


def test_decision_report_parser_recovers_json_inside_fenced_block_with_intro():
    agent = object.__new__(AgenticWorkflow)
    response_text = """
```json
Here is the structured report:
{
  "decision_evidence_report": {
    "schema_version": 2,
    "observed_universe": ["SH600519"],
    "candidate_review": [],
    "actions_planned_or_taken": [{"action": "no_trade", "reason_text": "wait", "risk_controls_cited": []}],
    "workflow_trace": {
      "has_candidate_review": false,
      "has_news_evidence": false,
      "has_price_evidence": false,
      "has_risk_checks": false,
      "has_action_reason": true,
      "missing_required_sections": []
    }
  }
}
```
"""

    report, error = agent._extract_benchmark_decision_report(response_text)

    assert error is None
    assert report["schema_version"] == 2
    assert report["actions_planned_or_taken"][0]["action"] == "no_trade"


def test_legacy_benchmark_report_name_still_parses():
    agent = AgenticWorkflow(
        signature="test-agent",
        basemodel="test-model",
        stock_symbols=["SH600519"],
        stock_json_path="./data_flow/ai_stock_data.json",
        news_csv_path="./data_flow/news.csv",
        openai_api_key="test",
        init_date="2026-01-12",
    )
    response_text = '{"benchmark_decision_report": {"schema_version": 1, "actions": [], "candidates": [], "workflow_coverage": {}, "risk_summary": []}}'

    report, error = agent._extract_benchmark_decision_report(response_text)

    assert error is None
    assert report["schema_version"] == 1


def test_agent_system_prompt_preserves_json_schema_braces():
    prompt = get_agent_system_prompt("2026-01-12", "test-agent", current_time="2026-01-12 14:00:00", decision_count=3)

    assert STOP_SIGNAL in prompt
    assert '"decision_evidence_report"' in prompt
    assert '"schema_version": 2' in prompt
    assert "benchmark" not in prompt.lower()
    assert "Do not trade merely to be active" in prompt
    assert "active waiting" in prompt
    assert "RSI_3 is a sensitive short-horizon trigger, not a standalone decision maker" in prompt
    assert "signal_evaluation" in prompt
    assert "momentum_trend_conflict" in prompt
    assert "Limit up/down statuses represent execution/liquidity constraints" in prompt
    assert "favor executing at least one trade" not in prompt
    assert "Never expose excuses" not in prompt
    assert "available_to_sell" in prompt
    assert "locked_today" in prompt
    assert "never call more than one position-mutating tool" in prompt
