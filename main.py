import os
import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import uuid
import sys
from typing import Optional, TextIO

# 必须在任何模块导入之前设置，避免 HuggingFace tokenizers 的 fork 警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

from utils.runtime_config import write_runtime_config_value
from utils.backup_utils import run_backup_snapshot
from utils.news_cache_guard import NewsCacheIntegrityError, validate_news_cache_integrity
from agent_engine.agent.agent import AgenticWorkflow
from utilities.prepare_benchmark_snapshots import prepare_benchmark_snapshots, should_prepare_snapshots
from utilities.prefetch_historical_news import prefetch_historical_news
DEFAULT_STOCK_SYMBOLS = AgenticWorkflow.DEFAULT_STOCK_SYMBOLS

# Agent class mapping table - for dynamic import and instantiation
AGENT_REGISTRY = {
    "AgenticWorkflow": {
        "module": "agent_engine.agent.agent",
        "class": "AgenticWorkflow"
    },
}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "false").lower() in ("1", "true", "yes")


def _truthy_config(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _configured_realtime_mode(config: dict) -> str:
    run_config = config.get("run_config", {})
    mode = str(run_config.get("realtime_mode", "") or "").strip().lower()
    if mode == "backtest":
        return ""
    if mode not in ("", "stop", "wait"):
        raise ValueError("run_config.realtime_mode must be one of: backtest, stop, wait")
    return mode


def configure_backtest_mode(config: dict) -> str:
    """Apply run_config.backtest_mode to BACKTEST_MODE and return the effective value."""
    run_config = config.get("run_config", {})
    if "backtest_mode" in run_config:
        os.environ["BACKTEST_MODE"] = "true" if _truthy_config(run_config.get("backtest_mode")) else "false"
    else:
        # 默认回测模式：工具只读 news.csv / shared snapshot，不访问实时外部数据源。
        os.environ.setdefault("BACKTEST_MODE", "true")
    return os.environ.get("BACKTEST_MODE", "true")


def _defer_startup_data_prep(realtime_mode: str, end_date_obj, current_date) -> bool:
    """In live wait mode, build data just-in-time at each decision point."""
    return realtime_mode == "wait" and end_date_obj >= current_date


def _allows_current_or_future_dates(realtime_mode: str) -> bool:
    """Real-time modes guard future decisions themselves, so current date is valid."""
    return realtime_mode in ("wait", "stop")


def should_prefetch_news_before_run(config: dict) -> bool:
    """Fetch the configured decision-date news before building shared snapshots."""
    if _truthy_env("SKIP_NEWS_PREFETCH"):
        return False
    run_config = config.get("run_config", {})
    return _truthy_config(run_config.get("prefetch_news_before_run", True))


def prefetch_configured_news_before_run(config: dict, stock_symbols) -> None:
    """Prefetch news into news.csv before snapshot construction.

    Uses date_range.end_date as the anchor date to keep runs reproducible. For
    backfills, set run_config.prefetch_news_lookback_days > 1.
    """
    if not should_prefetch_news_before_run(config):
        print("📰 News prefetch skipped.")
        return

    date_range = config.get("date_range", {})
    init_date = date_range.get("init_date")
    end_date = date_range.get("end_date")
    if not init_date or not end_date:
        raise ValueError("date_range.init_date/end_date must be configured before news prefetch")

    run_config = config.get("run_config", {})
    data_config = config.get("data_config", {})
    try:
        lookback_days = max(1, int(run_config.get("prefetch_news_lookback_days", 1)))
    except Exception:
        lookback_days = 1
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    init_dt = datetime.strptime(init_date, "%Y-%m-%d")
    lookback_start_dt = end_dt - timedelta(days=lookback_days - 1)
    respect_init_date = _truthy_config(run_config.get("prefetch_news_respect_init_date", False))
    start_dt = max(init_dt, lookback_start_dt) if respect_init_date else lookback_start_dt
    output_path = Path(data_config.get("news_csv_path", "./data_flow/news.csv"))
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    sleep_seconds = float(run_config.get("prefetch_news_sleep_seconds", 1.0))
    max_pages = int(run_config.get("prefetch_news_max_pages", 2))
    page_size = int(run_config.get("prefetch_news_page_size", 50))
    timeout = int(run_config.get("prefetch_news_timeout", 30))
    max_retries = int(run_config.get("prefetch_news_max_retries", 3))
    retry_backoff_seconds = float(run_config.get("prefetch_news_retry_backoff_seconds", 3.0))
    tushare_sources = run_config.get("prefetch_news_tushare_sources")
    if not isinstance(tushare_sources, list):
        tushare_sources = None
    tushare_chunk_days = int(run_config.get("prefetch_news_tushare_chunk_days", 1))
    tushare_chunk_hours_raw = run_config.get("prefetch_news_tushare_chunk_hours", 6)
    tushare_chunk_hours = int(tushare_chunk_hours_raw) if tushare_chunk_hours_raw not in (None, "") else None
    sina_max_pages_raw = run_config.get("prefetch_news_sina_max_pages")
    sina_max_pages = int(sina_max_pages_raw) if sina_max_pages_raw not in (None, "") else None
    sina_page_sleep_seconds = float(run_config.get("prefetch_news_sina_page_sleep_seconds", 0.8))
    print(
        "📰 Prefetching news before run: "
        f"{start_dt.strftime('%Y-%m-%d')} -> {end_date}, "
        f"timeout={timeout}s, retries={max_retries}, sleep={sleep_seconds}s, output={output_path}"
    )
    prefetch_historical_news(
        symbols=stock_symbols,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_date,
        output_path=output_path,
        page_size=page_size,
        max_pages=max_pages,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        use_akshare_stock_news=_truthy_config(run_config.get("prefetch_news_use_akshare_stock_news", True)),
        use_akshare_calendar_events=_truthy_config(run_config.get("prefetch_news_use_akshare_calendar_events", True)),
        use_tushare_news=_truthy_config(run_config.get("prefetch_news_use_tushare", True)),
        tushare_sources=tushare_sources,
        tushare_chunk_days=tushare_chunk_days,
        tushare_chunk_hours=tushare_chunk_hours,
        use_sina_deep=_truthy_config(run_config.get("prefetch_news_use_sina_deep", True)),
        sina_max_pages=sina_max_pages,
        sina_page_sleep_seconds=sina_page_sleep_seconds,
        use_sse_announcements=_truthy_config(run_config.get("prefetch_news_use_sse_announcements", True)),
        use_cninfo_fulltext=_truthy_config(run_config.get("prefetch_news_use_cninfo_fulltext", True)),
    )


class _TeeStream:
    """Write process output to both terminal and a job log."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())


def _install_job_log_tee(label: str = "main") -> Optional[TextIO]:
    """Ensure every run has a jobs/*.log transcript, even single-process smoke runs."""
    existing = os.getenv("ASTOCK_JOB_LOG_PATH")
    if existing:
        print(f"🧾 Job log: {existing}")
        return None

    jobs_dir = Path(__file__).parent / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in label) or "main"
    log_file = jobs_dir / f"{safe_label}-{uuid.uuid4().hex[:8]}.log"
    lf = open(log_file, "w", encoding="utf-8", buffering=1)
    os.environ["ASTOCK_JOB_LOG_PATH"] = str(log_file)
    sys.stdout = _TeeStream(sys.stdout, lf)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, lf)  # type: ignore[assignment]
    print(f"🧾 Job log: {log_file}")
    return lf


def _maybe_run_backup(reason: str) -> None:
    if _truthy_env("SKIP_AUTO_BACKUP"):
        print("⚠️ Auto-backup skipped via SKIP_AUTO_BACKUP.")
        return

    retain_value = os.getenv("AUTO_BACKUP_RETAIN")
    retain = None
    if retain_value:
        try:
            retain = int(retain_value)
        except ValueError:
            retain = None

    ok = run_backup_snapshot(reason=reason, retain=retain)
    if not ok:
        print("⚠️ Backup snapshot failed; continuing without blocking trading.")


def validate_benchmark_startup(config, enabled_models):
    errors = []
    try:
        datetime.strptime(config["date_range"]["init_date"], "%Y-%m-%d")
        datetime.strptime(config["date_range"]["end_date"], "%Y-%m-%d")
    except Exception as exc:
        errors.append(f"date_range 格式错误: {exc}")

    needs_openrouter = any(
        "openrouter.ai" in str(model.get("openai_base_url", ""))
        for model in enabled_models
    )
    if needs_openrouter and not os.getenv("OPENROUTER_API_KEY"):
        errors.append("缺少 OPENROUTER_API_KEY（用于 OpenRouter 模型）")

    data_config = config.get("data_config", {})
    stock_path = Path(data_config.get("stock_json_path", "./data_flow/ai_stock_data.json"))
    if not stock_path.is_absolute():
        stock_path = Path(__file__).parent / stock_path
    if not stock_path.exists():
        errors.append(f"股票数据文件不存在: {stock_path}")

    news_path = Path(data_config.get("news_csv_path", "./data_flow/news.csv"))
    if not news_path.is_absolute():
        news_path = Path(__file__).parent / news_path
    try:
        validate_news_cache_integrity(news_path, strict=True)
    except NewsCacheIntegrityError as exc:
        errors.append(f"新闻缓存完整性错误: {exc}")

    if should_prepare_snapshots() and not os.getenv("TSL_USER") and not os.getenv("TSL_USERNAME"):
        print("⚠️ 未检测到 TSL_USER/TSL_USERNAME；snapshot 准备将优先使用本地数据，缺数据时可能 fail-fast。")

    if errors:
        for error in errors:
            print(f"❌ {error}")
        raise SystemExit(1)


def get_agent_class(agent_type):
    """
    Dynamically import and return the corresponding class based on agent type name
    
    Args:
        agent_type: Agent type name (e.g., "AgenticWorkflow")
        
    Returns:
        Agent class
        
    Raises:
        ValueError: If agent type is not supported
        ImportError: If unable to import agent module
    """
    if agent_type not in AGENT_REGISTRY:
        supported_types = ", ".join(AGENT_REGISTRY.keys())
        raise ValueError(
            f"❌ Unsupported agent type: {agent_type}\n"
            f"   Supported types: {supported_types}"
        )
    
    agent_info = AGENT_REGISTRY[agent_type]
    module_path = agent_info["module"]
    class_name = agent_info["class"]
    
    try:
        # Dynamic import module
        import importlib
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        print(f"✅ Successfully loaded Agent class: {agent_type} (from {module_path})")
        return agent_class
    except ImportError as e:
        raise ImportError(f"❌ Unable to import agent module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"❌ Class {class_name} not found in module {module_path}: {e}")


def load_config(config_path=None):
    """
    Load configuration file from settings directory
    
    Args:
        config_path: Configuration file path, if None use default config
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Default configuration file path
        config_path = Path(__file__).parent / "settings" / "default_config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ Configuration file does not exist: {config_path}")
        exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Successfully loaded configuration file: {config_path}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ Configuration file JSON format error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Failed to load configuration file: {e}")
        exit(1)


async def main(config_path=None):
    """Run trading experiment using AgenticWorkflow class

    Args:
        config_path: Configuration file path, if None use default config
    """
    # Load configuration file
    config = load_config(config_path)
    
    # Get Agent type
    agent_type = config.get("agent_type", "AgenticWorkflow")
    try:
        AgentClass = get_agent_class(agent_type)
    except (ValueError, ImportError, AttributeError) as e:
        print(str(e))
        exit(1)
    
    # Get date range from configuration file. Dates are intentionally not
    # overrideable from shell env to keep runs reproducible.
    INIT_DATE = config["date_range"]["init_date"]
    END_DATE = config["date_range"]["end_date"]
    run_config = config.get("run_config", {})
    try:
        realtime_mode = _configured_realtime_mode(config)
    except ValueError as exc:
        print(f"❌ {exc}")
        exit(1)

    # Validate date range
    INIT_DATE_obj = datetime.strptime(INIT_DATE, "%Y-%m-%d").date()
    END_DATE_obj = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    if INIT_DATE_obj > END_DATE_obj:
        print("❌ INIT_DATE is greater than END_DATE")
        exit(1)
    
    # Validate that dates don't exceed current date unless a real-time mode is active.
    # "wait" sleeps until future decision points; "stop" exits when it reaches one.
    current_date = datetime.now().date()
    is_realtime_mode = _allows_current_or_future_dates(realtime_mode)
    
    if not is_realtime_mode:
        # 非实时模式：不允许未来日期（回测模式）
        if INIT_DATE_obj > current_date:
            print(f"❌ INIT_DATE ({INIT_DATE}) cannot be in the future. Current date is {current_date.strftime('%Y-%m-%d')}")
            exit(1)
        
        # If END_DATE equals or exceeds current date, exit to avoid testing on current date
        if END_DATE_obj >= current_date:
            print(f"⚠️  END_DATE ({END_DATE}) is equal to or exceeds current date ({current_date.strftime('%Y-%m-%d')}).")
            print("❌ Cannot run trading test on current date or future dates. Please set END_DATE to a past date.")
            print('💡 Tip: Set "run_config": {"realtime_mode": "wait"} to enable real-time mode.')
            exit(1)
    else:
        # 实时模式：允许当前和未来日期；wait 会等待，stop 会在未来时点停止。
        if INIT_DATE_obj > current_date:
            print(f"⚠️  INIT_DATE ({INIT_DATE}) is in the future. REALTIME_MODE={realtime_mode} will not run future decisions early.")
        if END_DATE_obj >= current_date:
            print(f"✅ REALTIME_MODE={realtime_mode} enabled: current/future decision points will not run early.")

    # Get model list from configuration file (only select enabled models)
    enabled_models = [
        model for model in config["models"]
        if model.get("enabled", True)
    ]

    # Internal child-process filtering used by config-driven parallel mode.
    only_signature = os.getenv("ASTOCK_CHILD_SIGNATURE")
    if only_signature:
        filtered = [m for m in enabled_models if m.get("signature") == only_signature]
        if not filtered:
            print(f"❌ 未找到 signature={only_signature} 的启用模型")
            return
        enabled_models = filtered

    validate_benchmark_startup(config, enabled_models)
    effective_backtest_mode = configure_backtest_mode(config)
    
    # Get agent configuration
    agent_config = config.get("agent_config", {})
    data_config = config.get("data_config", {})
    log_config = config.get("log_config", {})
    trading_rules = config.get("trading_rules", {})
    risk_management = config.get("risk_management", {})
    max_steps = agent_config.get("max_steps", 10)
    max_retries = agent_config.get("max_retries", 3)
    base_delay = agent_config.get("base_delay", 0.5)
    initial_cash = agent_config.get("initial_cash", 1000000.0)
    global_force_replay = bool(agent_config.get("force_replay", False))
    parallel_run = _truthy_config(run_config.get("parallel_run", False))
    try:
        parallel_spawn_delay_seconds = max(0.0, float(run_config.get("parallel_spawn_delay_seconds", 0.0)))
    except Exception:
        parallel_spawn_delay_seconds = 0.0
    write_runtime_config_value("REALTIME_MODE", realtime_mode)
    if "snapshot_hourly_cache_days" in run_config:
        os.environ["SNAPSHOT_HOURLY_CACHE_DAYS"] = str(run_config.get("snapshot_hourly_cache_days"))
    
    # Get DataManager paths
    stock_json_path = data_config.get("stock_json_path", "./data_flow/ai_stock_data.json")
    news_csv_path = data_config.get("news_csv_path", "./data_flow/news.csv")
    
    # Display enabled model information
    model_names = [m.get("name", m.get("signature")) for m in enabled_models]
    
    print("🚀 Starting trading experiment")
    print(f"🤖 Agent type: {agent_type}")
    print(f"📅 Date range: {INIT_DATE} to {END_DATE}")
    print(f"🤖 Model list: {model_names}")
    print(f"⚙️  Agent config: max_steps={max_steps}, max_retries={max_retries}, base_delay={base_delay}, initial_cash={initial_cash}")
    print(f"🧭 BACKTEST_MODE={effective_backtest_mode}")

    # Multiprocess dispatch mode: spawn one child per model and return
    _maybe_run_backup(f"main:{INIT_DATE}->{END_DATE}")

    defer_startup_data_prep = _defer_startup_data_prep(realtime_mode, END_DATE_obj, current_date)

    if not _truthy_env("NEWS_ALREADY_PREFETCHED"):
        if defer_startup_data_prep:
            print("📰 实时等待模式：跳过启动时新闻预抓取，将在每个决策点前处理。")
        else:
            prefetch_configured_news_before_run(config, DEFAULT_STOCK_SYMBOLS)
            os.environ["NEWS_ALREADY_PREFETCHED"] = "true"

    if should_prepare_snapshots() and not _truthy_env("SNAPSHOTS_ALREADY_PREPARED"):
        if defer_startup_data_prep:
            print("📦 实时等待模式：跳过启动时批量 snapshot，将在每个决策点到点生成。")
        else:
            print("📦 正在准备共享回测 snapshot（跑 agent 前）...")
            prepared_count = await prepare_benchmark_snapshots(config, AgentClass, DEFAULT_STOCK_SYMBOLS)
            os.environ["SNAPSHOTS_ALREADY_PREPARED"] = "true"
            print(f"✅ 共享回测 snapshot 就绪：{prepared_count} 个决策点")

    if parallel_run and not only_signature:
        LOG_DIR = Path(__file__).parent / "jobs"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for model_idx, model_config in enumerate(enabled_models):
            sig = model_config.get("signature")
            job_id = f"{sig}-{uuid.uuid4().hex[:8]}"
            log_file = LOG_DIR / f"{job_id}.log"
            try:
                # 使用文本模式打开，确保编码正确，并设置行缓冲
                lf = open(log_file, "w", encoding="utf-8", buffering=1)
            except Exception:
                lf = open(log_file, "w", encoding="utf-8", buffering=1)
            cmd = [sys.executable, "-u", str(Path(__file__).parent / "main.py")]
            if config_path:
                cmd.append(config_path)
            env = os.environ.copy()
            for stale_key in ("INIT_DATE", "END_DATE", "ONLY_SIGNATURE", "FORCE_REPLAY", "RESET_POSITIONS", "PARALLEL_RUN", "REALTIME_MODE"):
                env.pop(stale_key, None)
            env["ASTOCK_CHILD_SIGNATURE"] = sig or ""
            env["ASTOCK_JOB_LOG_PATH"] = str(log_file)
            env["RUNTIME_ENV_PATH"] = str(Path(__file__).parent / "settings" / "runtime" / f"runtime_env_{sig}.json")
            # 确保 PYTHONUNBUFFERED 环境变量被设置，强制无缓冲输出
            env["PYTHONUNBUFFERED"] = "1"
            subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
            )
            print(f"▶️ 启动子进程: {sig} -> {log_file}")
            if parallel_spawn_delay_seconds > 0 and model_idx < len(enabled_models) - 1:
                print(f"⏳ 错峰启动：等待 {parallel_spawn_delay_seconds:.1f}s 后启动下一个模型")
                await asyncio.sleep(parallel_spawn_delay_seconds)
        print("✅ 并行子进程已全部启动（父进程退出）")
        return

    # Same-process concurrency: build all agents and run concurrently
    agents = []
    for model_config in enabled_models:
        # Read basemodel and signature directly from configuration file
        model_name = model_config.get("name", "unknown")
        basemodel = model_config.get("basemodel")
        signature = model_config.get("signature")
        
        # Read OpenAI specific configuration
        openai_base_url = model_config.get("openai_base_url", None)
        openai_api_key = model_config.get("openai_api_key", None)
        
        # Read Google Gemini specific configuration
        google_api_key = model_config.get("google_api_key", None)
        safety_settings = model_config.get("safety_settings", None)
        
        # Read model parameters (thinking/reasoning configs)
        parameters = model_config.get("parameters", None)

        # Validate required fields
        if not basemodel:
            print(f"❌ Model {model_name} missing basemodel field")
            continue
        if not signature:
            print(f"❌ Model {model_name} missing signature field")
            continue
        
        print("=" * 60)
        print(f"🤖 Processing model: {model_name}")
        print(f"📝 Signature: {signature}")
        print(f"🔧 BaseModel: {basemodel}")
        
        # Initialize runtime configuration (compat; each agent uses its own context during run)
        write_runtime_config_value("SIGNATURE", signature)
        write_runtime_config_value("TODAY_DATE", END_DATE)
        write_runtime_config_value("IF_TRADE", False)


        # Get log path configuration
        log_path = log_config.get("log_path", "./data_flow/trading_summary_each_agent")
        # Determine replay/reset behavior for this agent
        force_replay_flag = bool(model_config.get("force_replay", False) or global_force_replay)
        if force_replay_flag:
            agent_storage = Path(log_path) / signature
            if agent_storage.exists():
                shutil.rmtree(agent_storage, ignore_errors=True)
                print(f"🗑️ Cleared stored state for {signature} (force replay enabled)")

        try:
            # 使用配置的股票池（不再依赖CSV文件）
            stock_symbols_to_use = DEFAULT_STOCK_SYMBOLS
            print(f"📊 使用股票池: {len(stock_symbols_to_use)} 只股票")
            print(f"   股票列表: {stock_symbols_to_use}")
            
            # Dynamically create Agent instance
            agent = AgentClass(
                signature=signature,
                basemodel=basemodel,
                stock_symbols=stock_symbols_to_use,
                stock_json_path=stock_json_path,
                news_csv_path=news_csv_path,
                macro_csv_path=None,
                log_path=log_path,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                google_api_key=google_api_key,
                safety_settings=safety_settings,
                parameters=parameters,
                max_steps=max_steps,
                max_retries=max_retries,
                base_delay=base_delay,
                initial_cash=initial_cash,
                init_date=INIT_DATE,
                trading_rules=trading_rules,
                risk_management=risk_management,
                force_replay=force_replay_flag,
            )
            
            print(f"✅ {agent_type} 实例创建成功: {agent}")
            # 延后执行，统一并发启动
            await agent.initialize()
            print("✅ 初始化成功")
            agents.append(agent)
            
        except Exception as e:
            print(f"❌ Error processing model {model_name} ({signature}): {str(e)}")
            print(f"📋 Error details: {e}")
            raise
        
        print("=" * 60)
        print(f"✅ Model {model_name} ({signature}) initialized")
        print("=" * 60)

    # 并发运行所有 Agent 的日期区间
    if agents:
        await asyncio.gather(*(a.run_date_range(INIT_DATE, END_DATE) for a in agents))
        # 输出每个模型的最终摘要
        for agent in agents:
            summary = agent.get_position_summary()
            print(f"📊 Final position summary ({agent.signature}):")
            print(f"   - Latest date: {summary.get('latest_date')}")
            print(f"   - Total records: {summary.get('total_records')}")
            print(f"   - Cash balance: ${summary.get('positions', {}).get('CASH', 0):.2f}")
    print("🎉 All models processing completed!")
    
if __name__ == "__main__":
    import sys
    
    # Support specifying configuration file through command line arguments
    # Usage: python main.py [config_path]
    # Example: python main.py settings/my_config.json
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if config_path:
        print(f"📄 Using specified configuration file: {config_path}")
    else:
        print(f"📄 Using default configuration file: settings/default_config.json")
    
    _job_log_handle = _install_job_log_tee(os.getenv("ASTOCK_CHILD_SIGNATURE") or "main")
    try:
        asyncio.run(main(config_path))
    finally:
        if _job_log_handle is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _job_log_handle.close()
