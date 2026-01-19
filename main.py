import os
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import uuid
import sys

# 必须在任何模块导入之前设置，避免 HuggingFace tokenizers 的 fork 警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

from utils.runtime_config import write_runtime_config_value
from utils.backup_utils import run_backup_snapshot
from agent_engine.agent.agent import AgenticWorkflow
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
    
    # Get date range from configuration file
    INIT_DATE = config["date_range"]["init_date"]
    END_DATE = config["date_range"]["end_date"]
    
    # Environment variables can override dates in configuration file
    if os.getenv("INIT_DATE"):
        INIT_DATE = os.getenv("INIT_DATE")
        print(f"⚠️  Using environment variable to override INIT_DATE: {INIT_DATE}")
    if os.getenv("END_DATE"):
        END_DATE = os.getenv("END_DATE")
        print(f"⚠️  Using environment variable to override END_DATE: {END_DATE}")
    
    # Validate date range
    INIT_DATE_obj = datetime.strptime(INIT_DATE, "%Y-%m-%d").date()
    END_DATE_obj = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    if INIT_DATE_obj > END_DATE_obj:
        print("❌ INIT_DATE is greater than END_DATE")
        exit(1)
    
    # Validate that dates don't exceed current date (unless REALTIME_MODE=wait)
    current_date = datetime.now().date()
    realtime_mode = os.getenv("REALTIME_MODE", "").strip().lower()
    is_realtime_wait_mode = realtime_mode == "wait"
    
    if not is_realtime_wait_mode:
        # 非实时模式：不允许未来日期（回测模式）
        if INIT_DATE_obj > current_date:
            print(f"❌ INIT_DATE ({INIT_DATE}) cannot be in the future. Current date is {current_date.strftime('%Y-%m-%d')}")
            exit(1)
        
        # If END_DATE equals or exceeds current date, exit to avoid testing on current date
        if END_DATE_obj >= current_date:
            print(f"⚠️  END_DATE ({END_DATE}) is equal to or exceeds current date ({current_date.strftime('%Y-%m-%d')}).")
            print("❌ Cannot run trading test on current date or future dates. Please set END_DATE to a past date.")
            print("💡 Tip: Set REALTIME_MODE=wait to enable real-time trading mode that can wait for future decision points.")
            exit(1)
    else:
        # 实时等待模式：允许当前和未来日期，系统会等待到决策时点
        if INIT_DATE_obj > current_date:
            print(f"⚠️  INIT_DATE ({INIT_DATE}) is in the future. REALTIME_MODE=wait will wait for the date to arrive.")
        if END_DATE_obj >= current_date:
            print(f"✅ REALTIME_MODE=wait enabled: Will wait for decision points until END_DATE ({END_DATE})")

    # Get model list from configuration file (only select enabled models)
    enabled_models = [
        model for model in config["models"]
        if model.get("enabled", True)
    ]

    # Environment-based filtering for single model run
    only_signature = os.getenv("ONLY_SIGNATURE")
    if only_signature:
        filtered = [m for m in enabled_models if m.get("signature") == only_signature]
        if not filtered:
            print(f"❌ 未找到 signature={only_signature} 的启用模型")
            return
        enabled_models = filtered
    
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
    global_force_replay = (
        agent_config.get("force_replay", False)
        or _truthy_env("FORCE_REPLAY")
        or _truthy_env("RESET_POSITIONS")
    )
    
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

    # Multiprocess dispatch mode: spawn one child per model and return
    _maybe_run_backup(f"main:{INIT_DATE}->{END_DATE}")

    if _truthy_env("PARALLEL_RUN") and not only_signature:
        LOG_DIR = Path(__file__).parent / "logs" / "jobs"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for model_config in enabled_models:
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
            env["ONLY_SIGNATURE"] = sig or ""
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
            # Can choose to continue processing next model, or exit
            # continue  # Continue processing next model
            exit()  # Or exit program
        
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
    
    asyncio.run(main(config_path))

