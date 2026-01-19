
import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

_BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_RUNTIME_PATH = _BASE_DIR / "settings" / "runtime" / "runtime_env.json"


def _resolve_runtime_env_path() -> Path:
    """Return a writable runtime_env.json path, ensuring directory exists."""
    env_path = os.environ.get("RUNTIME_ENV_PATH")

    def _ensure_parent(path: Path) -> Path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return _DEFAULT_RUNTIME_PATH
        return path

    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.parent.exists():
            candidate = _ensure_parent(candidate)
        if candidate.suffix.lower() != ".json":
            candidate = candidate.with_suffix(".json")
        os.environ["RUNTIME_ENV_PATH"] = str(candidate)
        return candidate

    os.environ["RUNTIME_ENV_PATH"] = str(_DEFAULT_RUNTIME_PATH)
    _DEFAULT_RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_RUNTIME_PATH


def _load_runtime_env() -> dict:
    path = _resolve_runtime_env_path()

    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        # 静默失败，不影响程序运行
        pass
    return {}


def get_runtime_config_value(key: str, default=None):
    """Get runtime configuration value from runtime_env.json or environment variables.
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    _RUNTIME_ENV = _load_runtime_env()
    
    if key in _RUNTIME_ENV:
        return _RUNTIME_ENV[key]
    return os.getenv(key, default)

def write_runtime_config_value(key: str, value: any):
    """Write runtime configuration value to runtime_env.json.
    
    Args:
        key: Configuration key to set
        value: Value to write
    """
    _RUNTIME_ENV = _load_runtime_env()
    _RUNTIME_ENV[key] = value
    path = _resolve_runtime_env_path()

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(_RUNTIME_ENV, f, ensure_ascii=False, indent=4)
    except Exception as e:
        # 尝试回退到默认路径
        fallback = _DEFAULT_RUNTIME_PATH
        if path != fallback:
            try:
                fallback.parent.mkdir(parents=True, exist_ok=True)
                with fallback.open("w", encoding="utf-8") as f:
                    json.dump(_RUNTIME_ENV, f, ensure_ascii=False, indent=4)
                os.environ["RUNTIME_ENV_PATH"] = str(fallback)
                return
            except Exception:
                pass
        print(f"Warning: Failed to write runtime_env.json: {e}")

def extract_llm_conversation(conversation: dict, output_type: str):
    """Extract information from a conversation payload.

    Args:
        conversation: A mapping that includes 'messages' (list of dicts or objects with attributes).
        output_type: 'final' to return the model's final answer content; 'all' to return the full messages list.

    Returns:
        For 'final': the final assistant content string if found, otherwise None.
        For 'all': the original messages list (or empty list if missing).
    """

    def get_field(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def get_nested(obj, path, default=None):
        current = obj
        for key in path:
            current = get_field(current, key, None)
            if current is None:
                return default
        return current

    messages = get_field(conversation, "messages", []) or []

    if output_type == "all":
        return messages

    if output_type == "final":
        # Prefer the last message with finish_reason == 'stop' and non-empty content.
        for msg in reversed(messages):
            finish_reason = get_nested(msg, ["response_metadata", "finish_reason"])
            content = get_field(msg, "content")
            if finish_reason == "stop" and isinstance(content, str) and content.strip():
                return content

        # Fallback: last AI-like message with non-empty content and not a tool call.
        for msg in reversed(messages):
            content = get_field(msg, "content")
            additional_kwargs = get_field(msg, "additional_kwargs", {}) or {}
            tool_calls = None
            if isinstance(additional_kwargs, dict):
                tool_calls = additional_kwargs.get("tool_calls")
            else:
                tool_calls = getattr(additional_kwargs, "tool_calls", None)

            is_tool_invoke = isinstance(tool_calls, list)
            # Tool messages often have 'tool_call_id' or 'name' (tool name)
            has_tool_call_id = get_field(msg, "tool_call_id") is not None
            tool_name = get_field(msg, "name")
            is_tool_message = has_tool_call_id or isinstance(tool_name, str)

            if not is_tool_invoke and not is_tool_message and isinstance(content, str) and content.strip():
                return content

        return None

    raise ValueError("output_type must be 'final' or 'all'")


def extract_llm_tool_messages(conversation: dict):
    """Return all ToolMessage-like entries from the conversation.

    A ToolMessage is identified heuristically by having either:
      - a non-empty 'tool_call_id', or
      - a string 'name' (tool name) and no 'finish_reason' like normal AI messages

    Supports both dict-based and object-based messages.
    """

    def get_field(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def get_nested(obj, path, default=None):
        current = obj
        for key in path:
            current = get_field(current, key, None)
            if current is None:
                return default
        return current

    messages = get_field(conversation, "messages", []) or []
    tool_messages = []
    for msg in messages:
        tool_call_id = get_field(msg, "tool_call_id")
        name = get_field(msg, "name")
        finish_reason = get_nested(msg, ["response_metadata", "finish_reason"])  # present for AIMessage
        # Treat as ToolMessage if it carries a tool_call_id, or looks like a tool response
        if tool_call_id or (isinstance(name, str) and not finish_reason):
            tool_messages.append(msg)
    return tool_messages


# 向后兼容别名（已弃用，将在未来版本中移除）
def get_config_value(key: str, default=None):
    """Deprecated: Use get_runtime_config_value instead."""
    return get_runtime_config_value(key, default)

def write_config_value(key: str, value: any):
    """Deprecated: Use write_runtime_config_value instead."""
    return write_runtime_config_value(key, value)

def extract_conversation(conversation: dict, output_type: str):
    """Deprecated: Use extract_llm_conversation instead."""
    return extract_llm_conversation(conversation, output_type)

def extract_tool_messages(conversation: dict):
    """Deprecated: Use extract_llm_tool_messages instead."""
    return extract_llm_tool_messages(conversation)

