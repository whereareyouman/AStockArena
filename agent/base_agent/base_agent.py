"""
BaseAgent class - Base class for trading agents
Encapsulates core functionality including local tool management, AI agent creation, and trading execution
"""

import copy
import os
import sys
import json
import asyncio
import threading
import random
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import threading

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Import project tools
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

load_dotenv()

from data_manager import DataManager
from agent.shared_prefetch import SharedPrefetchCoordinator
from tools.general_tools import extract_conversation, extract_tool_messages, get_config_value, write_config_value
from tools.json_file_manager import safe_read_json
from tools.price_tools import (
    add_no_trade_record,
    get_yesterday_date,
    get_latest_position,
    normalize_decision_time,
    normalize_positions,
    normalize_symbol,
    strip_exchange_prefix,
    upsert_position_record,
    get_price_limits,
)
from prompts.agent_prompt import get_agent_system_prompt, STOP_SIGNAL

# 并发安全：用于保护 news.csv 写入
NEWS_FILE_LOCK = threading.Lock()


class BaseAgent:
    """
    Base class for trading agents
    
    Main functionalities:
    1. Local tool management (DataManager and price tools)
    2. AI agent creation and configuration
    3. Trading execution and decision loops
    4. Logging and management
    5. Position and configuration management
    """
    
    # 科创板代表性股票（STAR Market Stocks）
    DEFAULT_STOCK_SYMBOLS = [
        "SH688008",  # 澜起科技 *
        "SH688111",  # 金山办公 *
        "SH688009",  # 中国通号 *
        "SH688981",  # 中芯国际 *
        "SH688256",  # 寒武纪 *
        "SH688271",  # 联影医疗 *
        "SH688047",  # 龙芯中科 *
        "SH688617",  # 惠泰医疗 *
        "SH688303",  # 大全能源 *
        "SH688180",  # 君实生物 *
    ]
    MINI_CANDLE_COUNT = 6
    RECENT_CLOSE_COUNT = 6
    LIMIT_ORDER_SUCCESS_RATE = 0.10
    LIMIT_THRESHOLD_RATIO = 0.999
    def __init__(
        self,
        signature: str,
        basemodel: str,
        stock_symbols: Optional[List[str]] = None,
        stock_json_path: str = "./data/ai_stock_data.json",
        news_csv_path: str = "./data/news.csv",
        macro_csv_path: Optional[str] = None,
        log_path: Optional[str] = None,
        max_steps: int = 10,
        max_retries: int = 3,
        base_delay: float = 0.5,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        safety_settings: Optional[Dict[str, str]] = None,
        initial_cash: float = 1000000.0,
        init_date: Optional[str] = None,
        trading_rules: Optional[Dict[str, float]] = None,
        risk_management: Optional[Dict[str, float]] = None,
        force_replay: bool = False,
    ):
        """
        Initialize BaseAgent
        
        Args:
            signature: Agent signature/name
            basemodel: Base model name
            stock_symbols: List of stock symbols
            stock_json_path: Path to stock price JSON file (ai_stock_data.json)
            news_csv_path: Path to news CSV file
            macro_csv_path: Path to macro news CSV file
            log_path: Log path, defaults to ./data/agent_data
            max_steps: Maximum reasoning steps
            max_retries: Maximum retry attempts
            base_delay: Base delay time for retries
            openai_base_url: OpenAI API base URL
            openai_api_key: OpenAI API key
            google_api_key: Google Gemini API key
            safety_settings: Google Gemini safety settings
            initial_cash: Initial cash amount
            init_date: Initialization date
            trading_rules: Dictionary with trading rule settings
            risk_management: Dictionary with risk management settings
        """
        self.signature = signature
        self.basemodel = basemodel
        self.stock_symbols = stock_symbols or self.DEFAULT_STOCK_SYMBOLS
        self.allowed_symbols = {
            normalize_symbol(sym) for sym in self.stock_symbols if normalize_symbol(sym)
        }

        def _resolve_path(input_path: Optional[str], default_subpath: Optional[str]) -> Optional[str]:
            target = input_path or default_subpath
            if not target:
                return None
            if os.path.isabs(target):
                return target
            cleaned = target.lstrip("./")
            return os.path.join(project_root, cleaned)

        self.stock_json_path = _resolve_path(stock_json_path, "data/ai_stock_data.json")
        self.news_csv_path = _resolve_path(news_csv_path, "data/news.csv")
        self.macro_csv_path = _resolve_path(macro_csv_path, None)
        self._prefetched_news: Dict[str, Dict[str, Any]] = {}
        self._prefetched_prices: Dict[str, Dict[str, Any]] = {}
        self._prefetched_indicators: Dict[str, Dict[str, Any]] = {}
        self._last_prefetch_bundle: Optional[Dict[str, Any]] = None
        self._current_snapshot_info: Dict[str, Any] = {}
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.initial_cash = initial_cash
        if not init_date:
            raise ValueError("init_date must be provided for BaseAgent")
        self.init_date = init_date
        self.force_replay = force_replay
        
        # Load trading rules and risk management from config
        self.trading_rules = trading_rules or {}
        self.risk_management = risk_management or {}
        
        # Set log path
        self.base_log_path = _resolve_path(log_path, "data/agent_data")
        os.makedirs(self.base_log_path, exist_ok=True)
        shared_override = os.getenv("SHARED_PREFETCH_DIR")
        self.shared_prefetch_root = _resolve_path(shared_override, "data/agent_data/shared")
        if self.shared_prefetch_root:
            os.makedirs(self.shared_prefetch_root, exist_ok=True)
        self.prefetch_coordinator = SharedPrefetchCoordinator(base_dir=self.shared_prefetch_root)
        
        # Set OpenAI configuration
        if openai_base_url is None:
            if self.basemodel.startswith("qwen"):
                self.openai_base_url = (
                    os.getenv("QWEN_API_BASE")
                    or os.getenv("DASHSCOPE_API_BASE")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            else:
                self.openai_base_url = os.getenv("OPENAI_API_BASE")
        else:
            self.openai_base_url = openai_base_url

        if openai_api_key is None:
            if self.basemodel.startswith("qwen"):
                self.openai_api_key = (
                    os.getenv("QWEN_API_KEY")
                    or os.getenv("DASHSCOPE_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                )
            else:
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.openai_api_key = openai_api_key
        
        # Set Google Gemini configuration
        if google_api_key is None:
            self.google_api_key = os.getenv("GEMINI_API_KEY")
        else:
            self.google_api_key = google_api_key
        self.safety_settings = safety_settings
        
        # 预先清理新闻缓存，避免加载非白名单股票
        if self.news_csv_path:
            self._purge_news_csv(self.news_csv_path)
        
        # Initialize DataManager（现在不需要CSV，完全依赖TinySoft实时数据）
        try:
            self.dm = DataManager(
                stock_csv_path=None,  # 不再使用CSV，完全依赖TinySoft实时获取
                news_csv_path=self.news_csv_path,
                macro_csv_path=self.macro_csv_path
            )
            print(f"✅ DataManager 初始化成功（使用TinySoft实时数据源）")
            if self.dm.news_df is not None:
                self.dm.news_df = self._filter_allowed_news_df(self.dm.news_df)
        except Exception as e:
            print(f"❌ DataManager 初始化失败: {e}")
            self.dm = None
        
        # Initialize components
        self.local_tools: List = []  # 本地 DataManager 工具
        self.tools: List = []  # 合并后的所有工具
        self.model: Optional[Any] = None
        self.agent: Optional[Any] = None
        
        # Data paths
        self.data_path = os.path.join(self.base_log_path, self.signature)
        self.position_file = os.path.join(self.data_path, "position", "position.jsonl")
        
        # --- 并发运行时上下文（替代对全局 runtime_env.json 的读取依赖） ---
        self.runtime_context: Dict[str, Any] = {
            "TODAY_DATE": None,
            "CURRENT_TIME": None,
            "DECISION_COUNT": 0
        }
    
    def _reset_agent_storage(self) -> None:
        """Remove existing agent data directory (positions + logs) when replaying."""
        agent_path = Path(self.data_path)
        if agent_path.exists():
            shutil.rmtree(agent_path, ignore_errors=True)
    
    def _get_context_value(self, key: str):
        """优先读取实例级上下文，其次回退到全局配置（兼容旧逻辑）"""
        if key in self.runtime_context and self.runtime_context[key] is not None:
            return self.runtime_context[key]
        return get_config_value(key)
    
    def _is_allowed_symbol(self, symbol: Optional[str], *, allow_sell_existing: bool = False) -> bool:
        """
        判断符号是否在允许列表中。
        allow_sell_existing=True 时，如果符号当前持仓中存在，也允许。
        """
        normalized = normalize_symbol(symbol) if symbol else None
        if normalized and normalized in self.allowed_symbols:
            return True
        if allow_sell_existing and normalized:
            latest_positions, _, _ = get_latest_position(
                self._get_context_value("TODAY_DATE") or self.init_date,
                self.signature
            )
            return normalized in latest_positions
        return False

    def _allowed_symbol_list(self) -> List[str]:
        return sorted(sym for sym in self.allowed_symbols if sym)

    def _filter_allowed_news_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty or 'symbol' not in df.columns:
            return df
        filtered = df[df['symbol'].astype(str).apply(
            lambda sym: self._is_allowed_symbol(normalize_symbol(sym))
        )]
        return filtered

    def _filter_allowed_news_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for item in records:
            sym = normalize_symbol(item.get("symbol"))
            if self._is_allowed_symbol(sym):
                filtered.append(item)
        return filtered

    def _prefetch_all_news(self, today_date: str, current_time: str, max_retries: int = 2) -> None:
        print("📰 Prefetching news for all whitelisted symbols...")
        cutoff = pd.to_datetime(today_date) - pd.Timedelta(days=3)
        for sym in self.stock_symbols:
            normalized = normalize_symbol(sym)
            if not normalized:
                continue
            raw_query = f"{strip_exchange_prefix(normalized) or normalized} 最新消息"
            result_str = self.search_stock_news(raw_query, max_retries=max_retries)
            try:
                payload = json.loads(result_str)
            except Exception:
                continue
            if not payload.get("success"):
                continue

            entries = payload.get("historical_news", []) + payload.get("realtime_news", [])
            entries = self._filter_allowed_news_records(entries)
            recent_items: List[Dict[str, Any]] = []
            for item in entries:
                title = str(item.get("title") or item.get("新闻标题") or "").strip()
                if not title:
                    continue
                publish_raw = item.get("publish_time") or item.get("发布时间")
                publish_dt = pd.to_datetime(publish_raw, errors="coerce")
                if publish_dt is not None and publish_dt.tzinfo is not None:
                    publish_dt = publish_dt.tz_convert("Asia/Shanghai").tz_localize(None)
                if publish_dt is not None and publish_dt < cutoff:
                    continue
                display_time = publish_dt.strftime("%Y-%m-%d %H:%M") if publish_dt is not None else str(publish_raw)
                recent_items.append({
                    "title": title[:120],
                    "publish_time": display_time
                })

            recent_items.sort(key=lambda item: item.get("publish_time", ""), reverse=True)
            truncated_items = recent_items[:5]
            self._prefetched_news[normalized] = {
                "news": truncated_items,
                "count": len(truncated_items)
            }

    def _prefetch_all_prices(self, current_time: str) -> None:
        print("💹 Prefetching hourly price snapshots...")
        for sym in self.stock_symbols:
            normalized = normalize_symbol(sym)
            if not normalized:
                continue
            try:
                payload_str = self.get_hourly_stock_data(normalized, current_time)
                payload = json.loads(payload_str)
            except Exception as e:
                print(f"⚠️ 预抓取价格失败：{normalized} - {e}")
                continue
            if "summary" in payload:
                self._prefetched_prices[normalized] = payload

    def _prefetch_all_indicators(self, today_date: str, current_time: str) -> None:
        print("📈 Prefetching technical indicators...")
        end_date = current_time.split(" ")[0]
        for sym in self.stock_symbols:
            normalized = normalize_symbol(sym)
            if not normalized:
                continue
            try:
                payload_str = self.get_technical_indicators(normalized, end_date, lookback_days=10)
                payload = json.loads(payload_str)
            except Exception as e:
                print(f"⚠️ 预抓取指标失败：{normalized} - {e}")
                continue
            indicators = payload.get("indicators") if isinstance(payload, dict) else None
            if indicators:
                self._prefetched_indicators[normalized] = payload

    def _build_observation_summary(self) -> str:
        if not self.stock_symbols:
            return "  • (no whitelisted symbols configured)"

        lines: List[str] = []
        for sym in self.stock_symbols:
            normalized = normalize_symbol(sym)
            if not normalized:
                continue

            price_payload = self._prefetched_prices.get(normalized, {})
            price_summary = price_payload.get("summary") if isinstance(price_payload, dict) else None
            price_text = "Px: -"
            if price_summary:
                close = price_summary.get("close")
                change_pct = price_summary.get("change_pct")
                if close is not None:
                    price_text = f"Px ¥{close:,.2f}"
                if change_pct is not None:
                    sign = "+" if change_pct >= 0 else ""
                    price_text += f" ({sign}{change_pct:.2f}%)"

            indicator_payload = self._prefetched_indicators.get(normalized, {})
            indicators = indicator_payload.get("indicators") if isinstance(indicator_payload, dict) else None
            indicator_parts: List[str] = []
            if indicators:
                if indicators.get("SMA_10") is not None:
                    indicator_parts.append(f"SMA10 {indicators['SMA_10']:.2f}")
                if indicators.get("RSI_10") is not None:
                    indicator_parts.append(f"RSI10 {indicators['RSI_10']:.1f}")
                macd_val = indicators.get("MACD_12_26_9")
                if macd_val is not None:
                    indicator_parts.append(f"MACD {macd_val:.2f}")
            indicator_text = "; ".join(indicator_parts) if indicator_parts else "Indicators: -"

            news_payload = self._prefetched_news.get(normalized, {})
            news_titles = [item.get("title", "") for item in news_payload.get("news", [])]
            news_titles = [title for title in news_titles if title]
            if news_titles:
                news_text = " | ".join(news_titles[:2])
            else:
                news_text = "no recent news (≤3d)"

            line = f"  • {normalized}: {price_text}; {indicator_text}; News: {news_text}"
            lines.append(line)

        if not lines:
            return "  • (prefetch unavailable)"
        return "\n".join(lines)

    def _symbols_signature(self) -> str:
        normalized = sorted(sym for sym in self.allowed_symbols if sym)
        return "|".join(normalized)

    def _collect_prefetch_bundle(
        self,
        today_date: str,
        current_time: str,
        decision_count: int,
    ) -> Dict[str, Any]:
        """
        Run all prefetch helpers and return a serializable snapshot bundle
        for shared caching/logging.
        """
        self._prefetched_news.clear()
        self._prefetched_prices.clear()
        self._prefetched_indicators.clear()
        self._prefetch_all_news(today_date, current_time, max_retries=2)
        self._prefetch_all_prices(current_time)
        self._prefetch_all_indicators(today_date, current_time)

        snapshot_id = f"{today_date}_{current_time.replace(':', '-').replace(' ', '_')}"
        summary = self._build_observation_summary()
        now_iso = datetime.now(timezone.utc).isoformat()

        bundle: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "schema_version": 1,
            "today_date": today_date,
            "decision_time": current_time,
            "decision_count": decision_count,
            "generated_at": now_iso,
            "source_agent": self.signature,
            "symbols": list(self.stock_symbols),
            "normalized_symbols": sorted(sym for sym in self.allowed_symbols if sym),
            "symbols_signature": self._symbols_signature(),
            "news": copy.deepcopy(self._prefetched_news),
            "prices": copy.deepcopy(self._prefetched_prices),
            "indicators": copy.deepcopy(self._prefetched_indicators),
            "observation_summary": summary,
            "prefetch_config": {
                "news_lookback_days": 3,
                "indicator_lookback_days": 10,
                "hourly_candles_expected": 40,
            },
        }
        self._last_prefetch_bundle = bundle
        return bundle

    def _apply_prefetch_bundle(self, bundle: Dict[str, Any]) -> str:
        """
        Load prefetched structures from a snapshot bundle back into the agent.
        Returns the observation summary text.
        """
        self._last_prefetch_bundle = bundle
        news_payload = bundle.get("news") or {}
        prices_payload = bundle.get("prices") or {}
        indicators_payload = bundle.get("indicators") or {}
        self._prefetched_news = copy.deepcopy(news_payload)
        self._prefetched_prices = copy.deepcopy(prices_payload)
        self._prefetched_indicators = copy.deepcopy(indicators_payload)
        summary = bundle.get("observation_summary")
        if not summary:
            summary = self._build_observation_summary()
        return summary

    def _purge_news_csv(self, csv_path: Optional[str]) -> None:
        if not csv_path or not os.path.exists(csv_path):
            return
        with NEWS_FILE_LOCK:
            df = None
            for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功读取 {csv_path} 以进行清理")
                    break
                except Exception:
                    continue
            if df is None or df.empty:
                return
            df = self._sanitize_news_dataframe(df)
            filtered_df = self._filter_allowed_news_df(df)
            if filtered_df is None:
                return
            if len(filtered_df) != len(df):
                filtered_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"🧹 已清理 {csv_path} 中的非白名单新闻记录")
    
    # --- 本地工具函数定义 (DM Functions) ---
    
    def get_current_stock_prices(self, symbols: List[str], target_time: str) -> str:
        """
        获取多个股票在指定时间点或之前最新的收盘价格。
        
        Args:
            symbols (List[str]): 股票代码列表(例如 ["MSFT", "AAPL"])。
            target_time (str): 目标时间(YYYY-MM-DD HH:MM:SS)。
        
        Returns:
            str: JSON 字符串格式的字典,映射股票代码到价格(如果未找到则为 null)。
        """
        if not self.dm:
            return json.dumps({"error": "DataManager 未初始化"})
        try:
            prices_dict = self.dm.get_prices_at(symbols=symbols, target_time=target_time)
            return json.dumps(prices_dict)
        except Exception as e:
            return json.dumps({"error": f"获取当前价格时出错: {str(e)}"})
    
    def get_hourly_stock_data(self, symbol: str, end_time: str, lookback_hours: Optional[int] = 24) -> str:
        """
        获取单个股票的小时线数据（60分钟 K 线）以及简要摘要。
        摘要包含：最新一根 K 线、上一根收盘价、最近若干收盘价；同时返回一小段最新的小时 K 线列表。
        """
        if not self.dm:
            return json.dumps({"error": "DataManager 未初始化"})

        normalized_symbol = normalize_symbol(symbol)
        if not self._is_allowed_symbol(normalized_symbol, allow_sell_existing=True):
            return json.dumps({
                "error": "该股票不在允许的研究名单中",
                "allowed_symbols": self._allowed_symbol_list(),
                "symbol": normalized_symbol
            }, ensure_ascii=False)

        def _to_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        try:
            window = lookback_hours or 24
            plain_symbol = strip_exchange_prefix(normalized_symbol) if normalized_symbol else symbol
            query_symbol = plain_symbol or symbol
            df = self.dm.get_hourly_stock_data(
                symbol=query_symbol,
                end_date=end_time, 
                lookback_hours=window
            )
            if df.empty:
                fallback_df = self._build_single_price_dataframe(normalized_symbol or symbol, end_time)
                if fallback_df is not None:
                    df = fallback_df
                else:
                    return json.dumps({"error": f"未找到 {normalized_symbol or symbol} 的小时线数据"})

            df = df.sort_index()
            latest = df.iloc[-1]
            latest_ts = latest.name
            latest_close = _to_float(latest.get("close"))
            prev_close = _to_float(df.iloc[-2]["close"]) if len(df) > 1 and "close" in df.columns else None

            recent_closes: List[Optional[float]] = []
            if "close" in df.columns:
                closes_series = df["close"].dropna().tail(min(len(df), self.RECENT_CLOSE_COUNT))
                recent_closes = [_to_float(val) for val in closes_series]

            summary: Dict[str, Any] = {
                "timestamp": str(latest_ts),
                "open": _to_float(latest.get("open", latest_close)),
                "high": _to_float(latest.get("high", latest_close)),
                "low": _to_float(latest.get("low", latest_close)),
                "close": latest_close,
                "volume": _to_float(latest.get("volume")),
                "previous_close": prev_close,
                "recent_closes": recent_closes,
            }
            if prev_close is not None and latest_close is not None and prev_close != 0:
                summary["change"] = round(latest_close - prev_close, 4)
                summary["change_pct"] = round(((latest_close - prev_close) / prev_close) * 100, 4)

            mini_count = min(len(df), self.MINI_CANDLE_COUNT)
            candles: List[Dict[str, Any]] = []
            candles_df = df.tail(mini_count).reset_index()
            for _, row in candles_df.iterrows():
                candles.append({
                    "timestamp": str(row.get("timestamp")),
                    "open": _to_float(row.get("open", row.get("close"))),
                    "high": _to_float(row.get("high", row.get("close"))),
                    "low": _to_float(row.get("low", row.get("close"))),
                    "close": _to_float(row.get("close")),
                    "volume": _to_float(row.get("volume"))
                })

            payload = {
                "symbol": symbol,
                "lookback_hours": window,
                "total_candles_available": int(len(df)),
                "summary": summary,
                "candles": candles
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"获取 {symbol} 小时线数据时出错: {str(e)}"})

    def _build_single_price_dataframe(self, symbol: str, timestamp: str) -> Optional[pd.DataFrame]:
        """
        构造仅含一条记录的小时线 DataFrame，使用前一交易日收盘价或 DataManager 可获得的价格。
        """
        normalized_symbol = normalize_symbol(symbol) or symbol
        plain_symbol = strip_exchange_prefix(normalized_symbol) or normalized_symbol

        try:
            ts = pd.to_datetime(timestamp)
        except Exception:
            ts = pd.Timestamp(datetime.now())

        date_str = ts.strftime("%Y-%m-%d")
        fallback_price = self._get_previous_close(normalized_symbol, date_str)

        if (fallback_price is None) and self.dm:
            try:
                fallback_price = self.dm.get_price_at(plain_symbol, ts)
            except Exception:
                fallback_price = None

        if fallback_price is None:
            return None

        record = {
            "timestamp": ts,
            "symbol": plain_symbol,
            "open": float(fallback_price),
            "high": float(fallback_price),
            "low": float(fallback_price),
            "close": float(fallback_price),
            "volume": 0.0,
        }
        df = pd.DataFrame([record])
        df.set_index("timestamp", inplace=True)
        return df
    
    def get_technical_indicators(self, symbol: str, end_date: str, lookback_days: int = 10) -> str:
        """
        获取技术指标：读取历史小时线指标 + 计算实时小时线指标 + 保存更新到 ai_stock_data.json
        
        Args:
            symbol (str): 股票代码
            end_date (str): 结束日期 (YYYY-MM-DD) 或结束时间 (YYYY-MM-DD HH:MM:SS)
            lookback_days (int): 用于计算指标的历史数据天数，默认30天（转换为小时数）
            
        Returns:
            str: JSON字符串，包含历史小时线指标和实时小时线指标
        """
        if not self.dm:
            return json.dumps({"error": "DataManager 未初始化"})
        
        normalized_symbol = normalize_symbol(symbol)
        if not self._is_allowed_symbol(normalized_symbol, allow_sell_existing=True):
            return json.dumps({
                "error": "该股票不在允许的研究名单中",
                "allowed_symbols": self._allowed_symbol_list(),
                "symbol": normalized_symbol
            }, ensure_ascii=False)
        
        try:
            # 1. 读取历史小时线指标与行情（从 ai_stock_data.json，使用JsonFileManager）
            historical_indicators: List[Dict[str, Any]] = []
            historical_price_data: List[Dict[str, Any]] = []
            ai_stock_data_path = self.stock_json_path or os.path.join(project_root, "data", "ai_stock_data.json")
            from tools.json_file_manager import safe_read_json
            
            all_data = safe_read_json(ai_stock_data_path, default={})
            stock_entry: Optional[Dict[str, Any]] = None
            for key in [symbol, f"SH{symbol}", f"SZ{symbol}"]:
                entry = all_data.get(key)
                if entry:
                    stock_entry = entry
                    break
            
            def _reload_stock_entry() -> Optional[Dict[str, Any]]:
                refreshed = safe_read_json(ai_stock_data_path, default={})
                for candidate in [symbol, f"SH{symbol}", f"SZ{symbol}"]:
                    entry = refreshed.get(candidate)
                    if entry:
                        return entry
                return None

            if stock_entry:
                if "小时线指标" in stock_entry:
                    historical_indicators = stock_entry["小时线指标"] or []
                    print(f"📚 读取到 {len(historical_indicators)} 条历史小时线技术指标（股票：{symbol}）")
                if "小时线行情" in stock_entry:
                    historical_price_data = stock_entry["小时线行情"] or []
                    print(f"📚 读取到 {len(historical_price_data)} 条历史小时线行情（股票：{symbol}）")
            else:
                print(f"⚠️ 未在 {ai_stock_data_path} 找到 {symbol} 的历史记录，尝试从 TinySoft 回填。")

            max_expected_candles = lookback_days * 4
            if not historical_price_data or len(historical_price_data) < max_expected_candles:
                try:
                    ndays = max(lookback_days, 10)
                    self.dm.save_ts_data(symbols=[symbol], ndays=ndays, out_path=ai_stock_data_path)
                    stock_entry = _reload_stock_entry()
                    if stock_entry and "小时线行情" in stock_entry:
                        historical_price_data = stock_entry["小时线行情"] or []
                        print(f"📡 通过 TinySoft 回填 {len(historical_price_data)} 条历史小时线行情（股票：{symbol}）")
                except Exception as e:
                    print(f"⚠️ 回填小时线行情失败：{e}")

            if not historical_indicators and stock_entry and "小时线指标" in stock_entry:
                historical_indicators = stock_entry["小时线指标"] or []
                if historical_indicators:
                    print(f"📚 读取到 {len(historical_indicators)} 条历史小时线技术指标（股票：{symbol}）")
            
            # 2. 组装指标字段白名单
            indicator_keys = [
                'SMA_10',
                'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                'RSI_10',
                'BBL_10_2.0', 'BBM_10_2.0', 'BBU_10_2.0'
            ]

            def build_indicator_payload(source_label: str, indicator_dict: Dict[str, Any], timestamp_hint: Optional[Any] = None, include_saved_path: bool = False) -> str:
                summary = {key: indicator_dict.get(key) for key in indicator_keys}
                payload: Dict[str, Any] = {
                    "source": source_label,
                    "symbol": symbol,
                    "timestamp": str(timestamp_hint) if timestamp_hint else str(end_date),
                    "indicators": summary
                }
                if include_saved_path:
                    payload["saved_to"] = ai_stock_data_path
                return json.dumps(payload, ensure_ascii=False)
            
            # 3. 获取实时小时线数据（转换为小时数：每天4小时交易时间，乘以天数）
            lookback_hours = lookback_days * 4  # 每天4小时交易时间（9:30-11:30, 13:00-15:00）
            # 如果 end_date 只包含日期，添加当前时间
            if len(end_date) == 10:  # YYYY-MM-DD
                end_time = f"{end_date} 15:00:00"  # 使用收盘时间
            else:
                end_time = end_date
            
            df_realtime = self.dm.get_hourly_stock_data(
                symbol=symbol, 
                end_date=end_time, 
                lookback_hours=lookback_hours
            )
            
            # 4. 合并历史+实时数据
            df = pd.DataFrame()  # 最终合并后的DataFrame
            
            # 如果有历史价格数据，先转换为DataFrame
            if historical_price_data:
                if len(historical_price_data) > max_expected_candles:
                    historical_price_data = historical_price_data[-max_expected_candles:]
                try:
                    df_historical = pd.DataFrame(historical_price_data)
                    # 转换日期列为datetime（naive，无时区）
                    df_historical['timestamp'] = pd.to_datetime(df_historical['date'], utc=False)
                    # 如果转换后有时区，移除时区信息（转为naive）
                    if df_historical['timestamp'].dt.tz is not None:
                        df_historical['timestamp'] = df_historical['timestamp'].dt.tz_localize(None)
                    # 重命名列以匹配DataFrame格式
                    df_historical = df_historical.rename(columns={
                        'close': 'close',
                        'vol': 'volume'
                    })
                    # 只保留需要的列
                    if 'close' in df_historical.columns:
                        df_historical = df_historical[['timestamp', 'close', 'volume']]
                        df_historical.set_index('timestamp', inplace=True)
                        df = df_historical.copy()
                        print(f"📚 已加载 {len(df)} 条历史小时线价格数据（naive datetime）")
                except Exception as e:
                    print(f"⚠️ 转换历史价格数据失败: {e}")
            
            # 合并实时数据（确保时区一致）
            if not df_realtime.empty:
                # 只使用close和volume列
                df_realtime_subset = df_realtime[['close', 'volume']].copy()
                # 移除时区信息（转为naive datetime，避免时区比较错误）
                if df_realtime_subset.index.tz is not None:
                    df_realtime_subset.index = df_realtime_subset.index.tz_localize(None)
                # 合并（去重，保留最新的）
                if not df.empty:
                    # 合并，实时数据覆盖历史数据中的相同时间戳
                    df = pd.concat([df, df_realtime_subset])
                    df = df[~df.index.duplicated(keep='last')]  # 保留最新的
                    df = df.sort_index()
                else:
                    df = df_realtime_subset
                print(f"📡 已合并 {len(df_realtime)} 条实时小时线价格数据，总计 {len(df)} 条（已统一为naive datetime）")
            
            # 限制窗口大小
            if not df.empty and len(df) > max_expected_candles:
                df = df.tail(max_expected_candles)
            
            # 检查合并后的数据量（小时线至少需要10个交易日 * 4小时 = 40条）
            min_hours_needed = max_expected_candles
            if df.empty:
                # 如果合并后仍为空，返回历史指标
                if historical_indicators:
                    print(f"⚠️ 无价格数据，返回历史指标")
                    latest_hist = historical_indicators[-1] if historical_indicators else {}
                    ts_hint = latest_hist.get("timestamp") or latest_hist.get("date")
                    return build_indicator_payload("historical_only", latest_hist, ts_hint)
                else:
                    return json.dumps({"error": f"未找到 {symbol} 的股票数据"})
            
            # 检查数据量是否足够计算技术指标
            if len(df) < min_hours_needed:
                if historical_indicators:
                    print(f"⚠️ 合并后数据不足（{len(df)}条 < {min_hours_needed}条），返回历史指标")
                    latest_hist = historical_indicators[-1] if historical_indicators else {}
                    ts_hint = latest_hist.get("timestamp") or latest_hist.get("date")
                    return build_indicator_payload("historical_only", latest_hist, ts_hint)
                else:
                    return json.dumps({
                        "error": f"数据量不足，需要至少{min_hours_needed}条小时线数据，合并后只有{len(df)}条"
                    }, ensure_ascii=False)
            
            # 5. 计算技术指标（使用合并后的完整小时线数据）
            try:
                # 确保数据按时间排序
                df = df.sort_index()
                
                # 确保DataFrame有.ta属性（pandas_ta需要正确初始化）
                if not hasattr(df, 'ta'):
                    # 如果.ta属性不存在，尝试重新导入并注册
                    try:
                        import pandas_ta as ta
                        # pandas_ta通过monkey patch的方式添加到DataFrame，确保已加载
                        if not hasattr(pd.DataFrame, 'ta'):
                            import pandas_ta.core as pta
                    except Exception as e:
                        print(f"⚠️ pandas_ta初始化失败: {e}")
                        raise Exception(f"pandas_ta不可用: {e}")
                
                # 确保有足够的列用于计算技术指标
                if 'close' not in df.columns:
                    raise Exception("DataFrame缺少'close'列")
                
                # 计算技术指标（基于小时线数据）
                # 使用10天参数计算指标
                df.ta.sma(length=10, append=True)
                df.ta.macd(append=True)
                df.ta.rsi(length=10, append=True)
                df.ta.bbands(length=10, append=True)
                print(f"📊 基于合并小时线数据（历史+实时，共{len(df)}条）计算技术指标（10天）")
            except Exception as e:
                print(f"⚠️ 计算技术指标失败: {e}")
                if historical_indicators:
                    latest_hist = historical_indicators[-1] if historical_indicators else {}
                    ts_hint = latest_hist.get("timestamp") or latest_hist.get("date")
                    return build_indicator_payload("historical_only", latest_hist, ts_hint)
                else:
                    return json.dumps({"error": f"计算技术指标失败: {str(e)}"})
            
            # 6. 提取最新的指标
            latest_row = df.iloc[-1]
            latest_indicators: Dict[str, Any] = {}
            for key in indicator_keys:
                value = latest_row.get(key) if hasattr(latest_row, "get") else latest_row[key] if key in latest_row else None
                latest_indicators[key] = value
            
            # 清理 NaN 值
            for key, value in latest_indicators.items():
                if pd.isna(value):
                    latest_indicators[key] = None

            print(f"📊 已计算技术指标（基于合并小时线数据：历史+实时，共{len(df)}条）")
            
            # 7. 保存到 ai_stock_data.json（使用 DataManager 的 save_ts_data，自动合并历史）
            try:
                self.dm.save_ts_data(symbols=[symbol], ndays=60, out_path=ai_stock_data_path)
                print(f"💾 已更新小时线技术指标到 {ai_stock_data_path}（合并历史数据）")
                
                # 重新读取合并后的完整指标数据（使用JsonFileManager）
                all_data_updated = safe_read_json(ai_stock_data_path, default={})
                for key in [symbol, f"SH{symbol}", f"SZ{symbol}"]:
                    if key in all_data_updated and "小时线指标" in all_data_updated[key]:
                        # 使用完整的合并后数据
                        historical_indicators = all_data_updated[key]["小时线指标"]
                        print(f"✅ 合并后指标：{len(historical_indicators)} 条")
                        break
            except Exception as e:
                print(f"⚠️ 保存技术指标失败: {e}")
            
            # 8. 返回组合结果（仅最新指标）
            indicator_timestamp = None
            if not df.empty:
                indicator_timestamp = df.index[-1]

            return build_indicator_payload(
                "combined_merged",
                latest_indicators,
                indicator_timestamp,
                include_saved_path=True
            )
        except Exception as e:
            return json.dumps({"error": f"获取技术指标时出错: {str(e)}"})

    def get_latest_position_tool(self, today_date: str) -> str:
        """从持仓文件中获取代理的最新交易持仓。此函数读取代理的持仓文件以找到前一个交易日的持仓。"""
        try:
            positions, _, _ = get_latest_position(today_date, self.signature)
            return json.dumps(positions)
        except Exception as e:
            return json.dumps({"error": f"获取最新持仓时出错: {str(e)}"})
    
    def add_no_trade_record_tool(self, today_date: str) -> str:
        """为当前代理在给定日期记录"无交易"操作。此函数更新代理的持仓文件以延续前一天的持仓。"""
        try:
            # 优先使用实例上下文，避免并发状态污染
            decision_time = self._get_context_value("CURRENT_TIME") or f"{today_date} 00:00:00"
            decision_count_raw = self._get_context_value("DECISION_COUNT")
            decision_count = int(decision_count_raw) if decision_count_raw is not None else 0
            add_no_trade_record(today_date, decision_time, decision_count, self.signature)
            return json.dumps({
                "success": True,
                "action": "no_trade",
                "date": today_date,
                "decision_time": decision_time,
                "agent": self.signature
            })
        except Exception as e:
            return json.dumps({"error": f"添加无交易记录时出错: {str(e)}"})
    
    def buy_stock(self, symbol: str, amount: int) -> str:
        """
        买入股票（使用当前小时级价格）。
        
        此函数模拟股票买入操作，包括：
        1. 获取当前持仓和操作ID
        2. 获取当前小时的股票价格（优先小时级，回退到日线开盘价）
        3. 验证买入条件（现金是否充足）
        4. 更新持仓（增加股票数量，减少现金）
        5. 记录交易到 position.jsonl 文件
        
        Args:
            symbol (str): 股票代码，如 "600519"
            amount (int): 买入数量，必须是100的整数倍
        
        Returns:
            str: JSON 字符串，成功时返回新持仓，失败时返回错误信息
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
            if not normalized_symbol:
                return json.dumps({"error": "无效的股票代码"})
            data_symbol = strip_exchange_prefix(normalized_symbol) or normalized_symbol

            if not self._is_allowed_symbol(normalized_symbol, allow_sell_existing=True):
                return json.dumps({
                    "error": "该股票不在允许的持仓/交易名单中",
                    "allowed_symbols": self._allowed_symbol_list(),
                    "symbol": normalized_symbol
                }, ensure_ascii=False)

            if not self._is_allowed_symbol(normalized_symbol):
                return json.dumps({
                    "error": "该股票不在允许的交易名单中",
                    "allowed_symbols": self._allowed_symbol_list(),
                    "symbol": normalized_symbol
                }, ensure_ascii=False)
            
            # 使用实例上下文，避免从共享 runtime_env.json 读取
            today_date = self._get_context_value("TODAY_DATE")
            current_time = self._get_context_value("CURRENT_TIME")
            decision_time = current_time or f"{today_date} 00:00:00"
            decision_count_raw = self._get_context_value("DECISION_COUNT")
            decision_count = int(decision_count_raw) if decision_count_raw is not None else 0
            if not today_date:
                return json.dumps({"error": "未设置 TODAY_DATE"})
            decision_time = normalize_decision_time(today_date, decision_time)
            
            # 交易单位检查 (100股的整数倍)
            if amount <= 0 or amount % 100 != 0:
                return json.dumps({
                    "error": "买入数量必须是100的整数倍且大于0",
                    "symbol": normalized_symbol,
                    "amount": amount
                })

            # 获取当前持仓和操作ID
            current_position, current_action_id, latest_record = get_latest_position(today_date, self.signature)
            
            # 获取当前时刻的股票价格 - 优先小时级数据
            try:
                this_symbol_price = None
                # 优先尝试获取小时级价格
                if current_time:
                    hourly_data = self.dm.get_hourly_stock_data(symbol=data_symbol, end_date=current_time, lookback_hours=1)
                    if not hourly_data.empty:
                        this_symbol_price = float(hourly_data['close'].iloc[-1])
                        print(f"💹 使用小时级价格: {normalized_symbol} = ¥{this_symbol_price} ({current_time})")
                
                # 如果没有小时级数据，回退到日线开盘价
                if this_symbol_price is None or pd.isna(this_symbol_price):
                    stock_data = self.dm.get_stock_data(symbol=data_symbol, end_date=today_date, lookback_days=1)
                    if stock_data.empty:
                        return json.dumps({"error": f"未找到股票 {normalized_symbol} 的价格数据", "symbol": normalized_symbol, "date": today_date})
                    this_symbol_price = float(stock_data['open'].iloc[-1]) if 'open' in stock_data.columns else float(stock_data['close'].iloc[-1])
                    print(f"💹 使用开盘价: {normalized_symbol} = ¥{this_symbol_price}")
                
                if pd.isna(this_symbol_price) or this_symbol_price <= 0:
                    return json.dumps({"error": f"股票 {normalized_symbol} 的价格数据无效", "symbol": normalized_symbol, "date": today_date, "price": this_symbol_price})
            except Exception as e:
                return json.dumps({"error": f"获取股票价格失败: {str(e)}", "symbol": normalized_symbol, "date": today_date})
            
            limit_info: Optional[Dict[str, float]] = None
            prev_close = self._get_previous_close(normalized_symbol, today_date)
            limit_info = get_price_limits(normalized_symbol, prev_close)
            allowed, reason = self._passes_price_limit_liquidity("sell", this_symbol_price, limit_info)
            if not allowed:
                return json.dumps({
                    "error": reason,
                    "symbol": normalized_symbol,
                    "price": this_symbol_price,
                    "limit_info": limit_info
                }, ensure_ascii=False)
            
            # --- 风险管理检查 ---
            single_stock_max_position = self.risk_management.get("single_stock_max_position", 0.20)
            total_assets = current_position.get("CASH", 0)
            for stock, data in current_position.items():
                if stock != "CASH":
                    # 假设我们需要一个价格来估算当前股票价值，这里用今天的开盘价
                    # 在真实场景中，可能需要更复杂的价格获取逻辑
                    stock_value = data.get("shares", 0) * this_symbol_price # 估算
                    total_assets += stock_value

            required_cash = this_symbol_price * amount
            if (required_cash / total_assets) > single_stock_max_position:
                 return json.dumps({
                    "error": f"单只股票持仓超过上限 ({single_stock_max_position * 100}%)",
                    "symbol": normalized_symbol,
                    "max_allowed_investment": total_assets * single_stock_max_position,
                    "requested_investment": required_cash
                })

            # --- 交易成本计算 ---
            commission_rate = self.trading_rules.get("commission_rate", 0.0003)
            min_commission = self.trading_rules.get("min_commission", 5.0)
            commission = max(required_cash * commission_rate, min_commission)
            total_cost = required_cash + commission

            # 验证买入条件
            cash_left = current_position.get("CASH", 0) - total_cost
            
            if cash_left < 0:
                return json.dumps({
                    "error": "现金不足（已考虑交易费用）！交易不被允许。",
                    "total_cost": total_cost,
                    "cash_available": current_position.get("CASH", 0),
                    "symbol": normalized_symbol,
                    "date": today_date
                })
            
            limit_info: Optional[Dict[str, float]] = None
            prev_close = self._get_previous_close(normalized_symbol, today_date)
            limit_info = get_price_limits(normalized_symbol, prev_close)
            allowed, reason = self._passes_price_limit_liquidity("buy", this_symbol_price, limit_info)
            if not allowed:
                return json.dumps({
                    "error": reason,
                    "symbol": normalized_symbol,
                    "price": this_symbol_price,
                    "limit_info": limit_info
                }, ensure_ascii=False)
            
            # 执行买入操作
            new_position = copy.deepcopy(current_position)
            new_position["CASH"] = cash_left
            
            # 更新持仓（维护加权平均成本）
            if normalized_symbol in new_position:
                existing_entry = new_position[normalized_symbol]
                current_shares = existing_entry.get("shares", 0)
                existing_avg = existing_entry.get("avg_price")
                if existing_avg is None:
                    existing_avg = this_symbol_price
                total_shares = current_shares + amount
                if total_shares > 0:
                    weighted_avg = ((existing_avg * current_shares) + (this_symbol_price * amount)) / total_shares
                else:
                    weighted_avg = this_symbol_price
                existing_entry["shares"] = total_shares
                existing_entry["avg_price"] = weighted_avg
            else:
                new_position[normalized_symbol] = {
                    "shares": amount,
                    "purchase_date": today_date,
                    "avg_price": this_symbol_price
                }

            new_position = normalize_positions(new_position)

            # 记录交易
            record: Dict[str, Any] = {
                "date": today_date,
                "decision_time": decision_time,
                "decision_count": decision_count,
                "this_action": {"action": "buy", "symbol": normalized_symbol, "amount": amount},
                "positions": new_position
            }
            if latest_record and latest_record.get("decision_time") == decision_time:
                record["id"] = latest_record.get("id")
            else:
                record["id"] = current_action_id + 1
            upsert_position_record(self.signature, record)
            
            write_config_value("IF_TRADE", True)
            return json.dumps({
                "success": True,
                "action": "buy",
                "symbol": normalized_symbol,
                "amount": amount,
                "price": this_symbol_price,
                "cost": required_cash,
                "commission": commission,
                "price_limit": limit_info,
                "decision_time": decision_time,
                "decision_count": decision_count,
                "new_position": new_position
            })
        
        except Exception as e:
            return json.dumps({"error": f"买入股票时出错: {str(e)}"})
    
    def search_stock_news(self, query: str, max_retries: int = 3) -> str:
        """
        搜索股票相关的实时新闻 + 读取历史新闻，使用 AKShare，失败重试。
        同时会从 DataManager 读取历史新闻，并将新闻保存到 news.csv
        
        Args:
            query: 搜索关键词（如 "600519 最新消息"）
            max_retries: 最大重试次数，默认3次
        
        Returns:
            str: JSON字符串，包含历史新闻和实时新闻
        """
        # 提取 6 位代码
        symbol = None
        if query.isdigit() and len(query) == 6:
            symbol = query
        else:
            import re
            m = re.search(r'\b(\d{6})\b', query)
            if m:
                symbol = m.group(1)
        if not symbol:
            return json.dumps({'success': False, 'message': '请提供6位A股代码，例如 600519'}, ensure_ascii=False)

        normalized_symbol = normalize_symbol(symbol)
        symbol_for_query = strip_exchange_prefix(normalized_symbol) or symbol

        if not self._is_allowed_symbol(normalized_symbol, allow_sell_existing=True):
            return json.dumps({
                "success": False,
                "message": "该股票不在允许的研究名单中，请聚焦预设的科创板列表。",
                "allowed_symbols": self._allowed_symbol_list()
            }, ensure_ascii=False)

        cached_news = self._prefetched_news.get(normalized_symbol)
        if cached_news:
            return json.dumps(cached_news, ensure_ascii=False)

        today_date = get_config_value("TODAY_DATE")
        current_time = get_config_value("CURRENT_TIME")
        if current_time:
            search_time = current_time
        elif today_date:
            search_time = f"{today_date} 00:00:00"
        else:
            search_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 增量过滤逻辑已移除，改回返回完整历史+实时新闻

        # 1. 读取历史新闻（从 DataManager）
        historical_news = []
        if self.dm and self.dm.news_df is not None:
            try:
                news_df = self.dm.get_news(end_date=today_date, symbols=[symbol_for_query], limit=20)
                news_df = self._filter_allowed_news_df(news_df)
                if news_df is not None and not news_df.empty:
                    # 将所有列转为字符串，避免 Timestamp 序列化问题
                    cleaned_news_df = self._sanitize_news_dataframe(news_df.astype(str))
                    historical_news = self._filter_allowed_news_records(cleaned_news_df.to_dict('records'))
                    print(f"📚 读取到 {len(historical_news)} 条历史新闻（股票：{normalized_symbol or symbol_for_query}）")
            except Exception as e:
                print(f"⚠️ 读取历史新闻失败: {e}")

        # 2. 获取实时新闻（AKShare，带重试）
        realtime_results = []
        csv_path = os.path.join('data', 'news.csv')
        
        # 创建调试文件夹用于保存错误信息
        debug_dir = os.path.join('data', 'debug', 'akshare_errors')
        os.makedirs(debug_dir, exist_ok=True)
        
        for attempt in range(1, max_retries + 1):
            # 初始化请求信息（在所有代码路径中都需要）
            request_info = {
                'symbol': symbol_for_query,
                'function': 'stock_news_em',
                'parameters': {'symbol': symbol_for_query}
            }
            
            try:
                import akshare as ak
                import time
                import traceback
                
                # 重试前添加延迟，避免频率限制
                # 第一次请求也延迟1秒，避免请求过快
                if attempt == 1:
                    wait_time_before = 1  # 第一次延迟1秒
                    print(f"⏳ 请求前等待 {wait_time_before} 秒（避免频率限制）...")
                    time.sleep(wait_time_before)
                else:
                    wait_time_before = (attempt - 1) * 5  # 递增延迟：5秒、10秒
                    print(f"⏳ 请求前等待 {wait_time_before} 秒（避免频率限制）...")
                    time.sleep(wait_time_before)
                
                print(f"📡 尝试获取实时新闻（尝试 {attempt}/{max_retries}，股票代码：{symbol_for_query}）...")
                
                # 验证股票代码格式（应该是6位数字）
                if not symbol_for_query.isdigit() or len(symbol_for_query) != 6:
                    print(f"⚠️ 股票代码格式错误: {symbol_for_query}，应该是6位数字")
                    break
                
                # 尝试拦截 HTTP 响应
                capture_context, request_info_captured = self._capture_akshare_response(symbol_for_query)
                
                # 调用 AKShare API（使用上下文管理器捕获响应）
                # 无论成功还是失败，都会在上下文中更新 request_info_captured
                try:
                    with capture_context:
                        news_df = ak.stock_news_em(symbol=symbol_for_query)
                    # 合并捕获的请求信息（request_info_captured 是可变对象，已在上下文中更新）
                    if request_info_captured:
                        request_info.update(request_info_captured)
                except Exception:
                    # 异常发生时，请求信息已经在上下文中被捕获，合并它
                    if request_info_captured:
                        request_info.update(request_info_captured)
                    raise  # 重新抛出异常，让外层统一处理
                
                if news_df is not None and not news_df.empty:
                    news_df = news_df.head(5)
                    for _, row in news_df.iterrows():
                        realtime_results.append({
                            'title': str(row.get('新闻标题', '')),
                            'content': str(row.get('新闻内容', '')),
                            'publish_time': str(row.get('发布时间', '')),
                            'source': 'AKShare-东方财富',
                            'url': str(row.get('新闻链接', '')),
                            'symbol': normalized_symbol or symbol_for_query,
                            'query': query,
                            'search_time': search_time
                        })
                    
                    realtime_results = self._filter_allowed_news_records(realtime_results)
                    print(f"✅ 成功获取 {len(realtime_results)} 条实时新闻（股票：{normalized_symbol or symbol_for_query}）")
                    
                    # 3. 保存实时新闻到 data/news.csv（追加模式，去重）——并发安全写入
                    try:
                        self._purge_news_csv(csv_path)
                        with NEWS_FILE_LOCK:
                            os.makedirs('data', exist_ok=True)
                            df_new = pd.DataFrame(realtime_results)
                            df_new = self._sanitize_news_dataframe(df_new)
                            df_new = self._filter_allowed_news_df(df_new)
                            if df_new is None or df_new.empty:
                                continue
                            if 'search_time' not in df_new.columns:
                                df_new['search_time'] = str(search_time)
                            else:
                                df_new['search_time'] = df_new['search_time'].astype(str)
                            dedupe_subset = ['symbol', 'title', 'search_time', 'query']
                            df_new = df_new.drop_duplicates(subset=dedupe_subset, keep='last')
                            
                            if os.path.exists(csv_path):
                                # 尝试多种编码读取现有文件
                                old = None
                                for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
                                    try:
                                        old = pd.read_csv(csv_path, encoding=encoding)
                                        print(f"✅ 使用 {encoding} 编码成功读取 news.csv")
                                        break
                                    except Exception:
                                        continue
                                
                                if old is not None:
                                    old = self._sanitize_news_dataframe(old)
                                    old = self._filter_allowed_news_df(old)
                                    if 'search_time' not in old.columns:
                                        old['search_time'] = pd.NA
                                    mask = ~(
                                        (old['symbol'].astype(str) == str(normalized_symbol or symbol_for_query)) &
                                        (old['query'].astype(str) == str(query)) &
                                        (old['search_time'].astype(str) == str(search_time))
                                    )
                                    old = old[mask]
                                    # 合并并去重
                                    combined = pd.concat([old, df_new], axis=0, ignore_index=True)
                                    combined = self._sanitize_news_dataframe(combined)
                                    combined = combined.drop_duplicates(subset=dedupe_subset, keep='last')
                                    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                    print(f"💾 已将新闻追加到 {csv_path}（去重后）")
                                else:
                                    # 如果所有编码都失败，直接覆盖
                                    df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                    print(f"⚠️ 无法读取旧文件，已创建新文件 {csv_path}")
                            else:
                                df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                print(f"💾 已创建新闻文件 {csv_path}")
                    except Exception as e:
                        print(f"⚠️ 保存新闻到CSV失败: {e}")
                    
                    # 返回组合结果：历史 + 实时
                    return json.dumps({
                        'success': True,
                        'source': 'akshare',
                        'historical_count': len(historical_news),
                        'realtime_count': len(realtime_results),
                        'total_count': len(historical_news) + len(realtime_results),
                        'historical_news': historical_news[:10],
                        'realtime_news': realtime_results,
                        'saved_to': csv_path
                    }, ensure_ascii=False)
                else:
                    print(f"⚠️ AKShare 返回空数据（尝试 {attempt}/{max_retries}，股票代码：{symbol_for_query}）")
                    print(f"💡 可能原因：1) 该股票暂无新闻 2) API返回空数据 3) 网络问题")
                    
            except json.JSONDecodeError as e:
                error_msg = str(e)
                print(f"⚠️ AKShare JSON解析失败（尝试 {attempt}/{max_retries}）: {error_msg}")
                print(f"💡 可能原因：API返回了非JSON内容（如HTML错误页面），通常是网络问题或反爬虫限制")
                if attempt < max_retries:
                    print(f"💡 建议：检查网络连接，或等待更长时间后重试")
                
                # 保存错误信息到文件（包含请求信息）
                self._save_akshare_error(
                    debug_dir, symbol_for_query, attempt, max_retries,
                    type(e).__name__, error_msg, traceback.format_exc(),
                    {'query': query, 'normalized_symbol': normalized_symbol, 'symbol_for_query': symbol_for_query},
                    request_info
                )
            except UnicodeDecodeError as e:
                error_msg = str(e)
                print(f"⚠️ AKShare 编码错误（尝试 {attempt}/{max_retries}）: {error_msg}")
                print(f"💡 可能原因：返回内容的编码格式不匹配")
                
                # 保存错误信息到文件（包含请求信息）
                self._save_akshare_error(
                    debug_dir, symbol_for_query, attempt, max_retries,
                    type(e).__name__, error_msg, traceback.format_exc(),
                    {'query': query, 'normalized_symbol': normalized_symbol, 'symbol_for_query': symbol_for_query},
                    request_info
                )
            except AttributeError as e:
                error_msg = str(e)
                if "'NoneType' object has no attribute" in error_msg:
                    print(f"⚠️ AKShare 返回None（尝试 {attempt}/{max_retries}）: {error_msg}")
                    print(f"💡 可能原因：API调用失败，返回了None，通常是网络问题或API限制")
                else:
                    print(f"⚠️ AKShare 属性错误（尝试 {attempt}/{max_retries}）: {error_msg}")
                
                # 保存错误信息到文件（包含请求信息）
                self._save_akshare_error(
                    debug_dir, symbol_for_query, attempt, max_retries,
                    type(e).__name__, error_msg, traceback.format_exc(),
                    {'query': query, 'normalized_symbol': normalized_symbol, 'symbol_for_query': symbol_for_query},
                    request_info
                )
            except ConnectionError as e:
                error_msg = str(e)
                print(f"⚠️ AKShare 连接错误（尝试 {attempt}/{max_retries}）: {error_msg}")
                print(f"💡 可能原因：网络连接失败，请检查网络或防火墙设置")
                
                # 保存错误信息到文件（包含请求信息）
                self._save_akshare_error(
                    debug_dir, symbol_for_query, attempt, max_retries,
                    type(e).__name__, error_msg, traceback.format_exc(),
                    {'query': query, 'normalized_symbol': normalized_symbol, 'symbol_for_query': symbol_for_query},
                    request_info
                )
            except TimeoutError as e:
                error_msg = str(e)
                print(f"⚠️ AKShare 请求超时（尝试 {attempt}/{max_retries}）: {error_msg}")
                print(f"💡 可能原因：API响应时间过长，可能是网络慢或服务器负载高")
                
                # 保存错误信息到文件（包含请求信息）
                self._save_akshare_error(
                    debug_dir, symbol_for_query, attempt, max_retries,
                    type(e).__name__, error_msg, traceback.format_exc(),
                    {'query': query, 'normalized_symbol': normalized_symbol, 'symbol_for_query': symbol_for_query},
                    request_info
                )
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"⚠️ AKShare 失败（尝试 {attempt}/{max_retries}）: [{error_type}] {error_msg}")
                
                # 特殊处理 JSON 解析错误（可能被包装在其他异常中）
                if "Expecting value" in error_msg or "JSON" in error_msg.upper():
                    print(f"💡 检测到JSON解析问题，可能是API返回了空响应或错误页面")
                    print(f"💡 建议：1) 检查网络连接 2) 验证股票代码 {symbol_for_query} 是否有效")
                    print(f"💡 可以尝试：手动访问东方财富网站查看该股票是否有新闻")
                
                # 保存错误信息到文件（包含请求信息）
                self._save_akshare_error(
                    debug_dir, symbol_for_query, attempt, max_retries,
                    error_type, error_msg, traceback.format_exc(),
                    {'query': query, 'normalized_symbol': normalized_symbol, 'symbol_for_query': symbol_for_query},
                    request_info
                )
                
                # 在最后一次尝试时打印完整堆栈
                if attempt == max_retries:
                    print(f"📋 完整错误信息:")
                    traceback.print_exc()
            
            # 如果不是最后一次尝试，等待后重试
            # 增加等待时间，避免频率限制
            if attempt < max_retries:
                wait_time = attempt * 5  # 递增延迟：5秒、10秒
                print(f"⏳ 等待 {wait_time} 秒后进行第 {attempt + 1} 次尝试...")
                import time
                time.sleep(wait_time)
            else:
                # 所有重试都失败
                print(f"❌ AKShare 所有 {max_retries} 次尝试均失败，将返回历史新闻")
        
        # 所有重试都失败，返回历史新闻（如果有）
        # 更新最新时间戳
        if realtime_results or historical_news:
            return json.dumps({
                'success': True,
                'source': 'akshare',
                'historical_count': len(historical_news),
                'realtime_count': len(realtime_results),
                'total_count': len(historical_news) + len(realtime_results),
                'historical_news': historical_news[:10],
                'realtime_news': realtime_results
            }, ensure_ascii=False)

        if historical_news:
            print(f"⚠️ AKShare 重试{max_retries}次均失败，返回历史新闻")
            return json.dumps({
                'success': True,
                'source': 'historical_only',
                'historical_count': len(historical_news),
                'realtime_count': 0,
                'total_count': len(historical_news),
                'historical_news': historical_news[:10],
                'realtime_news': [],
                'message': self._provider_downtime_message("AKShare")
            }, ensure_ascii=False)
        else:
            return json.dumps({
                'success': False, 
                'message': self._provider_downtime_message("AKShare")
            }, ensure_ascii=False)
    
    def sell_stock(self, symbol: str, amount: int) -> str:
        """
        卖出股票（使用当前小时级价格）。
        
        此函数模拟股票卖出操作，包括：
        1. 获取当前持仓和操作ID
        2. 获取当前小时的股票价格（优先小时级，回退到日线开盘价）
        3. 验证卖出条件（是否持有该股票，数量是否充足）
        4. 更新持仓（减少股票数量，增加现金）
        5. 记录交易到 position.jsonl 文件
        
        Args:
            symbol (str): 股票代码，如 "600519"
            amount (int): 卖出数量，必须是正整数
        
        Returns:
            str: JSON 字符串，成功时返回新持仓，失败时返回错误信息
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
            if not normalized_symbol:
                return json.dumps({"error": "无效的股票代码"})
            data_symbol = strip_exchange_prefix(normalized_symbol) or normalized_symbol
            
            # 使用实例上下文，避免从共享 runtime_env.json 读取
            today_date = self._get_context_value("TODAY_DATE")
            current_time = self._get_context_value("CURRENT_TIME")
            decision_time = current_time or f"{today_date} 00:00:00"
            decision_count_raw = self._get_context_value("DECISION_COUNT")
            decision_count = int(decision_count_raw) if decision_count_raw is not None else 0
            if not today_date:
                return json.dumps({"error": "未设置 TODAY_DATE"})
            decision_time = normalize_decision_time(today_date, decision_time)
            
            # 获取当前持仓和操作ID
            current_position, current_action_id, latest_record = get_latest_position(today_date, self.signature)
            
            # --- T+1 规则检查 ---
            if normalized_symbol in current_position and isinstance(current_position[normalized_symbol], dict):
                purchase_date = current_position[normalized_symbol].get("purchase_date")
                if purchase_date == today_date:
                    return json.dumps({
                        "error": "T+1规则限制：今日买入的股票不能在当日卖出",
                        "symbol": normalized_symbol,
                        "purchase_date": purchase_date
                    })

            # 获取当前时刻的股票价格 - 优先小时级数据
            try:
                this_symbol_price = None
                # 优先尝试获取小时级价格
                if current_time:
                    hourly_data = self.dm.get_hourly_stock_data(symbol=data_symbol, end_date=current_time, lookback_hours=1)
                    if not hourly_data.empty:
                        this_symbol_price = float(hourly_data['close'].iloc[-1])
                        print(f"💹 使用小时级价格: {normalized_symbol} = ¥{this_symbol_price} ({current_time})")
                
                # 如果没有小时级数据，回退到日线开盘价
                if this_symbol_price is None or pd.isna(this_symbol_price):
                    stock_data = self.dm.get_stock_data(symbol=data_symbol, end_date=today_date, lookback_days=1)
                    if stock_data.empty:
                        return json.dumps({"error": f"未找到股票 {normalized_symbol} 的价格数据", "symbol": normalized_symbol, "date": today_date})
                    this_symbol_price = float(stock_data['open'].iloc[-1]) if 'open' in stock_data.columns else float(stock_data['close'].iloc[-1])
                    print(f"💹 使用开盘价: {normalized_symbol} = ¥{this_symbol_price}")
                
                if pd.isna(this_symbol_price) or this_symbol_price <= 0:
                    return json.dumps({"error": f"股票 {normalized_symbol} 的价格数据无效", "symbol": normalized_symbol, "date": today_date, "price": this_symbol_price})
            except Exception as e:
                return json.dumps({"error": f"获取股票价格失败: {str(e)}", "symbol": normalized_symbol, "date": today_date})
            
            limit_info: Optional[Dict[str, float]] = None
            prev_close = self._get_previous_close(normalized_symbol, today_date)
            limit_info = get_price_limits(normalized_symbol, prev_close)
            allowed, reason = self._passes_price_limit_liquidity("sell", this_symbol_price, limit_info)
            if not allowed:
                return json.dumps({
                    "error": reason,
                    "symbol": normalized_symbol,
                    "price": this_symbol_price,
                    "limit_info": limit_info
                }, ensure_ascii=False)
            
            # 验证卖出条件
            if normalized_symbol not in current_position or not isinstance(current_position[normalized_symbol], dict):
                return json.dumps({"error": f"未持有股票 {normalized_symbol}！交易不被允许。", "symbol": normalized_symbol, "date": today_date})
            
            current_shares = current_position.get(normalized_symbol, {}).get("shares", 0)
            if current_shares < amount:
                return json.dumps({
                    "error": "持股数量不足！交易不被允许。",
                    "have": current_shares,
                    "want_to_sell": amount,
                    "symbol": normalized_symbol,
                    "date": today_date
                })
            
            # 执行卖出操作
            new_position = copy.deepcopy(current_position)
            new_position[normalized_symbol]["shares"] = current_shares - amount
            
            # 如果股票数量为0，则从持仓中移除
            if new_position[normalized_symbol]["shares"] == 0:
                del new_position[normalized_symbol]

            revenue = this_symbol_price * amount
            
            # --- 交易成本计算 ---
            commission_rate = self.trading_rules.get("commission_rate", 0.0003)
            min_commission = self.trading_rules.get("min_commission", 5.0)
            stamp_duty_rate = self.trading_rules.get("stamp_duty_rate", 0.0005)
            
            commission = max(revenue * commission_rate, min_commission)
            stamp_duty = revenue * stamp_duty_rate
            total_deduction = commission + stamp_duty
            
            net_revenue = revenue - total_deduction
            new_position["CASH"] = new_position.get("CASH", 0) + net_revenue
            new_position = normalize_positions(new_position)
            
            # 记录交易
            record: Dict[str, Any] = {
                "date": today_date,
                "decision_time": decision_time,
                "decision_count": decision_count,
                "this_action": {"action": "sell", "symbol": normalized_symbol, "amount": amount},
                "positions": new_position
            }
            if latest_record and latest_record.get("decision_time") == decision_time:
                record["id"] = latest_record.get("id")
            else:
                record["id"] = current_action_id + 1
            upsert_position_record(self.signature, record)
            
            write_config_value("IF_TRADE", True)
            return json.dumps({
                "success": True,
                "action": "sell",
                "symbol": normalized_symbol,
                "amount": amount,
                "price": this_symbol_price,
                "revenue": revenue,
                "commission": commission,
                "stamp_duty": stamp_duty,
                "net_revenue": net_revenue,
                "price_limit": limit_info,
                "decision_time": decision_time,
                "decision_count": decision_count,
                "new_position": new_position
            })
        
        except Exception as e:
            return json.dumps({"error": f"卖出股票时出错: {str(e)}"})
    
    async def initialize(self) -> None:
        """初始化 AI 模型和工具（纯本地模式）"""
        print(f"🚀 初始化代理: {self.signature}")
        print("💻 使用纯本地模式...")
        if self.dm:
            self.tools = [
                tool(self.search_stock_news),            # 🔄 已整合：历史+实时新闻，自动保存
                tool(self.get_technical_indicators),     # 🔄 已整合：历史+实时指标，自动保存
                tool(self.get_hourly_stock_data),
                tool(self.get_current_stock_prices),
                tool(self.get_latest_position_tool),
                tool(self.buy_stock),
                tool(self.sell_stock)
            ]
            print(f"✅ 已加载 {len(self.tools)} 个本地工具（所有工具已整合历史+实时数据）")
        else:
            print(f"❌ DataManager 未初始化,未加载任何工具")
            self.tools = []
        
        if self.basemodel.startswith("gemini"):
            print(f"🤖 Initializing Google Gemini model: {self.basemodel}")
            gemini_safety_settings = None
            if self.safety_settings:
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                gemini_safety_settings = {}
                harm_category_map = {
                    "HARM_CATEGORY_HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "HARM_CATEGORY_HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "HARM_CATEGORY_DANGEROUS_CONTENT": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                }
                harm_threshold_map = {
                    "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
                    "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
                for category, threshold in self.safety_settings.items():
                    if category in harm_category_map and threshold in harm_threshold_map:
                        gemini_safety_settings[harm_category_map[category]] = harm_threshold_map[threshold]
            self.model = ChatGoogleGenerativeAI(
                model=self.basemodel,
                google_api_key=self.google_api_key,
                safety_settings=gemini_safety_settings,
                max_retries=5,
                timeout=60
            )
            print(f"✅ Google Gemini model initialized with API key from {'environment' if self.google_api_key == os.getenv('GEMINI_API_KEY') else 'config'}")
        elif self.basemodel.startswith("qwen"):
            print(f"🤖 Initializing Qwen model: {self.basemodel}")
            dashscope_url = self.openai_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model = ChatOpenAI(
                model=self.basemodel,
                base_url=dashscope_url,
                api_key=self.openai_api_key,
                max_retries=5,
                timeout=180
            )
            print(f"✅ Qwen model initialized via {dashscope_url}")
        else:
            print(f"🤖 Initializing OpenAI-compatible model: {self.basemodel}")
            self.model = ChatOpenAI(
                model=self.basemodel,
                base_url=self.openai_base_url,
                api_key=self.openai_api_key,
                max_retries=3,
                timeout=30
            )
            print(f"✅ OpenAI-compatible model initialized")
        
        print(f"✅ Agent {self.signature} initialization completed")
    def _setup_logging(self, today_date: str, decision_time: str) -> str:
        """Set up (and reset) log file path for a specific decision time"""
        log_path = os.path.join(self.base_log_path, self.signature, 'log', today_date)
        os.makedirs(log_path, exist_ok=True)
        sanitized_time = decision_time.replace(":", "-").replace(" ", "_")
        log_file = os.path.join(log_path, f"{sanitized_time}.jsonl")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("")
        return log_file
    
    def _log_message(
        self,
        log_file: str,
        new_messages: List[Dict[str, str]],
        decision_time: Optional[str] = None,
        decision_count: Optional[int] = None
    ) -> None:
        """Log messages to log file"""
        sanitized_messages = self._sanitize_messages(new_messages)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "signature": self.signature,
            "decision_time": decision_time,
            "decision_count": decision_count,
            "new_messages": sanitized_messages
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _log_snapshot_reference(
        self,
        log_file: str,
        decision_time: str,
        decision_count: int,
    ) -> None:
        if not self._current_snapshot_info:
            return
        info = self._current_snapshot_info
        snapshot_id = info.get("snapshot_id") or "unknown"
        snapshot_path = info.get("snapshot_path") or "local-memory"
        mode = info.get("mode") or "local"
        created_flag = info.get("snapshot_created")
        content = (
            f"Shared snapshot [{mode}] id={snapshot_id}, path={snapshot_path}, "
            f"created_now={created_flag}"
        )
        self._log_message(
            log_file,
            [{"role": "system", "content": content}],
            decision_time=decision_time,
            decision_count=decision_count,
        )
    
    def _sanitize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create a lightweight copy of messages for logging (short summary / truncation)."""
        sanitized: List[Dict[str, str]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                sanitized_content = self._summarize_content(content)
                sanitized.append({**msg, "content": sanitized_content})
            else:
                sanitized.append(msg.copy())
        return sanitized
    
    def _summarize_content(self, content: str) -> str:
        """Summarize long or JSON-heavy content for logging."""
        if not content:
            return ""
        MAX_LEN = 500
        stripped = content.strip()
        if len(stripped) <= MAX_LEN and stripped.count("\n") <= 4:
            return stripped
        
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) == 1:
            return self._summarize_line(lines[0])
        
        summaries = []
        for line in lines[:3]:
            summaries.append(self._summarize_line(line))
        if len(lines) > 3:
            summaries.append(f"... ({len(lines)} entries)")
        return " | ".join(summaries)
    
    def _summarize_line(self, line: str) -> str:
        """Summarize single line (try JSON first, fallback to truncated text)."""
        try:
            data = json.loads(line)
        except Exception:
            return self._truncate_text(line)
        
        if isinstance(data, dict):
            if "error" in data:
                return f"error: {self._truncate_text(str(data.get('error')))}"
            if data.get("success") is False:
                return f"failed: {self._truncate_text(str(data.get('message') or data))}"
            parts: List[str] = []
            if "success" in data:
                parts.append(f"success={data['success']}")
            for key in ("historical_count", "realtime_count", "total_count"):
                if key in data:
                    parts.append(f"{key}={data[key]}")
            if "message" in data:
                parts.append(self._truncate_text(str(data["message"])))
            if not parts:
                parts.append(self._truncate_text(json.dumps(data, ensure_ascii=False)))
            return ", ".join(parts)
        if isinstance(data, list):
            return f"list[{len(data)}]"
        return self._truncate_text(str(data))
    
    def _truncate_text(self, text: str, max_len: int = 180) -> str:
        text = text.strip()
        return text if len(text) <= max_len else text[:max_len] + "...(truncated)"
    
    def _content_to_text(self, content: Any) -> str:
        """Normalize message content (which can be str/list/dict) into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item and item["text"]:
                        parts.append(str(item["text"]))
                    elif "json" in item:
                        try:
                            parts.append(json.dumps(item["json"], ensure_ascii=False))
                        except Exception:
                            parts.append(str(item["json"]))
                    elif "data" in item:
                        parts.append(str(item["data"]))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            if "text" in content and content["text"]:
                return str(content["text"])
            if "json" in content:
                try:
                    return json.dumps(content["json"], ensure_ascii=False)
                except Exception:
                    return str(content["json"])
        return str(content)
    
    def _combine_tool_outputs(self, tool_messages: List[Any]) -> str:
        """Join tool outputs into a single string while handling different payload shapes."""
        outputs: List[str] = []
        for msg in tool_messages:
            if isinstance(msg, dict):
                content = msg.get("content")
            else:
                content = getattr(msg, "content", None)
            text = self._content_to_text(content)
            if text:
                outputs.append(text)
        return "\n".join(outputs)
    
    def _capture_akshare_response(self, symbol: str):
        """尝试拦截 AKShare API 的 HTTP 请求和响应。
        
        使用 requests 的 monkey patching 来捕获响应内容，包括原始字节和多种解码方式。
        
        Args:
            symbol: 股票代码
            
        Returns:
            tuple: (capture_context, request_info_dict) - 上下文管理器和请求信息字典（可变对象）
        """
        request_info = {
            'symbol': symbol,
            'url': None,
            'method': None,
            'status_code': None,
            'response_text': None,
            'response_content_raw': None,  # 原始字节内容
            'response_content_length': None,  # 原始内容长度
            'response_content_decoded': None,  # 手动解码后的内容
            'response_headers': None,
            'request_headers': None
        }
        
        try:
            import requests
            import gzip
            
            # 存储原始方法
            original_send = requests.Session.send
            
            # 创建上下文管理器
            class ResponseCapture:
                def __init__(self, req_info):
                    self.request_info = req_info
                    self.original_send = original_send
                
                def __enter__(self):
                    # Monkey patch send 方法
                    def patched_send(self_session, request, **kwargs):
                        # 记录请求信息
                        req_info = self.request_info
                        req_info['url'] = request.url
                        req_info['method'] = request.method
                        
                        # 修改请求头，使用真实的浏览器 User-Agent 和其他 headers
                        # 避免被识别为爬虫
                        if request.headers is None:
                            request.headers = {}
                        
                        # 使用真实的 Chrome 浏览器 User-Agent
                        browser_user_agent = (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                        
                        # 修改或添加 headers
                        request.headers['User-Agent'] = browser_user_agent
                        request.headers['Accept'] = 'application/json, text/plain, */*'
                        request.headers['Accept-Language'] = 'zh-CN,zh;q=0.9,en;q=0.8'
                        request.headers['Accept-Encoding'] = 'gzip, deflate, br'
                        request.headers['Referer'] = 'http://quote.eastmoney.com/'
                        request.headers['Origin'] = 'http://quote.eastmoney.com'
                        request.headers['Connection'] = 'keep-alive'
                        
                        # 记录修改后的请求头
                        if request.headers:
                            req_info['request_headers'] = dict(request.headers)
                        
                        # 发送请求
                        response = self.original_send(self_session, request, **kwargs)
                        
                        # 记录响应信息
                        try:
                            req_info['status_code'] = response.status_code
                            if response.headers:
                                req_info['response_headers'] = dict(response.headers)
                            
                            # 保存原始字节内容（在 requests 自动解压之前）
                            try:
                                # 获取原始内容（如果还没有被解码）
                                if hasattr(response, 'content'):
                                    raw_content = response.content
                                    req_info['response_content_length'] = len(raw_content) if raw_content else 0
                                    
                                    # 保存原始字节的前5000字节（用于调试）
                                    if raw_content:
                                        try:
                                            # 尝试转换为可打印的字符串（只保存前1000字节以避免文件过大）
                                            content_preview = raw_content[:1000]
                                            # 如果内容是二进制，尝试编码为十六进制字符串
                                            if isinstance(content_preview, bytes):
                                                req_info['response_content_raw'] = content_preview.hex()[:2000]  # 限制长度
                                            else:
                                                req_info['response_content_raw'] = str(content_preview)[:2000]
                                        except Exception:
                                            req_info['response_content_raw'] = f"<无法显示: {len(raw_content)} 字节>"
                                    
                                    # 尝试手动解码 gzip 内容（如果响应是 gzip 压缩的）
                                    content_encoding = response.headers.get('Content-Encoding', '').lower()
                                    if 'gzip' in content_encoding and raw_content:
                                        try:
                                            decoded_content = gzip.decompress(raw_content)
                                            decoded_text = decoded_content.decode('utf-8', errors='ignore')
                                            # 只保存前2000个字符
                                            req_info['response_content_decoded'] = decoded_text[:2000]
                                        except Exception as gzip_err:
                                            req_info['gzip_decode_error'] = str(gzip_err)
                                else:
                                    req_info['response_content_length'] = 0
                            except Exception as content_err:
                                req_info['content_capture_error'] = str(content_err)
                            
                            # 保存解码后的文本内容（requests 自动处理的）
                            try:
                                text_content = response.text
                                req_info['response_text'] = text_content[:2000] if text_content else ""
                                req_info['response_text_length'] = len(text_content) if text_content else 0
                            except Exception as text_err:
                                req_info['response_text'] = ""
                                req_info['response_text_error'] = str(text_err)
                            
                            req_info['url'] = response.url or req_info.get('url')
                        except Exception as e:
                            req_info['capture_error'] = str(e)
                        
                        return response
                    
                    # 替换 send 方法
                    requests.Session.send = patched_send
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    # 恢复原始方法
                    requests.Session.send = self.original_send
                    return False  # 不抑制异常
            
            return ResponseCapture(request_info), request_info
            
        except Exception as e:
            # 如果拦截失败，返回空的上下文管理器和错误信息
            request_info['capture_error'] = str(e)
            
            class EmptyCapture:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
            
            return EmptyCapture(), request_info

    def _save_akshare_error(self, debug_dir: str, symbol: str, attempt: int, max_retries: int,
                            error_type: str, error_msg: str, traceback_str: str, context: dict,
                            request_info: Optional[dict] = None):
        """保存 AKShare API 错误信息到文件，便于后续分析。
        
        Args:
            debug_dir: 调试文件夹路径
            symbol: 股票代码
            attempt: 当前尝试次数
            max_retries: 最大重试次数
            error_type: 错误类型
            error_msg: 错误消息
            traceback_str: 堆栈跟踪
            context: 上下文信息（查询参数等）
            request_info: 请求信息（URL、参数、响应内容等）
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"akshare_error_{symbol}_{timestamp}_attempt{attempt}.json"
            filepath = os.path.join(debug_dir, filename)
            
            # 获取系统环境信息
            import sys
            import platform
            try:
                import akshare as ak
                akshare_version = getattr(ak, '__version__', 'unknown')
            except Exception:
                akshare_version = 'unknown'
            
            error_data = {
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'symbol': symbol,
                'attempt': attempt,
                'max_retries': max_retries,
                'error_type': error_type,
                'error_message': error_msg,
                'context': context,
                'traceback': traceback_str,
                'environment': {
                    'python_version': sys.version,
                    'platform': platform.platform(),
                    'akshare_version': akshare_version
                },
                'request_info': request_info or {}
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 错误信息已保存到: {filepath}")
            print(f"📋 请将文件发送给开发者进行分析: {filename}")
        except Exception as save_err:
            print(f"⚠️ 保存错误信息失败: {save_err}")
    
    def _extract_tool_errors(self, tool_response: str) -> List[str]:
        """从工具返回中提取真正的错误信息，用于最终总结说明。
        注意：警告、空数据或部分成功不应被视为错误。
        """
        errors: List[str] = []
        if not tool_response:
            return errors
        
        # 先尝试解析整个响应作为 JSON
        try:
            data = json.loads(tool_response)
            if isinstance(data, dict):
                # 只提取明确的错误，不包括警告或部分成功
                if data.get("error"):
                    error_msg = str(data["error"])
                    # 排除常见的警告信息（如"返回历史数据"、"客户端不可用"等）
                    # 这些是降级处理的正常情况，不应该视为错误
                    if "返回历史数据" not in error_msg and "不可用" not in error_msg:
                        # 只有在真正失败时才记录为错误
                        if "失败" in error_msg or "错误" in error_msg or "error" in error_msg.lower():
                            errors.append(error_msg)
                elif data.get("success") is False:
                    message = data.get("message", "")
                    if message and ("失败" in message or "错误" in message or "error" in message.lower()):
                        errors.append(str(message))
        except Exception:
            # 如果不是 JSON，尝试逐行解析
            for raw_line in tool_response.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        if data.get("error"):
                            error_msg = str(data["error"])
                            # 排除警告信息
                            if "返回历史数据" not in error_msg and "警告" not in error_msg:
                                errors.append(error_msg)
                        elif data.get("success") is False:
                            message = data.get("message", "")
                            if message and ("失败" in message or "错误" in message):
                                errors.append(str(message))
                except Exception:
                    continue
        
        return errors

    def _provider_downtime_message(self, provider_name: str) -> str:
        """Return a friendly message when upstream providers fail."""
        return f"{provider_name} 数据源暂时不可用，系统已切换到备份数据，请稍后重试。"

    def _fix_mojibake(self, value: Any) -> Any:
        if isinstance(value, str):
            clean_value = value.replace("\ufeff", "").strip()
            for _ in range(3):
                if not any(ch in clean_value for ch in ("Ã", "Â", "ï")) and not any(128 <= ord(ch) <= 255 for ch in clean_value):
                    break
                try:
                    decoded_value = clean_value.encode("latin1").decode("utf-8")
                except Exception:
                    break
                if decoded_value == clean_value:
                    break
                clean_value = decoded_value
            return clean_value
        return value

    def _sanitize_dataframe_text(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        sanitized = df.copy()
        raw_columns = [self._fix_mojibake(str(col)).strip() for col in sanitized.columns]
        unique_columns: List[str] = []
        column_indices: List[int] = []
        seen: set[str] = set()
        for idx, col_name in enumerate(raw_columns):
            base_name = col_name.split(".", 1)[0]
            if base_name not in seen:
                unique_columns.append(base_name)
                column_indices.append(idx)
                seen.add(base_name)
        sanitized = sanitized.iloc[:, column_indices]
        sanitized.columns = unique_columns
        for col in sanitized.columns:
            if sanitized[col].dtype == object:
                sanitized[col] = sanitized[col].apply(self._fix_mojibake)
        return sanitized

    def _normalize_symbol_value(self, value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        raw = str(value).strip()
        if raw.endswith(".0"):
            raw = raw[:-2]
        if raw.isdigit() and len(raw) < 6:
            raw = raw.zfill(6)
        normalized = normalize_symbol(raw)
        return normalized or raw.upper()

    def _sanitize_news_dataframe(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        sanitized = self._sanitize_dataframe_text(df)
        if sanitized is None:
            return None
        if "symbol" in sanitized.columns:
            sanitized["symbol"] = sanitized["symbol"].apply(self._normalize_symbol_value)
            sanitized = sanitized[sanitized["symbol"].astype(str).str.len() > 0]
        return sanitized
    
    def _get_previous_close(self, symbol: str, today_date: str) -> Optional[float]:
        if not self.dm:
            return None
        try:
            prev_date = get_yesterday_date(today_date)
            target_time = f"{prev_date} 15:00:00"
            plain_symbol = strip_exchange_prefix(symbol) or symbol
            price = self.dm.get_price_at(plain_symbol, target_time)
            if price is None and plain_symbol != symbol:
                price = self.dm.get_price_at(symbol, target_time)
            return float(price) if price is not None else None
        except Exception as e:
            print(f"⚠️ 获取前收失败: {e}")
            return None
    
    def _passes_price_limit_liquidity(
        self,
        action: str,
        price: Optional[float],
        limits: Optional[Dict[str, float]]
    ) -> Tuple[bool, Optional[str]]:
        if price is None or not limits:
            return True, None
        threshold = self.LIMIT_THRESHOLD_RATIO
        if action == "buy":
            upper = limits.get("upper")
            if upper is not None and price >= upper * threshold:
                if random.random() >= self.LIMIT_ORDER_SUCCESS_RATE:
                    return False, f"接近涨停价 ¥{upper:.2f}，买单成交概率仅10%，此次模拟未成交。"
        elif action == "sell":
            lower = limits.get("lower")
            if lower is not None and price <= lower / threshold:
                if random.random() >= self.LIMIT_ORDER_SUCCESS_RATE:
                    return False, f"接近跌停价 ¥{lower:.2f}，卖单成交概率仅10%，此次模拟未成交。"
        return True, None
    
    def _compute_portfolio_metrics(
        self,
        positions: Dict[str, Any],
        today_date: str,
        current_time: Optional[str]
    ) -> Dict[str, Any]:
        metrics = {
            "cash": float(positions.get("CASH", 0.0) or 0.0),
            "position_value": 0.0,
            "unrealized_total": 0.0,
            "total_equity": 0.0,
            "holdings": [],
        }
        holdings_symbols: List[str] = []
        processed: List[Dict[str, Any]] = []
        is_open_slot = bool(current_time and current_time.endswith("10:30:00"))
        opening_price_cache: Dict[str, Optional[float]] = {}
        opening_reference_time = f"{today_date} 10:30:00" if is_open_slot else None
        
        if not positions:
            metrics["total_equity"] = metrics["cash"]
            return metrics
        
        for symbol, data in positions.items():
            if symbol == "CASH":
                continue
            if not isinstance(data, dict):
                continue
            shares = int(data.get("shares", 0) or 0)
            if shares <= 0:
                continue
            holdings_symbols.append(symbol)
            processed.append({
                "symbol": symbol,
                "shares": shares,
                "purchase_date": data.get("purchase_date"),
                "avg_price": data.get("avg_price")
            })
        
        price_lookup: Dict[str, Optional[float]] = {}
        target_time = current_time or f"{today_date} 15:00:00"
        if self.dm and holdings_symbols:
            try:
                price_lookup = self.dm.get_prices_at(holdings_symbols, target_time) or {}
            except Exception as e:
                print(f"⚠️ Failed to fetch current prices: {e}")
                price_lookup = {}
        
        for item in processed:
            symbol = item["symbol"]
            shares = item["shares"]
            purchase_date = item.get("purchase_date")
            avg_price = item.get("avg_price")
            current_price = None
            if price_lookup:
                current_price = price_lookup.get(symbol.upper()) or price_lookup.get(symbol)
            if current_price is None and self.dm:
                try:
                    current_price = self.dm.get_price_at(symbol, target_time)
                except Exception:
                    current_price = None

            if current_price is None and is_open_slot and self.dm and opening_reference_time:
                plain_symbol = strip_exchange_prefix(symbol) or symbol
                if plain_symbol not in opening_price_cache:
                    try:
                        df_open = self.dm.get_hourly_stock_data(
                            symbol=plain_symbol,
                            end_date=opening_reference_time,
                            lookback_hours=1
                        )
                        if df_open is not None and not df_open.empty:
                            latest_row = df_open.iloc[-1]
                            price_candidate = latest_row.get("open")
                            if price_candidate is None or pd.isna(price_candidate):
                                price_candidate = latest_row.get("close")
                            opening_price_cache[plain_symbol] = float(price_candidate) if price_candidate is not None else None
                        else:
                            opening_price_cache[plain_symbol] = None
                    except Exception as e:
                        print(f"⚠️ Failed to fetch opening price for {plain_symbol}: {e}")
                        opening_price_cache[plain_symbol] = None
                current_price = opening_price_cache.get(plain_symbol)
            
            if avg_price is None and self.dm and purchase_date:
                try:
                    approx_time = f"{purchase_date} 15:00:00"
                    avg_price = self.dm.get_price_at(symbol, approx_time)
                except Exception:
                    avg_price = None
            
            market_value = float(current_price or 0.0) * shares if current_price is not None else 0.0
            cost_basis = None
            if avg_price is not None:
                cost_basis = float(avg_price) * shares
            unrealized = None
            if cost_basis is not None and current_price is not None:
                unrealized = market_value - cost_basis
                metrics["unrealized_total"] += unrealized
            
            metrics["position_value"] += market_value
            metrics["holdings"].append({
                "symbol": symbol,
                "shares": shares,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized": unrealized
            })
        
        metrics["total_equity"] = metrics["cash"] + metrics["position_value"]
        return metrics

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        try:
            parsed = pd.to_datetime(value)
            if hasattr(parsed, "to_pydatetime"):
                parsed = parsed.to_pydatetime()
            if isinstance(parsed, datetime):
                return parsed.replace(tzinfo=None)
        except Exception:
            return None
        return None
    
    async def _ainvoke_with_retry(
        self,
        message: List[Dict[str, str]],
        recursion_limit: Optional[int] = None
    ) -> Any:
        """Agent invocation with retry"""
        limit = recursion_limit or self.max_steps or 20
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.agent.ainvoke(
                    {"messages": message}, 
                    {"recursion_limit": limit}
                )
            except Exception as e:
                # Handle Google API specific errors
                error_type = type(e).__name__
                if "google" in str(type(e).__module__).lower():
                    print(f"⚠️ Google API error ({error_type}): {e}")
                else:
                    print(f"⚠️ Error ({error_type}): {e}")
                if attempt == self.max_retries:
                    print(f"❌ All {self.max_retries} attempts failed")
                    raise e
                
                wait_time = self.base_delay * attempt
                print(f"⚠️ Attempt {attempt} failed, retrying after {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    async def run_trading_session(
        self,
        today_date: str,
        current_time: str,
        decision_count: int = 1
    ) -> None:
        """
        Run single trading session
        
        Args:
            today_date: Trading date
            current_time: Current simulation time
            decision_count: Which decision this is (1-3)
        """
        print(f"📈 Starting trading session: {current_time} (Decision {decision_count}/3)")
        
        # Ensure config values are set (防止外部流程遗漏导致工具报错)
        write_config_value("TODAY_DATE", today_date)
        write_config_value("CURRENT_TIME", current_time)
        write_config_value("DECISION_COUNT", decision_count)
        self.runtime_context["TODAY_DATE"] = today_date
        self.runtime_context["CURRENT_TIME"] = current_time
        self.runtime_context["DECISION_COUNT"] = decision_count

        # Set up logging
        log_file = self._setup_logging(today_date, current_time)
        
        # Update system prompt with decision count
        self.agent = create_react_agent(
            self.model,
            tools=self.tools,
            prompt=get_agent_system_prompt(
                today_date,
                self.signature,
                dm=self.dm,
                current_time=current_time,
                decision_count=decision_count
            ),
        )
        
        snapshot_result = None
        snapshot_bundle: Optional[Dict[str, Any]] = None
        observation_summary = ""
        try:
            if not self.prefetch_coordinator:
                raise RuntimeError("Shared prefetch coordinator is not available")

            def _build_snapshot() -> Dict[str, Any]:
                return self._collect_prefetch_bundle(today_date, current_time, decision_count)

            snapshot_result = self.prefetch_coordinator.ensure_snapshot(
                today_date=today_date,
                current_time=current_time,
                symbols_signature=self._symbols_signature(),
                builder=_build_snapshot,
            )
            snapshot_bundle = snapshot_result.data
            observation_summary = self._apply_prefetch_bundle(snapshot_bundle)
        except Exception as e:
            print(f"⚠️ Shared prefetch unavailable ({e}), falling back to per-agent prefetch.")
            snapshot_bundle = self._collect_prefetch_bundle(today_date, current_time, decision_count)
            observation_summary = snapshot_bundle.get("observation_summary") or self._build_observation_summary()
            snapshot_result = None

        snapshot_id = (snapshot_bundle or {}).get("snapshot_id")
        snapshot_path = snapshot_result.path if snapshot_result else None
        snapshot_created = snapshot_result.created if snapshot_result else False
        self._current_snapshot_info = {
            "snapshot_id": snapshot_id,
            "snapshot_path": snapshot_path,
            "snapshot_created": snapshot_created,
            "mode": "shared" if snapshot_result else "local",
        }
        self._log_snapshot_reference(log_file, current_time, decision_count)

        # Build dynamic context message with full positions snapshot
        try:
            latest_positions, _, latest_record = get_latest_position(today_date, self.signature)
        except Exception as e:
            print(f"⚠️ Failed to load latest positions: {e}")
            latest_positions = {}
            latest_record = None

        positions_json = json.dumps(latest_positions, ensure_ascii=False, indent=2)
        last_action = None
        if latest_record:
            last_action = latest_record.get("this_action")

        if decision_count == 1:
            stage = "opening (observe & prepare)"
        elif decision_count == 2:
            stage = "midday (deploy capital)"
        else:
            stage = "afternoon (adjust/lock profits)"

        metrics = self._compute_portfolio_metrics(latest_positions, today_date, current_time)
        if metrics["cash"] <= 0 and not metrics["holdings"]:
            print("⚠️ Cash balance is zero with no holdings. Consider enabling FORCE_REPLAY to reset positions.")
        holdings_lines: List[str] = []
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
        
        context_message = (
            f"Trading context for {self.signature}:\n"
            f"- Date: {today_date}\n"
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
            "- 无论是否交易，必须在结尾使用 <FINISH_SIGNAL> 并给出清晰结论；若选择不交易，请说明具体原因与下一步计划。\n"
            "Please analyze current signals, adjust positions if needed, obey all rules, "
            "and provide a concise plan before calling tools or trades. Conclude with the required stop signal."
        )

        user_query = [{"role": "user", "content": context_message}]
        message = user_query.copy()
        
        # Log initial message
        self._log_message(log_file, user_query, decision_time=current_time, decision_count=decision_count)
        
        final_agent_summary: Optional[str] = None
        collected_tool_errors: List[str] = []
        
        try:
            response = await self._ainvoke_with_retry(message, recursion_limit=self.max_steps)
            
            # Extract agent response
            agent_response = extract_conversation(response, "final")
            if agent_response and agent_response.strip():
                final_agent_summary = agent_response
                if STOP_SIGNAL in agent_response:
                    print("✅ Received stop signal, trading session ended")
                else:
                    print("ℹ️ Agent completed without explicit stop signal")
                    collected_tool_errors.append("Missing stop signal in agent response")
                print(agent_response)
                self._log_message(
                    log_file,
                    [{"role": "assistant", "content": agent_response}],
                    decision_time=current_time,
                    decision_count=decision_count,
                )
            else:
                print("⚠️ Agent produced no final response")
                final_agent_summary = "NO_TRADE: 模型未提供有效输出。"
                collected_tool_errors.append("Agent produced no final response")
                self._log_message(
                    log_file,
                    [{"role": "system", "content": "Agent produced no final response."}],
                    decision_time=current_time,
                    decision_count=decision_count,
                )

            # Extract and summarize tool outputs for logging/error tracking
            tool_msgs = extract_tool_messages(response)
            tool_response = self._combine_tool_outputs(tool_msgs)
            if tool_response:
                collected_tool_errors.extend(self._extract_tool_errors(tool_response))
                tool_summary = self._summarize_content(tool_response) or self._truncate_text(tool_response, 600)
            self._log_message(
                log_file,
                [{"role": "system", "content": f"Tool summary: {tool_summary}"}],
                decision_time=current_time,
                decision_count=decision_count,
            )
        except Exception as e:
            print(f"❌ Trading session error: {str(e)}")
            print(f"Error details: {e}")
            import traceback

            traceback.print_exc()
            # 即使出错也要记录错误信息到日志
            try:
                error_msg = f"Trading session failed: {str(e)}"
                self._log_message(
                    log_file,
                    [{"role": "assistant", "content": error_msg}],
                    decision_time=current_time,
                    decision_count=decision_count,
                )
            except Exception:
                pass  # 如果日志记录也失败，至少不影响主流程
            raise
        finally:
            # 无论成功与否都尝试处理交易结果并记录状态
            try:
                await self._handle_trading_result(
                    today_date,
                    current_time,
                    decision_count,
                    log_file,
                    final_agent_summary,
                    collected_tool_errors,
                )
            except Exception as e:
                print(f"⚠️ Error handling trading result: {e}")
                try:
                    error_msg = f"Error handling trading result: {str(e)}"
                    self._log_message(
                        log_file,
                        [{"role": "system", "content": error_msg}],
                        decision_time=current_time,
                        decision_count=decision_count,
                    )
                except Exception:
                    pass
            try:
                from tools.backup_utils import run_backup_snapshot, save_pnl_snapshot
                reason = f"decision_{decision_count}_{today_date}_{current_time}"
                ok = run_backup_snapshot(reason=reason)
                if ok:
                    print(f"✅ Backup completed for decision {decision_count}")
                save_pnl_snapshot(reason=reason)  # 额外保存收益曲线
            except Exception as e:
                print(f"⚠️ Error during backup: {e}")
            # 注意：不在这里关闭 TinySoft 客户端，以保持会话复用
            # 会话会在 run_date_range 结束时统一关闭，避免频繁登录
            pass
    async def _handle_trading_result(
        self,
        today_date: str,
        decision_time: str,
        decision_count: int,
        log_file: str,
        final_agent_summary: Optional[str],
        collected_tool_errors: Optional[List[str]]
    ) -> None:
        """Handle trading results"""
        if_trade = get_config_value("IF_TRADE")
        if if_trade:
            write_config_value("IF_TRADE", False)
            print("✅ Trading completed")
            self._log_message(
                log_file,
                [{"role": "system", "content": "Trade executed during this session."}],
                decision_time=decision_time,
                decision_count=decision_count
            )
        else:
            print("📊 No trading, maintaining positions")
            try:
                add_no_trade_record(today_date, decision_time, decision_count, self.signature)
            except NameError as e:
                print(f"❌ NameError: {e}")
                raise
            write_config_value("IF_TRADE", False)
            needs_followup = not final_agent_summary or not str(final_agent_summary).strip()
            if needs_followup:
                reason_text = None
                if collected_tool_errors:
                    filtered = [msg for msg in collected_tool_errors if msg]
                    if filtered:
                        reason_text = "; ".join(dict.fromkeys(filtered))  # 去重保持顺序
                if reason_text is None:
                    reason_text = "工具执行未成功完成"
                clarification = (
                    "本轮交易最终未成交，已记录无交易。"
                    f"原因：{reason_text}。"
                )
                self._log_message(
                    log_file,
                    [{"role": "assistant", "content": clarification}],
                    decision_time=decision_time,
                    decision_count=decision_count
                )
    
    def _get_trading_hours(self, today_date: str) -> List[str]:
        return [
            f"{today_date} 10:30:00",  # 首次小时线可用
            f"{today_date} 11:30:00",  # 午间前（上午最后一根）
            f"{today_date} 14:00:00",  # 午后核心时段
        ]

    async def run_intraday_trading(self, today_date: str, start_index: int = 0) -> None:
        """
        每天进行3次交易决策：10:30、11:30、14:00。
        """
        trading_hours = self._get_trading_hours(today_date)

        for idx, current_time in enumerate(trading_hours[start_index:], start_index + 1):
            print(f"🕒 第 {idx}/3 次决策 - 时间: {current_time}")
            write_config_value("CURRENT_TIME", current_time)
            write_config_value("DECISION_COUNT", idx)  # 记录第几次决策
            await self.run_session_with_retry(today_date, current_time, decision_count=idx)

    def register_agent(self) -> None:
        """Register new agent, create initial positions"""
        position_exists = os.path.exists(self.position_file)
        if position_exists and not self.force_replay:
            print(f"⚠️ Position file {self.position_file} already exists, skipping registration")
            return
        if position_exists and self.force_replay:
            print(f"🗑️ force_replay=True, clearing existing data for {self.signature}")
            self._reset_agent_storage()
        
        # Ensure directory structure exists
        os.makedirs(self.data_path, exist_ok=True)
        position_dir = os.path.join(self.data_path, "position")
        log_dir = os.path.join(self.data_path, "log")
        os.makedirs(position_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Determine initial record date & decision time
        seed_date = self.init_date
        seed_decision_time = f"{self.init_date} 00:00:00"
        if not seed_date:
            seed_date = datetime.now().strftime("%Y-%m-%d")
            seed_decision_time = f"{seed_date} 00:00:00"
        
        # Create initial positions
        init_position = {symbol: {"shares": 0, "purchase_date": None} for symbol in self.stock_symbols}
        init_position['CASH'] = self.initial_cash
        
        with open(self.position_file, "w") as f:  # Use "w" mode to ensure creating new file
            f.write(json.dumps({
                "date": seed_date,
                "decision_time": seed_decision_time,
                "decision_count": 0,
                "id": 0, 
                "seed": True,
                "this_action": {"action": "seed", "symbol": "", "amount": 0},
                "positions": init_position
            }) + "\n")
        
        print(f"✅ Agent {self.signature} registration completed")
        print(f"📁 Position file: {self.position_file}")
        print(f"💰 Initial cash: ${self.initial_cash}")
        print(f"📊 Number of stocks: {len(self.stock_symbols)}")
    
    def get_trading_dates(self, init_date: str, end_date: str) -> List[str]:
        """
        Get trading date list
        
        Args:
            init_date: Start date
            end_date: End date
            
        Returns:
            List of trading dates
        """
        if not os.path.exists(self.position_file):
            self.register_agent()
        
        start_dt = datetime.strptime(init_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        last_completed: Optional[datetime] = None
        
        if os.path.exists(self.position_file):
            with open(self.position_file, "r") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                    except Exception:
                        continue
                    if doc.get("seed"):
                        continue
                    date_str = doc.get("date")
                    if not date_str:
                        continue
                    try:
                        decision_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    except Exception:
                        continue
                    if decision_dt < start_dt:
                        continue
                    if (last_completed is None) or (decision_dt > last_completed):
                        last_completed = decision_dt
        
        current_dt = start_dt
        if last_completed and last_completed >= start_dt:
            next_day = last_completed + timedelta(days=1)
            if next_day > current_dt:
                current_dt = next_day
        
        trading_dates: List[str] = []
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:
                trading_dates.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
        
        return trading_dates
    
    def _determine_resume_point(self, init_date: str, end_date: str) -> Optional[tuple[str, int]]:
        """
        根据持仓记录判断是否需要在某个日期/时间点重新开始。
        返回 (date, start_index)；start_index 对应 trading_hours 中的索引。
        """
        if not os.path.exists(self.position_file):
            return None

        try:
            init_dt = datetime.strptime(init_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except Exception:
            return None

        candidate_records: List[Dict[str, Any]] = []
        with open(self.position_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except Exception:
                    continue

                decision_time = doc.get("decision_time")
                decision_date = doc.get("date")
                if not decision_time or not decision_date:
                    continue

                try:
                    decision_date_obj = datetime.strptime(decision_date, "%Y-%m-%d").date()
                except Exception:
                    continue

                if decision_date_obj < init_dt or decision_date_obj > end_dt:
                    continue

                candidate_records.append(doc)

        if not candidate_records:
            return None

        candidate_records.sort(
            key=lambda item: (
                item.get("date", ""),
                item.get("decision_time", ""),
                item.get("id", 0),
            )
        )

        last_date = candidate_records[-1].get("date")
        if not last_date:
            return None

        trading_hours = self._get_trading_hours(last_date)
        recorded_times = sorted({
            item.get("decision_time")
            for item in candidate_records
            if item.get("date") == last_date and item.get("decision_time")
        })

        if not recorded_times:
            return None

        if set(recorded_times) >= set(trading_hours):
            # 最后一个交易日已全部完成
            return None

        last_time = recorded_times[-1]
        if last_time not in trading_hours:
            return None

        return last_date, trading_hours.index(last_time)
    
    async def run_session_with_retry(self, today_date: str, current_time: str, decision_count: int = 1) -> None:
        """Run method with retry"""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🔄 Attempting to run {self.signature} - {current_time} (Decision {decision_count}/3, Attempt {attempt})")
                await self.run_trading_session(today_date, current_time, decision_count)
                print(f"✅ {self.signature} - {current_time} run successful")
                return
            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {str(e)}")
                if attempt == self.max_retries:
                    print(f"💥 {self.signature} - {current_time} all retries failed")
                    raise
                else:
                    wait_time = self.base_delay * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
    
    async def run_date_range(self, init_date: str, end_date: str) -> None:
        """
        Run all trading days in date range
        
        Args:
            init_date: Start date
            end_date: End date
        """
        print(f"📅 Running date range: {init_date} to {end_date}")
        
        trading_dates = self.get_trading_dates(init_date, end_date)
        resume_point = self._determine_resume_point(init_date, end_date)

        start_indices: Dict[str, int] = {}
        dates_to_process: List[str] = []

        if resume_point:
            resume_date, resume_index = resume_point
            dates_to_process.append(resume_date)
            start_indices[resume_date] = resume_index

        for date in trading_dates:
            dates_to_process.append(date)

        # 保持顺序并去重
        ordered_dates: List[str] = []
        seen_dates = set()
        for date in dates_to_process:
            if date not in seen_dates:
                ordered_dates.append(date)
                seen_dates.add(date)

        if not ordered_dates:
            print(f"ℹ️ No trading days to process")
            if os.path.exists(self.position_file):
                print("💡 Hint: set FORCE_REPLAY=true (or agent_config.force_replay) to reset state for replays.")
            return
        
        print(f"📊 Trading days to process: {ordered_dates}")
        
        # Process each trading day
        for date in ordered_dates:
            print(f"🔄 Processing {self.signature} - Date: {date}")
            
            # Set configuration
            write_config_value("TODAY_DATE", date)
            write_config_value("SIGNATURE", self.signature)
            
            try:
                start_index = start_indices.get(date, 0)
                await self.run_intraday_trading(date, start_index=start_index)
            except Exception as e:
                print(f"❌ Error processing {self.signature} - Date: {date}")
                print(e)
                raise
        if self.dm:
            try:
                self.dm.close_ts_client(force=True)
            except Exception as close_err:
                print(f"⚠️ Failed to close TinySoft session: {close_err}")
        print(f"✅ {self.signature} processing completed")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary"""
        if not os.path.exists(self.position_file):
            return {"error": "Position file does not exist"}
        
        positions = []
        with open(self.position_file, "r") as f:
            for line in f:
                positions.append(json.loads(line))
        
        if not positions:
            return {"error": "No position records"}
        
        latest_position = positions[-1]
        return {
            "signature": self.signature,
            "latest_date": latest_position.get("date"),
            "positions": latest_position.get("positions", {}),
            "total_records": len(positions)
        }
    
    def __str__(self) -> str:
        return f"BaseAgent(signature='{self.signature}', basemodel='{self.basemodel}', stocks={len(self.stock_symbols)})"
    
    def __repr__(self) -> str:
        return self.__str__()
