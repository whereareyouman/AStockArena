"""
DataManager - 本地数据管理器
从 CSV 文件加载和提供股票价格和新闻数据
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from pandas.errors import OutOfBoundsDatetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json
import time  # 用于TinySoft查询延迟
from utils.json_file_manager import safe_update_json

logging.basicConfig(level=logging.INFO)


def normalize_stock_code(code: str) -> str:
    code = str(code).upper().strip()
    if code.startswith(("SH")):
        return code

    if len(code) == 6 and code.isdigit():
        if code.startswith(("688", "689", "600", "601", "603", "605", "730", "735")):
            return f"SH{code}"
    return code


class DataManager:
    """Loads and provides access to market news and stock price data."""

    _ts_login_blocked_until = 0.0
    _ts_unavailable_reason: Optional[str] = None

    def __init__(self, news_csv_path: Optional[str] = None, stock_csv_path: Optional[str] = None,
                 macro_csv_path: Optional[str] = None):
        self.logger = logging.getLogger("DataManager")
        self.news_df = None
        self.stock_df = None
        self.macro_df = None
        self.available_sources = []
        self.stock_symbols_available = []
        self._ts_cached_client = None
        self._ts_cached_ref = 0

        # --- News Loading ---
        if news_csv_path:
            if os.path.exists(news_csv_path):
                try:
                    self.logger.info(f"正在从以下位置加载新闻数据: {news_csv_path}")
                    self.news_df = pd.read_csv(news_csv_path)
                    # 兼容不同的列名（publish_time 或 created_at）
                    if 'publish_time' in self.news_df.columns and 'created_at' not in self.news_df.columns:
                        self.news_df['created_at'] = self.news_df['publish_time']
                    self.news_df['created_at'] = pd.to_datetime(self.news_df['created_at'], errors='coerce', utc=True)
                    self.news_df.dropna(subset=['created_at'], inplace=True)
                    self.news_df.sort_values(by='created_at', inplace=True)
                    self.available_sources.append('news')
                    self.logger.info(f"成功加载 {len(self.news_df)} 条新闻 (UTC 时间戳)")
                except OutOfBoundsDatetime:
                    self.logger.warning(f"新闻文件 {news_csv_path} 中出现 OutOfBoundsDatetime 错误，请检查数据范围",
                                        exc_info=False)
                except Exception as e:
                    self.logger.error(f"加载或处理新闻数据失败: {news_csv_path}", exc_info=True)
            else:
                self.logger.warning(f"新闻 CSV 路径 '{news_csv_path}' 未找到")

        # --- Macro News Loading ---
        if macro_csv_path:
            if os.path.exists(macro_csv_path):
                try:
                    self.logger.info(f"正在从以下位置加载宏观新闻数据: {macro_csv_path}")
                    self.macro_df = pd.read_csv(macro_csv_path)
                    self.macro_df['created_at'] = pd.to_datetime(self.macro_df['created_at'], errors='coerce', utc=True)
                    self.macro_df.dropna(subset=['created_at'], inplace=True)
                    self.macro_df.sort_values(by='created_at', inplace=True)
                    self.available_sources.append('macro_news')
                    self.logger.info(f"成功加载 {len(self.macro_df)} 条宏观新闻 (UTC 时间戳)")
                except OutOfBoundsDatetime:
                    self.logger.warning(f"宏观新闻文件 {macro_csv_path} 中出现 OutOfBoundsDatetime 错误，请检查数据范围",
                                        exc_info=False)
                except Exception as e:
                    self.logger.error(f"加载或处理宏观新闻数据失败: {macro_csv_path}", exc_info=True)
            else:
                self.logger.warning(f"宏观新闻 CSV 路径 '{macro_csv_path}' 未找到")

        # --- Stock Loading ---
        if stock_csv_path:
            if os.path.exists(stock_csv_path):
                try:
                    self.logger.info(f"正在从以下位置加载股票数据: {stock_csv_path}")
                    self.stock_df = pd.read_csv(stock_csv_path)
                    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                    if not all(col in self.stock_df.columns for col in required_cols):
                        missing = [col for col in required_cols if col not in self.stock_df.columns]
                        raise ValueError(
                            f"股票 CSV 缺少必需的列: {missing}. 找到的列: {list(self.stock_df.columns)}")

                    try:
                        self.stock_df['timestamp'] = pd.to_datetime(self.stock_df['timestamp'], infer_datetime_format=True,
                                                                    errors='coerce', utc=True)
                    except Exception:
                        self.logger.info("推断日期时间格式失败，使用标准转换处理股票数据")
                        self.stock_df['timestamp'] = pd.to_datetime(self.stock_df['timestamp'], errors='coerce', utc=True)

                    if self.stock_df['timestamp'].isnull().any():
                        self.logger.warning(
                            f"由于解析错误删除了 {self.stock_df['timestamp'].isnull().sum()} 行股票数据")
                        self.stock_df.dropna(subset=['timestamp'], inplace=True)

                    self.stock_df['symbol'] = self.stock_df['symbol'].astype(str).str.upper()
                    self.stock_df.dropna(subset=['symbol', 'close'], inplace=True)
                    self.stock_df.set_index('timestamp', inplace=True)
                    self.stock_df.sort_index(inplace=True)

                    self.stock_symbols_available = list(self.stock_df['symbol'].unique())
                    self.available_sources.append('stock_prices')
                    self.logger.info(f"成功加载 {len(self.stock_df)} 条股票数据点 (UTC 时间戳)")
                    self.logger.info(f"可用股票代码: {self.stock_symbols_available}")

                except OutOfBoundsDatetime:
                    self.logger.warning(f"股票文件 {stock_csv_path} 中出现 OutOfBoundsDatetime 错误，请检查数据范围",
                                        exc_info=False)
                    self.stock_df = None
                    self.stock_symbols_available = []
                except Exception as e:
                    self.logger.error(f"加载或处理股票数据失败: {stock_csv_path}", exc_info=True)
                    self.stock_df = None
                    self.stock_symbols_available = []
            else:
                self.logger.warning(f"股票 CSV 路径 '{stock_csv_path}' 未找到")

    def get_available_sources(self) -> List[str]:
        return self.available_sources

    def get_available_symbols(self) -> List[str]:
        return self.stock_symbols_available

    def _ensure_utc(self, dt_input: Any) -> Optional[pd.Timestamp]:
        if dt_input is None:
            return None
        try:
            dt = pd.to_datetime(dt_input, utc=True)
            if not isinstance(dt, pd.Timestamp):
                dt = pd.Timestamp(dt)
            return dt
        except (ValueError, TypeError, OutOfBoundsDatetime) as e:
            self.logger.warning(f"无法将 '{dt_input}' 转换为有效的 UTC 时间戳: {e}", exc_info=False)
            return None

    def get_macro_news(self,
                       end_date: Optional[Any] = None,
                       start_date: Optional[Any] = None,
                       symbols: Optional[List[str]] = None,
                       limit: int = 50) -> pd.DataFrame:
        """
        检索宏观新闻。主方案是获取最近特定天数内的宏观新闻。
        如果主方案未找到任何新闻，则激活备用方案，搜索所有历史。
        """
        LOOKBACK_DAYS = 30
        PRIMARY_LIMIT = 3

        if self.macro_df is None or self.macro_df.empty:
            return pd.DataFrame()

        end_dt = self._ensure_utc(end_date) if end_date else pd.Timestamp.now(tz='UTC')
        if end_dt is None:
            self.logger.error(f"无效的 end_date '{end_date}'。操作中止。")
            return pd.DataFrame()

        try:
            primary_start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
            self.logger.info(f"主方案 (Macro): 正在搜索最近 {LOOKBACK_DAYS} 天的宏观新闻 (上限 {PRIMARY_LIMIT} 条)...")

            date_mask = (self.macro_df['created_at'] >= primary_start_dt) & (self.macro_df['created_at'] <= end_dt)
            filtered_df = self.macro_df[date_mask].copy()

            if symbols and 'symbols' in filtered_df.columns:
                symbols_upper = [s.upper() for s in symbols]

                def check_symbols(row_symbols):
                    if pd.isna(row_symbols):
                        return False
                    try:
                        symbol_list_in_row = [s.strip().upper() for s in
                                              str(row_symbols).replace('[', '').replace(']', '').replace("'", '').split(
                                                  ',')]
                        return any(s in symbol_list_in_row for s in symbols_upper)
                    except Exception:
                        return False

                filtered_df = filtered_df[filtered_df['symbols'].apply(check_symbols)]

            filtered_df.sort_values(by='created_at', ascending=False, inplace=True)
            primary_result_df = filtered_df.head(PRIMARY_LIMIT)

            if not primary_result_df.empty:
                self.logger.info(f"主方案 (Macro) 成功，找到 {len(primary_result_df)} 条宏观新闻。")
                return primary_result_df
            else:
                self.logger.warning(
                    f"主方案 (Macro) 在最近 {LOOKBACK_DAYS} 天内未找到宏观新闻。正在激活备用方案..."
                )

                fallback_limit = limit
                self.logger.info(f"备用方案 (Macro): 搜索范围扩大至所有历史，limit 为 {fallback_limit} 条。")

                fallback_filtered_df = self.macro_df[self.macro_df['created_at'] <= end_dt].copy()

                if symbols and 'symbols' in fallback_filtered_df.columns:
                    symbols_upper = [s.upper() for s in symbols]

                    def check_symbols(row_symbols):
                        if pd.isna(row_symbols):
                            return False
                        try:
                            symbol_list_in_row = [s.strip().upper() for s in
                                                  str(row_symbols).replace('[', '').replace(']', '').replace("'",
                                                                                                             '').split(
                                                      ',')]
                            return any(s in symbol_list_in_row for s in symbols_upper)
                        except Exception:
                            return False

                    fallback_filtered_df = fallback_filtered_df[fallback_filtered_df['symbols'].apply(check_symbols)]

                fallback_filtered_df.sort_values(by='created_at', ascending=False, inplace=True)
                fallback_result_df = fallback_filtered_df.head(fallback_limit)

                self.logger.info(f"备用方案 (Macro) 完成，找到 {len(fallback_result_df)} 条宏观新闻。")
                return fallback_result_df

        except Exception as e:
            self.logger.error(
                f"在 get_macro_news 函数中发生未知错误 (end_date: {end_dt.date()}, 股票: {symbols}): {e}",
                exc_info=True)
            return pd.DataFrame()
    def _filter_and_get(self,
                        df_to_filter: pd.DataFrame,
                        start_dt: Optional[pd.Timestamp],
                        end_dt: pd.Timestamp,
                        symbols: Optional[List[str]],
                        limit: int) -> pd.DataFrame:
        """
        [新] 内部辅助函数，封装了原始的筛选和获取逻辑。
        """
        try:
            # 数据已在加载时按降序排序，所以筛选更高效
            filtered_df = df_to_filter[df_to_filter['created_at'] <= end_dt]
            if start_dt:
                filtered_df = filtered_df[filtered_df['created_at'] >= start_dt]

            # 按 symbols 筛选
            if symbols and 'symbols' in filtered_df.columns:
                symbols_upper = [s.upper() for s in symbols]

                def check_symbols(row_symbols):
                    if pd.isna(row_symbols): return False
                    try:
                        symbol_list_in_row = [s.strip().upper() for s in
                                              str(row_symbols).replace('[', '').replace(']', '').replace("'", '').split(
                                                  ',')]
                        return any(s in symbol_list_in_row for s in symbols_upper)
                    except:
                        return False

                # 使用 .copy() 避免 SettingWithCopyWarning
                filtered_df = filtered_df[filtered_df['symbols'].apply(check_symbols)].copy()

            # head(limit) 直接获取最新的 N 条记录
            return filtered_df.head(limit)
        except Exception as e:
            self.logger.error(f"Error during _filter_and_get: {e}", exc_info=True)
            return pd.DataFrame()
            
    def get_news(self, start_date: Optional[Any] = None, end_date: Optional[Any] = None,
                 symbols: Optional[List[str]] = None, limit: int = 150) -> pd.DataFrame:
        """
        检索新闻，仅返回「决策时点往前2天的00:00:00 到 决策时点」窗口内的数据，用于智能体读取历史背景。
        例如：如果决策时点是 2026-01-22 10:00:00，则返回 2026-01-20 00:00:00 到 2026-01-22 10:00:00 的新闻。
        （实时搜索 AKShare 获取的新新闻不受此限制。）
        """
        PRIMARY_LIMIT = 25

        if self.news_df is None or self.news_df.empty:
            return pd.DataFrame()

        # 处理 end_date：如果它是字符串且没有时区信息，假设它是 Asia/Shanghai 时区，然后转换为 UTC
        if end_date:
            if isinstance(end_date, str) and 'T' not in end_date and '+' not in end_date and end_date.count(':') >= 1:
                # 格式如 "2026-01-22 10:00:00"，假设是 Asia/Shanghai 时区
                try:
                    naive_dt = pd.to_datetime(end_date)
                    # 先本地化为 Asia/Shanghai，再转换为 UTC
                    end_dt = naive_dt.tz_localize('Asia/Shanghai').tz_convert('UTC')
                except Exception:
                    # 如果转换失败，使用 _ensure_utc 作为回退
                    end_dt = self._ensure_utc(end_date)
            else:
                end_dt = self._ensure_utc(end_date)
        else:
            end_dt = pd.Timestamp.now(tz='UTC')
        
        if end_dt is None:
            self.logger.error(f"无效的 end_date '{end_date}'。操作中止。")
            return pd.DataFrame()

        try:
            # 计算开始时间：决策时点往前推2天的00:00:00（Asia/Shanghai 时区的00:00:00）
            # 例如：如果 end_dt 是 2026-01-22 02:00:00 UTC（对应 2026-01-22 10:00:00 Asia/Shanghai），
            # 则 start_dt 应该是 2026-01-20 00:00:00 Asia/Shanghai（即 2026-01-19 16:00:00 UTC）
            # 为了简化，我们使用 end_dt 的日期部分（在 Asia/Shanghai 时区）来计算
            end_dt_shanghai = end_dt.tz_convert('Asia/Shanghai')
            end_date_only = end_dt_shanghai.date()
            start_date_only = end_date_only - timedelta(days=2)
            # 将开始日期转换为当天的00:00:00（Asia/Shanghai 时区），然后转换为 UTC
            start_dt_shanghai = pd.Timestamp.combine(start_date_only, datetime.min.time()).tz_localize('Asia/Shanghai')
            start_dt = start_dt_shanghai.tz_convert('UTC')
            
            # 日志输出使用 Asia/Shanghai 时区，更易读
            start_dt_display = start_dt.tz_convert('Asia/Shanghai')
            end_dt_display = end_dt.tz_convert('Asia/Shanghai')
            self.logger.info(f"新闻窗口: 从 {start_dt_display.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_dt_display.strftime('%Y-%m-%d %H:%M:%S')} (Asia/Shanghai, 上限 {PRIMARY_LIMIT} 条)...")

            date_mask = (self.news_df['created_at'] >= start_dt) & (self.news_df['created_at'] <= end_dt)
            filtered_df = self.news_df[date_mask].copy()

            if symbols and 'symbols' in filtered_df.columns:
                symbols_upper = [s.upper() for s in symbols]

                def check_symbols(row_symbols):
                    if pd.isna(row_symbols):
                        return False
                    try:
                        symbol_list_in_row = [s.strip().upper() for s in
                                              str(row_symbols).replace('[', '').replace(']', '').replace("'", '').split(
                                                  ',')]
                        return any(s in symbol_list_in_row for s in symbols_upper)
                    except Exception:
                        return False

                filtered_df = filtered_df[filtered_df['symbols'].apply(check_symbols)]

            filtered_df.sort_values(by='created_at', ascending=False, inplace=True)
            primary_result_df = filtered_df.head(min(PRIMARY_LIMIT, limit))

            # 日志输出使用 Asia/Shanghai 时区，更易读
            start_dt_display = start_dt.tz_convert('Asia/Shanghai')
            end_dt_display = end_dt.tz_convert('Asia/Shanghai')
            if primary_result_df.empty:
                self.logger.info(f"在时间窗口 [{start_dt_display.strftime('%Y-%m-%d %H:%M:%S')}, {end_dt_display.strftime('%Y-%m-%d %H:%M:%S')}] (Asia/Shanghai) 内没有可用的历史新闻，将返回空结果。")
            else:
                self.logger.info(f"找到 {len(primary_result_df)} 条新闻（限制在时间窗口 [{start_dt_display.strftime('%Y-%m-%d %H:%M:%S')}, {end_dt_display.strftime('%Y-%m-%d %H:%M:%S')}] (Asia/Shanghai) 内）。")
            return primary_result_df

        except Exception as e:
            self.logger.error(
                f"在 get_news 函数中发生未知错误 (end_date: {end_dt.date()}, 股票: {symbols}): {e}",
                exc_info=True)
            return pd.DataFrame()

    def get_stock_data(self, symbol: str, end_date: Any, start_date: Optional[Any] = None,
                       lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        已弃用：此方法原返回日线数据，现在改为返回小时线数据。
        建议直接使用 get_hourly_stock_data 方法。
        
        获取小时线数据（作为日线数据的替代）；如未提供 start_date 则按 lookback_days 回溯。
        返回列: [open, high, low, close, volume]，索引为 UTC 时间戳。
        """
        self.logger.warning(f"[TS] get_stock_data 已改为返回小时线数据，建议使用 get_hourly_stock_data")
        
        # 转换为小时数：每天4小时交易时间
        lookback_hours = (lookback_days or 1) * 4 if lookback_days else 4
        
        # 如果 end_date 只包含日期，添加收盘时间
        if isinstance(end_date, str) and len(end_date) == 10:  # YYYY-MM-DD
            end_time = f"{end_date} 15:00:00"
        else:
            end_time = end_date
        
        # 使用小时线查询方法
        return self.get_hourly_stock_data(symbol=symbol, end_date=end_time, lookback_hours=lookback_hours)

    def get_hourly_stock_data(self, symbol: str, end_date: Any, lookback_hours: int = 24) -> pd.DataFrame:
        """
        获取单个股票的小时线数据（60分钟K线）。
        
        Args:
            symbol (str): 股票代码（如 "600519"）
            end_date (Any): 结束日期时间（YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）
            lookback_hours (int): 往前查找多少小时的数据，默认24小时
        
        Returns:
            pd.DataFrame: 包含 [timestamp, symbol, open, high, low, close, volume] 的DataFrame
                          索引为 UTC 时间戳
        """
        symbol_upper = str(symbol).upper()
        
        # 解析时间窗口
        end_dt_local = pd.to_datetime(end_date) if end_date is not None else datetime.now()
        start_dt_local = end_dt_local - timedelta(hours=lookback_hours)
        
        with self._ts_client_session() as client:
            if client is None:
                self._log_ts_unavailable("无法获取小时线数据")
                return pd.DataFrame()
        
            try:
                # 使用 60分钟线周期
                r = client.query(
                    stock=symbol_upper,
                    begin_time=start_dt_local.to_pydatetime(),
                    end_time=end_dt_local.to_pydatetime(),
                    cycle='60分钟线',
                    fields='date, open, high, low, close, vol'
                )
                if r.error() != 0:
                    self.logger.warning(f"TinySoft 小时线查询失败: {r.message()}")
                    return pd.DataFrame()
                
                df = r.dataframe()
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # 统一列名
                rename_map = {
                    'date': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'vol': 'volume',
                    'volume': 'volume'
                }
                for k, v in list(rename_map.items()):
                    if k in df.columns and v not in df.columns:
                        df[v] = df[k]
                
                if 'timestamp' not in df.columns:
                    if 'date' in df.columns:
                        df['timestamp'] = df['date']
                    else:
                        self.logger.warning("TinySoft 返回缺少日期字段")
                        return pd.DataFrame()
                
                # 填入 symbol、转时区、排序
                df['symbol'] = symbol_upper
                ts = pd.to_datetime(df['timestamp'])
                try:
                    ts = ts.dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC')
                except Exception:
                    ts = pd.to_datetime(df['timestamp'], utc=True)
                df['timestamp'] = ts
                df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                self.logger.info(
                    f"[TS] get_hourly_stock_data 返回 {len(df)} 行，{symbol_upper} 从 {start_dt_local} 至 {end_dt_local}"
                )
                return df
            except Exception as e:
                self.logger.error(f"TinySoft 获取小时线时异常: {e}", exc_info=True)
                return pd.DataFrame()

    # ------------------- TinySoft 集成 -------------------
    def _create_ts_client(self):
        """创建新的 TinySoft 客户端实例并登录。
        如果遇到 Relogin refused，会先尝试清理旧会话再重试登录。
        """
        now = time.time()
        if DataManager._ts_login_blocked_until and now < DataManager._ts_login_blocked_until:
            return None

        try:
            import pyTSL as ts
        except Exception as e:
            self.logger.error(f"未安装或无法导入 pyTSL: {e}")
            DataManager._set_ts_unavailable("pyTSL 未安装或导入失败", cooldown=0)
            return None

        username = os.getenv("TSL_USERNAME") or os.getenv("TSL_USER")
        password = os.getenv("TSL_PASSWORD") or os.getenv("TSL_PASS")
        server = os.getenv("TSL_SERVER", "tsl.tinysoft.com.cn")
        port = int(os.getenv("TSL_PORT", "443"))

        if not username or not password:
            self.logger.error("缺少 TinySoft 账号环境变量 TSL_USERNAME / TSL_PASSWORD")
            DataManager._set_ts_unavailable("缺少账户配置", cooldown=0)
            return None

        # 如果已有缓存的客户端，先尝试清理（强制关闭旧会话）
        if self._ts_cached_client is not None:
            self.logger.info("检测到已有缓存的 TinySoft 客户端，先清理旧会话...")
            try:
                old_client = self._ts_cached_client
                self._ts_cached_client = None
                self._ts_cached_ref = 0
                self._logout_ts_client(old_client)
                self.logger.info("旧会话已清理，等待 3 秒后重新登录...")
                time.sleep(3)  # 等待旧会话完全释放
            except Exception as cleanup_err:
                self.logger.warning(f"清理旧会话时异常: {cleanup_err}")

        try:
            client = ts.Client(username, password, server, port)

            # 尝试登录
            ok = client.login()
            if ok != 1:
                last_err = getattr(client, "last_error", lambda: "unknown")()
                err_text = str(last_err)
                if "Relogin refused" in err_text:
                    self.logger.warning("TinySoft 登录被拒（可能已有会话在线）。")
                    # 先尝试登出当前客户端
                    try:
                        client.logout()
                        self.logger.info("已执行 logout()，等待 5 秒后重试登录...")
                        time.sleep(5)
                        # 重新创建客户端并重试登录
                        client = ts.Client(username, password, server, port)
                        ok = client.login()
                        if ok == 1:
                            self.logger.info("重试登录成功（通过等待后重新登录）")
                        else:
                            self.logger.warning("重试登录仍然失败，可能被其他进程占用")
                            try:
                                client.logout()
                            except Exception:
                                pass
                            DataManager._set_ts_unavailable("登录被拒（可能已有会话在线）", cooldown=180)
                            self.logger.warning("请等待其他会话结束或重启程序后重试。")
                            return None
                    except Exception as retry_err:
                        self.logger.warning(f"重试登录时异常: {retry_err}")
                        try:
                            client.logout()
                        except Exception as logout_err:
                            self.logger.debug(f"Relogin refused 后执行 logout() 失败: {logout_err}")
                        DataManager._set_ts_unavailable("登录被拒（可能已有会话在线）", cooldown=180)
                        self.logger.warning("可运行 'python tsl_logout.py' 清理历史会话后重试。")
                        return None
                else:
                    self.logger.error(f"TinySoft 登录失败: {last_err}")
                    DataManager._set_ts_unavailable(f"登录失败：{err_text}", cooldown=60)
                    try:
                        client.logout()
                    except Exception as logout_err:
                        self.logger.debug(f"登录失败后执行 logout() 失败: {logout_err}")
                    return None

            # 登录成功
            DataManager._ts_unavailable_reason = None
            DataManager._ts_login_blocked_until = 0.0
            self.logger.info("TinySoft 登录成功")
            return client
        except Exception as e:
            self.logger.error(f"TinySoft 客户端初始化异常: {e}")
            DataManager._set_ts_unavailable("客户端初始化异常", cooldown=60)
            return None

    def _logout_ts_client(self, client) -> None:
        """安全登出 TinySoft 客户端。根据天软官方建议，logout 后删除对象并等待1-2秒确保注销完成。"""
        if client is None:
            return
        try:
            client.logout()
            # 根据天软官方建议：logout 后删除对象，避免自动重新登录
            del client
            # sleep 1-2秒，保证本次登录注销完成，避免占用多个登录数
            time.sleep(1.5)
        except Exception as e:
            self.logger.warning(f"TinySoft 登出异常: {e}")

    @contextmanager
    def _ts_client_session(self):
        client = self._ts_cached_client
        if client is None:
            client = self._create_ts_client()
            if client is None:
                yield None
                return
            self._ts_cached_client = client

        self._ts_cached_ref += 1
        try:
            yield client
        finally:
            self._ts_cached_ref -= 1
            if self._ts_cached_ref < 0:
                self._ts_cached_ref = 0

    def close_ts_client(self, force: bool = False) -> None:
        """
        主动关闭缓存的 TinySoft 会话。
        force=True 时无视引用计数立即登出，用于进程结束或异常恢复。
        根据天软官方建议，遵循"一次登录，多次交互"原则，避免频繁登录退出。
        """
        client = self._ts_cached_client
        if client is None:
            return
        if self._ts_cached_ref > 0 and not force:
            return
        # 先保存客户端引用，然后清空缓存，再执行 logout（避免在 logout 过程中被其他线程使用）
        self._ts_cached_client = None
        self._ts_cached_ref = 0
        self._logout_ts_client(client)

    @classmethod
    def _set_ts_unavailable(cls, reason: str, cooldown: int = 60):
        cls._ts_unavailable_reason = reason
        if cooldown:
            cls._ts_login_blocked_until = max(cls._ts_login_blocked_until, time.time() + cooldown)

    def _log_ts_unavailable(self, context: str):
        reason = DataManager._ts_unavailable_reason
        if reason:
            self.logger.warning(f"TinySoft 客户端不可用（{reason}），{context}")
        else:
            self.logger.error(f"TinySoft 客户端不可用，{context}")

    # ------------------- TinySoft 扩展：打包价格+指标并保存 -------------------
    def _ts_query_day(self, c, symbol: str, begin_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """
        已弃用：日线查询方法。系统已改为完全基于小时线数据。
        保留此方法仅为向后兼容，建议使用 _ts_query_hour 或 get_hourly_stock_data。
        """
        self.logger.warning(f"[TS] _ts_query_day 已弃用，建议使用小时线查询")
        return None

    def _ts_query_hour(self, c, symbol: str, begin_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        try:
            r = c.query(stock=symbol, begin_time=begin_time, end_time=end_time, cycle='60分钟线', fields='date, close, vol, amount, buy1')
            if r.error() != 0:
                self.logger.warning(f"[TS] 小时线查询失败: {r.message()}")
                return None
            df = r.dataframe()
            if df is None or df.empty:
                return None
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            return df
        except Exception as e:
            self.logger.warning(f"[TS] 小时线查询异常: {e}")
            return None

    def _ts_query_indicators(self, c, symbol: str, ndays: int, cycle: str) -> Optional[List[Dict[str, Any]]]:
        """
        查询技术指标。cycle: '60m'（小时线指标）
        
        注意：日线指标已弃用，不再支持 cycle='day'。
        """
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            
            # 只支持小时线指标
            if cycle != '60m':
                self.logger.warning(f"[TS] 指标查询仅支持 '60m'（小时线），不支持 '{cycle}'")
                cycle = '60m'  # 默认使用小时线
            
            # 小时线指标
            tsl = f"""
SetSysParam(pn_stock(),'{symbol}');
SetSysParam(pn_date(),{end_date}T);
setsysparam(pn_cycle(),cy_60m());
setsysparam(pn_rate(),0);
setsysparam(pn_Nday(),{ndays});
V:=KDJ_f(9,3,3,1);
B:=boll_f(20,1);
D:=Nday2("Date",datetostr(sp_time()));
Return D|V[0]|V[1]|V[2]|B[0]|B[1]|B[2]|B[3];
"""
            r = c.exec(tsl)
            if r.error() != 0:
                self.logger.warning(f"[TS] 小时线指标查询失败: {r.message()}")
                return None
            return r.value()
        except Exception as e:
            self.logger.warning(f"[TS] 小时线指标查询异常: {e}")
            return None

    def _batch_query_hour_data(self, c, symbols: List[str], begin_time: datetime, end_time: datetime) -> Dict[str, Optional[pd.DataFrame]]:
        """
        批量查询多只股票的小时线行情数据。
        
        Args:
            c: TinySoft 客户端
            symbols: 股票代码列表
            begin_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict[str, Optional[pd.DataFrame]]: 股票代码 -> DataFrame 的映射，查询失败返回 None
        """
        result = {}
        for symbol in symbols:
            symbol_normalized = normalize_stock_code(symbol)  # 归一化代码
            try:
                hour_df = self._ts_query_hour(c, symbol_normalized, begin_time, end_time)
                result[symbol_normalized] = hour_df
            except Exception as e:
                self.logger.warning(f"[TS] 批量查询小时线行情失败 {symbol_normalized}: {e}")
                result[symbol_normalized] = None
            # 批次间延迟，避免并发过多
            time.sleep(0.5)
        return result

    def _batch_query_hour_indicators(self, c, symbols: List[str], ndays: int) -> Dict[str, Optional[List[Dict[str, Any]]]]:
        """
        批量查询多只股票的小时线指标数据。
        
        Args:
            c: TinySoft 客户端
            symbols: 股票代码列表
            ndays: 回溯天数
            
        Returns:
            Dict[str, Optional[List[Dict[str, Any]]]]: 股票代码 -> 指标列表的映射，查询失败返回 None
        """
        result = {}
        for symbol in symbols:
            symbol_normalized = normalize_stock_code(symbol)  # 归一化代码
            try:
                hour_ind = self._ts_query_indicators(c, symbol_normalized, ndays=int(ndays), cycle='60m')
                result[symbol_normalized] = hour_ind
            except Exception as e:
                self.logger.warning(f"[TS] 批量查询小时线指标失败 {symbol_normalized}: {e}")
                result[symbol_normalized] = None
            # 批次间延迟，避免并发过多
            time.sleep(0.5)
        return result

    # 全局锁，确保同一时间只有一个save_ts_data在执行
    _save_ts_data_lock = None
    _lock_init = False
    
    def save_ts_data(self, symbols: List[str], ndays: int, out_path: str) -> bool:
        """
        为给定股票列表批量采集 TinySoft 的小时线行情与指标，保存为 JSON 文件。
        如果文件已存在，会读取并合并历史数据，避免重复。
        使用文件锁确保并发安全，即使TinySoft查询失败也不会丢失历史数据。
        使用全局锁确保同一时间只有一个save_ts_data在执行，避免TinySoft并发过多。
        
        优化：批量查询所有股票的小时线行情和小时线指标，而不是逐个处理。
        """
        # 初始化全局锁（只在第一次调用时）
        if not DataManager._lock_init:
            try:
                import threading
                DataManager._save_ts_data_lock = threading.Lock()
                DataManager._lock_init = True
            except Exception:
                pass  # 如果没有threading，继续执行但无锁保护
        
        # 如果有全局锁，先获取锁
        if DataManager._save_ts_data_lock:
            DataManager._save_ts_data_lock.acquire()
        
        try:
            with self._ts_client_session() as c:
                if c is None:
                    self._log_ts_unavailable("无法导出数据")
                    return False

                # 确保查询到当前时间的最新数据
                # 使用当前时间作为结束时间，确保能获取到最新的可用数据
                now = datetime.now()
                begin_time = now - timedelta(days=int(ndays))
                end_time = now
                
                # 注意：TinySoft 查询会返回 begin_time 到 end_time 之间所有可用的数据
                # 如果某些时间点的数据还未生成（如交易日内未来时间点），则不会返回
                # 这是正常的，因为数据是按实际交易时间生成的

                # 批量查询所有股票的小时线行情
                self.logger.info(f"开始批量查询 {len(symbols)} 只股票的小时线行情...")
                hour_data_dict = self._batch_query_hour_data(c, symbols, begin_time, end_time)
                
                # 批次间延迟，避免并发过多
                time.sleep(0.5)
                
                # 批量查询所有股票的小时线指标
                self.logger.info(f"开始批量查询 {len(symbols)} 只股票的小时线指标...")
                hour_indicators_dict = self._batch_query_hour_indicators(c, symbols, ndays)

                def update_data(existing_data: Dict[str, Any]) -> Dict[str, Any]:
                    """更新函数：在文件锁保护下更新数据"""
                    # 复制现有数据，确保不修改原数据
                    payload: Dict[str, Any] = existing_data.copy()
                    
                    for symbol in symbols:
                        symbol_upper = normalize_stock_code(symbol)  # 归一化代码，确保带 SH/SZ 前缀
                        # 获取该股票的历史数据（如果有）
                        data: Dict[str, Any] = {}
                        if symbol_upper in existing_data:
                            # 深拷贝历史数据，避免修改原数据
                            import copy
                            data = copy.deepcopy(existing_data[symbol_upper])
                        else:
                            # 新股票，初始化为空结构
                            data = {}

                        # 更新小时线行情（从批量查询结果获取）
                        hour_df = hour_data_dict.get(symbol_upper)
                        if hour_df is not None and not hour_df.empty:
                            new_hour_records = hour_df.to_dict(orient='records')
                            # 合并并去重（基于日期+时间）
                            if '小时线行情' in data and data['小时线行情']:
                                existing_datetimes = {r['date']: i for i, r in enumerate(data['小时线行情'])}
                                for new_record in new_hour_records:
                                    if new_record['date'] in existing_datetimes:
                                        data['小时线行情'][existing_datetimes[new_record['date']]] = new_record
                                    else:
                                        data['小时线行情'].append(new_record)
                                data['小时线行情'].sort(key=lambda x: x['date'])
                            else:
                                data['小时线行情'] = new_hour_records
                            self.logger.debug(f"已更新 {symbol_upper} 的小时线行情: {len(data.get('小时线行情', []))} 条")
                        elif '小时线行情' not in data and symbol_upper not in existing_data:
                            # TinySoft查询失败且是新股票，不设置小时线行情（保留为空）
                            self.logger.warning(f"无法获取 {symbol_upper} 的小时线行情，保留为空")
                        else:
                            # TinySoft查询失败但已有历史数据，保留历史数据
                            self.logger.warning(f"无法获取 {symbol_upper} 的小时线行情，保留历史数据")

                        # 更新小时线指标（从批量查询结果获取）
                        hour_ind = hour_indicators_dict.get(symbol_upper)
                        if hour_ind is not None:
                            data['小时线指标'] = hour_ind
                            self.logger.debug(f"已更新 {symbol_upper} 的小时线指标")
                        elif '小时线指标' not in data:
                            # 如果没有历史指标，不设置（避免覆盖为空）
                            self.logger.warning(f"无法获取 {symbol_upper} 的小时线指标")

                        # 只有在该股票有数据时才添加到payload
                        if data:
                            payload[symbol_upper] = data
                    
                    return payload

                # 使用文件锁安全更新
                try:
                    success = safe_update_json(out_path, update_data, default={})
                    if success:
                        self.logger.info(f"已保存 TinySoft 小时线数据至 {out_path}（合并历史数据，{len(symbols)} 只股票）")
                    else:
                        self.logger.error(f"保存 {out_path} 失败")
                    return success
                except Exception as e:
                    self.logger.error(f"保存 {out_path} 失败: {e}")
                    return False
        except Exception as e:
            self.logger.error(f"TinySoft数据采集失败: {e}")
            return False
        finally:
            # 释放全局锁
            if DataManager._save_ts_data_lock:
                try:
                    DataManager._save_ts_data_lock.release()
                except Exception:
                    pass

    def get_indicators_ts(self, symbol: str, ndays: int = 7, cycle: str = '60m') -> Optional[pd.DataFrame]:
        """
        获取 TinySoft 小时线技术指标（当前包含 KDJ、BOLL），返回 DataFrame（列：Date,K,D,J,BOLL,UPR,DWN,CLOSE）。
        
        注意：日线指标已弃用，仅支持 cycle='60m'（小时线）。
        """
        with self._ts_client_session() as c:
            if c is None:
                self._log_ts_unavailable("无法获取指标")
                return None

            try:
                raw = self._ts_query_indicators(c, str(symbol).upper(), ndays=int(ndays), cycle=cycle)
                if raw is None:
                    return None

                # 规范化为 DataFrame
                if isinstance(raw, pd.DataFrame):
                    df = raw.copy()
                elif isinstance(raw, list):
                    # 可能是 list[dict] 或 list[list]
                    if len(raw) > 0 and isinstance(raw[0], dict):
                        df = pd.DataFrame(raw)
                    else:
                        # 尝试用固定列构造
                        cols = ["Date","K","D","J","BOLL","UPR","DWN","CLOSE"]
                        df = pd.DataFrame(raw, columns=cols[:len(raw[0])]) if len(raw) > 0 else pd.DataFrame(columns=cols)
                else:
                    # 兜底为空
                    return None

                # 列名标准化
                rename_map = {"Date":"Date","K":"K","D":"D","J":"J","BOLL":"BOLL","UPR":"UPR","DWN":"DWN","CLOSE":"CLOSE"}
                df = df.rename(columns=rename_map)
                # 排序与类型处理（如有 Date 列）
                if "Date" in df.columns:
                    try:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.sort_values("Date")
                    except Exception:
                        pass
                return df
            except Exception as e:
                self.logger.error(f"获取 TinySoft 指标异常: {e}")
                return None

    def get_data_summary(self) -> Dict[str, Any]:
        """提供可用数据源及其大致时间范围的摘要"""
        summary = {"available_sources": self.available_sources}
        if self.news_df is not None and not self.news_df.empty:
            try:
                min_news_date = self.news_df['created_at'].min().strftime('%Y-%m-%d')
                max_news_date = self.news_df['created_at'].max().strftime('%Y-%m-%d')
                summary['news_range'] = f"{min_news_date} 至 {max_news_date}"
            except Exception as e:
                summary['news_range'] = f"获取范围时出错: {e}"
        else:
            summary['news_range'] = "N/A"

        if self.stock_df is not None and not self.stock_df.empty:
            try:
                min_stock_date = self.stock_df.index.min().strftime('%Y-%m-%d')
                max_stock_date = self.stock_df.index.max().strftime('%Y-%m-%d')
                summary['stock_range'] = f"{min_stock_date} 至 {max_stock_date}"
                summary['stock_symbols'] = self.stock_symbols_available
            except Exception as e:
                summary['stock_range'] = f"获取范围时出错: {e}"
                summary['stock_symbols'] = []
        else:
            summary['stock_range'] = "N/A"
            summary['stock_symbols'] = []

        self.logger.debug(f"数据摘要: {summary}")
        return summary

    def get_price_at(self, symbol: str, target_time: Any) -> Optional[float]:
        """获取指定时间或之前最近的收盘价"""
        if self.stock_df is None or self.stock_df.empty:
            return None
        symbol_upper = symbol.upper()
        if symbol_upper not in self.stock_symbols_available:
            self.logger.warning(f"get_price_at: 股票代码 '{symbol_upper}' 不可用")
            return None

        target_dt = self._ensure_utc(target_time)
        if target_dt is None:
            self.logger.error(f"get_price_at: 无效的 target_time '{target_time}'")
            return None

        try:
            symbol_df = self.stock_df[self.stock_df['symbol'] == symbol_upper]
            if symbol_df.empty:
                return None
            price_data = symbol_df.loc[:target_dt]
            if price_data.empty:
                self.logger.debug(f"在 {target_dt} 或之前未找到 {symbol_upper} 的股票数据")
                return None
            price = float(price_data['close'].iloc[-1])
            actual_time = price_data.index[-1]
            self.logger.debug(
                f"get_price_at: 找到价格 ${price:.2f}，股票: {symbol_upper}，时间: {target_dt} (实际时间: {actual_time})")
            return price
        except KeyError:
            self.logger.debug(f"在 {target_dt} 或之前未找到 {symbol_upper} 的股票数据 (KeyError)")
            return None
        except Exception as e:
            self.logger.error(f"get_price_at 时发生错误 (股票: {symbol}, 时间: {target_dt}): {e}",
                              exc_info=False)
            return None

    def get_prices_at(self, symbols: List[str], target_time: Any) -> Dict[str, Optional[float]]:
        """获取多个股票在指定时间或之前最近的收盘价"""
        prices = {}
        target_dt = self._ensure_utc(target_time)
        if target_dt is None:
            self.logger.error(f"get_prices_at: 无效的 target_time '{target_time}'。返回空字典。")
            return {symbol.upper(): None for symbol in symbols}

        self.logger.debug(f"get_prices_at: 获取 {symbols} 在 {target_dt} 或之前的价格")
        for symbol in symbols:
            prices[symbol.upper()] = self.get_price_at(symbol, target_dt)

        return prices

    def get_latest_price(self, symbol: Optional[str] = None) -> Optional[float]:
        """获取数据中最新时间戳的收盘价"""
        if self.stock_df is None or self.stock_df.empty:
            return None
        symbol_upper = symbol.upper() if symbol else None

        target_df = self.stock_df
        if symbol_upper:
            if symbol_upper not in self.stock_symbols_available:
                return None
            target_df = self.stock_df[self.stock_df['symbol'] == symbol_upper]

        if target_df.empty:
            return None
        try:
            return float(target_df['close'].iloc[-1])
        except IndexError:
            return None
        except Exception as e:
            self.logger.error(f"get_latest_price 时发生错误 (股票: {symbol}): {e}", exc_info=False)
            return None

