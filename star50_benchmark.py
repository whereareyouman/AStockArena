#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAR 50 指数被动跟踪基准脚本
用于模拟被动跟踪 STAR 50 指数（000688.SH）的投资组合，作为主动策略的基准对照组
"""

import os
import sys
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    # 尝试从项目根目录加载 .env 文件
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # 如果项目根目录没有，尝试从当前目录加载
        load_dotenv()
except ImportError:
    # 如果没有安装 python-dotenv，跳过
    pass

from utils.json_file_manager import safe_read_json, safe_write_json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('star50_benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Star50Benchmark")

# 常量配置
INDEX_CODE = "000688.SH"  # STAR 50 指数代码
LOT_SIZE = 100  # 手数：100股

# 从配置文件读取交易规则
def load_trading_rules() -> Dict[str, float]:
    """从 default_config.json 加载交易规则"""
    config_path = os.path.join(project_root, "settings", "default_config.json")
    try:
        config = safe_read_json(config_path, default={})
        trading_rules = config.get("trading_rules", {})
        return {
            "commission_rate": trading_rules.get("commission_rate", 0.0003),
            "min_commission": trading_rules.get("min_commission", 5.0),
            "stamp_duty_rate": trading_rules.get("stamp_duty_rate", 0.0005)
        }
    except Exception as e:
        logger.warning(f"无法加载配置文件，使用默认交易规则: {e}")
        return {
            "commission_rate": 0.0003,
            "min_commission": 5.0,
            "stamp_duty_rate": 0.0005
        }

TRADING_RULES = load_trading_rules()


class Star50Benchmark:
    """
    STAR 50 指数被动跟踪基准类
    实现被动跟踪 STAR 50 指数的投资组合管理
    """
    
    def __init__(self, 
                 init_date: str,
                 end_date: Optional[str] = None,
                 initial_capital: Optional[float] = None,
                 state_file: str = "data_flow/star50_benchmark/benchmark_state.json",
                 data_dir: str = "data_flow/star50_benchmark"):
        """
        初始化基准跟踪器
        
        Args:
            init_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)，如果为None则运行到当前日期
            initial_capital: 初始资金（如果为None，将通过网格搜索自动优化）
            state_file: 状态文件路径
            data_dir: 数据存储目录
        """
        self.init_date = init_date
        self.end_date = end_date
        # 初始资金将在网格搜索优化时自动计算，这里先设置一个占位值
        self.initial_capital = initial_capital if initial_capital is not None else 1000000.0
        self.state_file = state_file
        self.data_dir = data_dir
        
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        # 状态变量
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}  # { 'stock_code': shares }
        self.position_costs: Dict[str, float] = {}  # { 'stock_code': avg_cost_per_share } 记录平均成本价
        self.position_purchase_dates: Dict[str, str] = {}  # { 'stock_code': purchase_date } 记录买入日期
        self.nav_history: List[Dict[str, Any]] = []
        
        # 三个决策时点
        self.decision_times = ["10:30:00", "11:30:00", "14:00:00"]
        
        # position.jsonl 文件路径（用于与 visualize.py 兼容）
        self.position_jsonl_path = os.path.join(
            project_root, 
            "data_flow", 
            "trading_summary_each_agent", 
            "star50-benchmark", 
            "position", 
            "position.jsonl"
        )
        os.makedirs(os.path.dirname(self.position_jsonl_path), exist_ok=True)
        
        # 股票数据和权重数据存储（用于保存到 JSON，格式类似 ai_stock_data.json）
        # 格式: { "SH688111": [{"date": "...", "close": ..., "weight": ...}, ...], ... }
        self.stock_data_dict: Dict[str, List[Dict[str, Any]]] = {}  # 按股票代码组织的数据
        
        # 权重数据（只获取一次）
        self.index_weights_df: Optional[pd.DataFrame] = None
        self.stock_codes_list: List[str] = []  # 成分股代码列表
        
        # 本地股票数据文件路径（用于断点续传）
        self.stock_data_file = os.path.join(self.data_dir, "stock_prices_cache.json")
        
        # 初始化 akshare（用于获取权重）
        self._init_akshare()
        
        # 初始化 TinySoft（用于获取股票价格）
        self.ts_client = None
        self._init_tinysoft()
        
        # 加载历史状态
        self.load_state()
        
        # 获取权重（只获取一次）
        self._load_index_weights()
    
    def _init_akshare(self):
        """初始化 akshare（用于获取指数权重）"""
        try:
            import akshare as ak
            self.ak = ak
            logger.info("AkShare 初始化成功（用于获取权重）")
        except ImportError:
            logger.error("未安装 akshare 库，请运行: pip install akshare")
            raise
        except Exception as e:
            logger.error(f"初始化 AkShare 失败: {e}")
            raise
    
    def _init_tinysoft(self):
        """初始化 TinySoft（用于获取股票价格）"""
        try:
            import pyTSL as ts
            import time
            import os
        except ImportError:
            logger.error("未安装 pyTSL 库，请运行: pip install pyTSL")
            raise
        
        username = os.getenv("TSL_USERNAME") or os.getenv("TSL_USER")
        password = os.getenv("TSL_PASSWORD") or os.getenv("TSL_PASS")
        server = os.getenv("TSL_SERVER", "tsl.tinysoft.com.cn")
        port = int(os.getenv("TSL_PORT", "443"))
        
        if not username or not password:
            raise ValueError("未找到 TinySoft 账号，请设置环境变量 TSL_USERNAME 和 TSL_PASSWORD")
        
        try:
            self.ts_client = ts.Client(username, password, server, port)
            ok = self.ts_client.login()
            if ok != 1:
                raise ValueError(f"TinySoft 登录失败: {self.ts_client.last_error()}")
            logger.info("TinySoft 登录成功（用于获取股票价格）")
        except Exception as e:
            logger.error(f"初始化 TinySoft 失败: {e}")
            raise
    
    def calculate_trade_cost(self, amount: int, price: float, side: str) -> float:
        """
        计算交易成本
        
        Args:
            amount: 交易股数
            price: 交易价格
            side: 'BUY' 或 'SELL'
        
        Returns:
            总交易成本（佣金 + 印花税）
        
        注意：
            - 佣金 = max(交易金额 * 0.0003, 5.0)  # 最低佣金5元
            - 印花税 = 交易金额 * 0.0005（仅在卖出时收取）
        """
        trade_value = amount * price
        
        # 计算佣金
        commission = trade_value * TRADING_RULES["commission_rate"]
        # 应用最低佣金规则
        real_commission = max(TRADING_RULES["min_commission"], commission)
        
        # 计算印花税（仅在卖出时收取）
        stamp_duty = trade_value * TRADING_RULES["stamp_duty_rate"] if side == "SELL" else 0.0
        
        total_cost = real_commission + stamp_duty
        
        return total_cost
    
    def _load_index_weights(self):
        """加载指数权重（只获取一次，存储在类中）"""
        if self.index_weights_df is not None:
            logger.info("权重数据已加载，跳过重复获取")
            return
        
        # 使用第一个交易日获取权重
        init_date_clean = self.init_date.replace('-', '')
        logger.info(f"获取指数权重（仅获取一次，日期: {init_date_clean}）")
        
        try:
            self.index_weights_df = self._fetch_index_weights(init_date_clean)
            if self.index_weights_df is not None and not self.index_weights_df.empty:
                self.stock_codes_list = self.index_weights_df['con_code'].tolist()
                logger.info(f"权重数据已加载: {len(self.stock_codes_list)} 只成分股")
            else:
                raise ValueError("获取到的权重数据为空")
        except Exception as e:
            logger.error(f"获取权重失败: {e}")
            raise
    
    def find_optimal_capital(self, prices: Dict[str, float]) -> float:
        """
        网格搜索优化器：找到最优初始资金以减少现金拖累
        
        Args:
            prices: 股票价格字典
        
        Returns:
            最优初始资金
        """
        if self.index_weights_df is None or self.index_weights_df.empty:
            raise ValueError("权重数据未加载")
        
        logger.info("开始网格搜索优化初始资金...")
        
        # Step 1: 计算基础资金（Floor Capital）
        weights_df = self.index_weights_df.copy()
        base_capitals = []
        
        for _, row in weights_df.iterrows():
            stock_code = row['con_code']
            weight = row['weight'] / 100.0  # 权重转换为小数
            
            if stock_code not in prices or prices[stock_code] <= 0:
                continue
            
            price = prices[stock_code]
            # Base_Capital = Price * 100 / Weight
            base_capital = (price * LOT_SIZE) / weight
            base_capitals.append(base_capital)
        
        if not base_capitals:
            raise ValueError("无法计算基础资金：缺少有效的价格数据")
        
        base_capital = max(base_capitals)
        logger.info(f"基础资金（Floor Capital）: {base_capital:,.2f} 元")
        
        # Step 2: 网格搜索（多阶段优化）
        # 准备数据：权重和价格数组
        valid_stocks = []
        weights_list = []
        prices_list = []
        
        for _, row in weights_df.iterrows():
            stock_code = row['con_code']
            weight = row['weight'] / 100.0
            
            if stock_code in prices and prices[stock_code] > 0:
                valid_stocks.append(stock_code)
                weights_list.append(weight)
                prices_list.append(prices[stock_code])
        
        weights_arr = np.array(weights_list)
        prices_arr = np.array(prices_list)
        
        # 第一阶段：粗搜索（大范围，低精度）
        # 扩大搜索范围以找到更优解（达到<0.5%现金比例）
        search_range_1 = [base_capital, base_capital * 10.0]
        n_candidates_1 = 5000
        candidate_capitals_1 = np.linspace(search_range_1[0], search_range_1[1], n_candidates_1)
        
        best_capital = base_capital
        best_cash_ratio = 1.0
        original_cash_ratio = 1.0
        
        logger.info(f"第一阶段：粗搜索 {n_candidates_1} 个候选值，范围: {search_range_1[0]:,.2f} - {search_range_1[1]:,.2f}")
        
        # 对每个候选资金进行模拟
        for candidate_cap in candidate_capitals_1:
            # 计算目标股数：floor((candidate_cap * weight) / (price * 100)) * 100
            target_values = candidate_cap * weights_arr
            target_shares = np.floor(target_values / (prices_arr * LOT_SIZE)) * LOT_SIZE
            
            # 计算实际投资金额（包括交易成本）
            mask = target_shares > 0
            if np.any(mask):
                cost_basis = target_shares[mask] * prices_arr[mask]
                actual_invested = np.sum(cost_basis)
                for i in np.where(mask)[0]:
                    trade_cost = self.calculate_trade_cost(int(target_shares[i]), prices_arr[i], 'BUY')
                    actual_invested += trade_cost
            else:
                actual_invested = 0.0
            
            # 计算现金比例
            cash_ratio = 1.0 - (actual_invested / candidate_cap) if candidate_cap > 0 else 1.0
            
            # 记录原始资金（第一个候选值）的现金比例
            if candidate_cap == candidate_capitals_1[0]:
                original_cash_ratio = cash_ratio
            
            # 选择现金比例最低的候选值
            if cash_ratio < best_cash_ratio:
                best_cash_ratio = cash_ratio
                best_capital = candidate_cap
        
        logger.info(f"第一阶段最优: {best_capital:,.2f} 元, 现金比例: {best_cash_ratio:.4%}")
        
        # 迭代优化：不断精细搜索直到现金比例低于0.5%
        target_cash_ratio = 0.005  # 0.5%
        iteration = 0
        max_iterations = 20  # 增加最大迭代次数
        
        while best_cash_ratio > target_cash_ratio and iteration < max_iterations:
            iteration += 1
            # 在最优值附近进行精细搜索（扩大搜索范围以提高找到更优解的概率）
            # 根据当前现金比例动态调整搜索范围
            if best_cash_ratio > 0.02:  # 如果现金比例>2%，大幅扩大搜索范围
                search_range = [best_capital * 0.90, best_capital * 1.20]
            elif best_cash_ratio > 0.01:  # 如果现金比例>1%，中等扩大搜索范围
                search_range = [best_capital * 0.95, best_capital * 1.15]
            elif best_cash_ratio > 0.005:  # 如果现金比例>0.5%，小幅扩大搜索范围
                search_range = [best_capital * 0.97, best_capital * 1.10]
            else:
                search_range = [best_capital * 0.98, best_capital * 1.05]
            
            # 根据搜索范围调整候选值数量（范围越大，候选值越多）
            range_size = search_range[1] - search_range[0]
            n_candidates = max(20000, int(range_size / (best_capital * 0.0001)))  # 确保足够的搜索精度
            candidate_capitals = np.linspace(search_range[0], search_range[1], n_candidates)
            
            logger.info(f"迭代 {iteration}：精细搜索 {n_candidates} 个候选值，范围: {search_range[0]:,.2f} - {search_range[1]:,.2f} (当前最优: {best_cash_ratio:.4%})")
            
            improved = False
            for candidate_cap in candidate_capitals:
                target_values = candidate_cap * weights_arr
                target_shares = np.floor(target_values / (prices_arr * LOT_SIZE)) * LOT_SIZE
                
                mask = target_shares > 0
                if np.any(mask):
                    cost_basis = target_shares[mask] * prices_arr[mask]
                    actual_invested = np.sum(cost_basis)
                    for i in np.where(mask)[0]:
                        trade_cost = self.calculate_trade_cost(int(target_shares[i]), prices_arr[i], 'BUY')
                        actual_invested += trade_cost
                else:
                    actual_invested = 0.0
                
                cash_ratio = 1.0 - (actual_invested / candidate_cap) if candidate_cap > 0 else 1.0
                
                if cash_ratio < best_cash_ratio:
                    best_cash_ratio = cash_ratio
                    best_capital = candidate_cap
                    improved = True
            
            if improved:
                logger.info(f"迭代 {iteration} 最优: {best_capital:,.2f} 元, 现金比例: {best_cash_ratio:.4%}")
            else:
                logger.info(f"迭代 {iteration} 未找到更优解，当前最优: {best_capital:,.2f} 元, 现金比例: {best_cash_ratio:.4%}")
                # 即使未找到更优解，也继续尝试（可能需要在更大范围内搜索）
                # 但如果连续多次未改进，可以尝试更大的跳跃
                if iteration >= 3 and not improved:
                    # 尝试在更大的范围内搜索
                    logger.info(f"  尝试扩大搜索范围...")
                    extended_range = [best_capital * 0.85, best_capital * 1.30]
                    n_extended = 30000
                    extended_candidates = np.linspace(extended_range[0], extended_range[1], n_extended)
                    logger.info(f"  扩展搜索: {n_extended} 个候选值，范围: {extended_range[0]:,.2f} - {extended_range[1]:,.2f}")
                    
                    for candidate_cap in extended_candidates:
                        target_values = candidate_cap * weights_arr
                        target_shares = np.floor(target_values / (prices_arr * LOT_SIZE)) * LOT_SIZE
                        
                        mask = target_shares > 0
                        if np.any(mask):
                            cost_basis = target_shares[mask] * prices_arr[mask]
                            actual_invested = np.sum(cost_basis)
                            for i in np.where(mask)[0]:
                                trade_cost = self.calculate_trade_cost(int(target_shares[i]), prices_arr[i], 'BUY')
                                actual_invested += trade_cost
                        else:
                            actual_invested = 0.0
                        
                        cash_ratio = 1.0 - (actual_invested / candidate_cap) if candidate_cap > 0 else 1.0
                        
                        if cash_ratio < best_cash_ratio:
                            best_cash_ratio = cash_ratio
                            best_capital = candidate_cap
                            improved = True
                            logger.info(f"  扩展搜索找到更优解: {best_capital:,.2f} 元, 现金比例: {best_cash_ratio:.4%}")
                    
                    if not improved:
                        logger.info(f"  扩展搜索也未找到更优解，可能已达到理论最优")
                        break
            
            # 如果已经达到目标，停止迭代
            if best_cash_ratio <= target_cash_ratio:
                logger.info(f"✓ 已达到目标现金比例 ({target_cash_ratio:.2%})，停止迭代")
                break
        
        # Step 3: 输出结果
        reduction = ((original_cash_ratio - best_cash_ratio) / original_cash_ratio * 100) if original_cash_ratio > 0 else 0
        
        logger.info("=" * 60)
        logger.info("网格搜索优化结果:")
        logger.info(f"  基础资金（Floor Capital）: {base_capital:,.2f} 元")
        logger.info(f"  最优资金（Optimal Capital）: {best_capital:,.2f} 元")
        logger.info(f"  预测现金拖累（Predicted Cash Drag）: {best_cash_ratio:.2%}")
        logger.info(f"  原始现金拖累（Original Cash Drag）: {original_cash_ratio:.2%}")
        logger.info(f"  现金拖累减少（Reduction）: {reduction:.2f}%")
        logger.info("=" * 60)
        
        return best_capital
    
    def _calculate_initial_positions(self, prices: Dict[str, float]):
        """计算初始持仓（基于权重和价格）"""
        if self.index_weights_df is None or self.index_weights_df.empty:
            raise ValueError("权重数据未加载")
        
        logger.info("计算初始持仓（基于权重和价格）")
        
        # 使用优化后的资金
        total_assets = self.initial_capital
        weights_df = self.index_weights_df.copy()
        
        for _, row in weights_df.iterrows():
            stock_code = row['con_code']
            weight = row['weight'] / 100.0  # 权重转换为小数
            
            if stock_code not in prices:
                logger.warning(f"股票 {stock_code} 无价格数据，跳过")
                continue
            
            price = prices[stock_code]
            if price <= 0:
                logger.warning(f"股票 {stock_code} 价格无效，跳过")
                continue
            
            # 计算目标市值
            target_value = total_assets * weight
            
            # 计算目标股数（优化现金使用）
            # 计算买入成本（包括交易费用）
            min_cost_per_lot = price * LOT_SIZE + self.calculate_trade_cost(LOT_SIZE, price, 'BUY')
            
            if target_value < min_cost_per_lot:
                # 目标市值不足以买入100股，保持为0（现金拖累）
                target_shares = 0
                logger.debug(f"股票 {stock_code}: 目标市值 {target_value:.2f} < 最小成本 {min_cost_per_lot:.2f}，目标股数=0（现金拖累）")
            else:
                # 计算可买入的最大手数（考虑交易成本）
                max_affordable_shares = math.floor((target_value - self.calculate_trade_cost(LOT_SIZE, price, 'BUY')) / (price * LOT_SIZE)) * LOT_SIZE
                target_shares = max_affordable_shares
            
            # 执行买入（扣除手续费）
            if target_shares > 0:
                cost_basis = target_shares * price
                trade_cost = self.calculate_trade_cost(target_shares, price, 'BUY')
                total_required = cost_basis + trade_cost
                
                if self.cash >= total_required:
                    self.positions[stock_code] = target_shares
                    # 记录买入价格（close_price，不包含交易成本，与原有系统一致）
                    self.position_costs[stock_code] = price
                    self.position_purchase_dates[stock_code] = self.init_date
                    self.cash -= total_required
                    logger.debug(f"初始买入: {stock_code} {target_shares}股 @ {price:.2f}, 成本={cost_basis:.2f}, 交易费用={trade_cost:.2f}")
                else:
                    # 现金不足，调整买入数量
                    available_cash = self.cash - trade_cost
                    if available_cash >= price * LOT_SIZE:
                        max_shares = math.floor(available_cash / (price * LOT_SIZE)) * LOT_SIZE
                        if max_shares > 0:
                            cost_basis = max_shares * price
                            trade_cost = self.calculate_trade_cost(max_shares, price, 'BUY')
                            total_required = cost_basis + trade_cost
                            self.positions[stock_code] = max_shares
                            # 记录买入价格（close_price，不包含交易成本，与原有系统一致）
                            self.position_costs[stock_code] = price
                            self.position_purchase_dates[stock_code] = self.init_date
                            self.cash -= total_required
                            logger.debug(f"初始买入（调整）: {stock_code} {max_shares}股 @ {price:.2f}, 成本={cost_basis:.2f}, 交易费用={trade_cost:.2f}")
        
        # 计算最终资产
        total_assets, stock_value, cash = self.calculate_nav(prices)
        cash_ratio = cash / total_assets if total_assets > 0 else 0.0
        
        logger.info(f"初始持仓计算完成: 总资产={total_assets:.2f}, 股票市值={stock_value:.2f}, 现金={cash:.2f}, 现金比例={cash_ratio:.4f}, 持仓数={len([p for p in self.positions.values() if p > 0])}")
    
    def _load_stock_prices_from_cache(self, trade_date: str, target_time: str = None) -> Optional[Dict[str, float]]:
        """
        从本地缓存文件加载股票价格
        
        Args:
            trade_date: 交易日期 (YYYY-MM-DD)
            target_time: 目标时间点 (HH:MM:SS)
        
        Returns:
            价格字典，如果不存在则返回 None
        """
        if not os.path.exists(self.stock_data_file):
            return None
        
        try:
            cache_data = safe_read_json(self.stock_data_file, default={})
            decision_time = f"{trade_date} {target_time}" if target_time else f"{trade_date} 15:00:00"
            
            prices = {}
            for stock_code, data_list in cache_data.items():
                for data_point in data_list:
                    if data_point.get('date') == decision_time:
                        prices[stock_code] = data_point.get('close')
                        break
            
            return prices if prices else None
        except Exception as e:
            logger.debug(f"从缓存加载价格失败: {e}")
            return None
    
    def _save_stock_prices_to_cache(self, trade_date: str, target_time: str, prices: Dict[str, float]):
        """
        保存股票价格到本地缓存文件
        
        Args:
            trade_date: 交易日期 (YYYY-MM-DD)
            target_time: 目标时间点 (HH:MM:SS)
            prices: 价格字典
        """
        decision_time = f"{trade_date} {target_time}"
        
        # 加载现有缓存
        cache_data = {}
        if os.path.exists(self.stock_data_file):
            cache_data = safe_read_json(self.stock_data_file, default={})
        
        # 更新缓存
        for stock_code, price in prices.items():
            if stock_code not in cache_data:
                cache_data[stock_code] = []
            
            # 检查是否已存在该时间点的数据
            exists = False
            for data_point in cache_data[stock_code]:
                if data_point.get('date') == decision_time:
                    data_point['close'] = price
                    exists = True
                    break
            
            if not exists:
                cache_data[stock_code].append({
                    'date': decision_time,
                    'close': price
                })
        
        # 保存缓存
        safe_write_json(self.stock_data_file, cache_data, backup=False)
    
    def download_all_stock_data(self, trading_days: List[str], last_trade_date: Optional[str] = None, last_time: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        批量下载所有交易日的股票数据（支持断点续传）
        
        Args:
            trading_days: 交易日列表 (YYYY-MM-DD 格式)
        
        Returns:
            { 'trade_date': { 'stock_code': price } } 字典
        """
        logger.info(f"开始批量下载股票数据（共 {len(trading_days)} 个交易日，{len(self.stock_codes_list)} 只股票）")
        
        # 加载已有缓存
        cache_data = {}
        if os.path.exists(self.stock_data_file):
            cache_data = safe_read_json(self.stock_data_file, default={})
            logger.info(f"已加载缓存数据，包含 {len(cache_data)} 只股票的历史数据")
        
        all_prices = {}
        downloaded_count = 0
        skipped_count = 0
        
        for i, trade_date in enumerate(trading_days, 1):
            trade_date_clean = trade_date.replace('-', '')
            logger.info(f"下载进度 {i}/{len(trading_days)}: {trade_date}")
            
            # 确定该交易日需要下载的时点列表
            if last_trade_date and trade_date == last_trade_date:
                time_points = self.decision_times[:self.decision_times.index(last_time) + 1]
            else:
                time_points = self.decision_times
            
            # 检查该交易日是否已有数据（所有需要的时点）
            has_all_data = True
            for time_str in time_points:
                cached_prices = self._load_stock_prices_from_cache(trade_date, time_str)
                if cached_prices is None or len(cached_prices) < len(self.stock_codes_list):
                    has_all_data = False
                    break
            
            if has_all_data:
                logger.info(f"  跳过 {trade_date}（数据已存在）")
                skipped_count += 1
                # 从缓存加载数据
                for time_str in time_points:
                    if trade_date not in all_prices:
                        all_prices[trade_date] = {}
                    cached_prices = self._load_stock_prices_from_cache(trade_date, time_str)
                    if cached_prices:
                        all_prices[trade_date][time_str] = cached_prices
                continue
            
            # 下载该交易日的时点数据
            for time_str in time_points:
                try:
                    prices = self.get_stock_prices(self.stock_codes_list, trade_date_clean, time_str)
                    if prices:
                        # 保存到缓存
                        self._save_stock_prices_to_cache(trade_date, time_str, prices)
                        if trade_date not in all_prices:
                            all_prices[trade_date] = {}
                        all_prices[trade_date][time_str] = prices
                        downloaded_count += 1
                        logger.debug(f"  已下载 {trade_date} {time_str}: {len(prices)} 只股票")
                except Exception as e:
                    logger.error(f"  下载 {trade_date} {time_str} 失败: {e}")
                    # 继续下载其他时点
                    continue
            
            # 每下载10个交易日保存一次缓存
            if downloaded_count % 10 == 0:
                logger.info(f"  已下载 {downloaded_count} 个时点数据，缓存已保存")
        
        logger.info(f"数据下载完成: 新下载 {downloaded_count} 个时点，跳过 {skipped_count} 个交易日")
        return all_prices
    
    def _fetch_index_weights(self, trade_date: str) -> pd.DataFrame:
        """
        获取指定日期的指数成分股权重（使用 akshare）
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
        
        Returns:
            DataFrame with columns: ['index_code', 'con_code', 'weight', 'trade_date']
        
        Raises:
            ValueError: 如果无法获取权重数据或数据格式不正确
        """
        # 使用 akshare 获取 STAR 50 指数成分股权重
        # 指数代码: 000688 (科创50，不需要 sh 前缀)
        df_ak = self.ak.index_stock_cons_weight_csindex(symbol="000688")
        
        if df_ak is None or df_ak.empty:
            raise ValueError(f"未获取到指数成分股权重数据 (日期: {trade_date})")
        
        # akshare 返回的格式：['日期', '指数代码', '指数名称', ..., '成分券代码', '成分券名称', '权重']
        # 检查必需的列是否存在
        if '成分券代码' not in df_ak.columns:
            raise ValueError(f"返回数据中缺少'成分券代码'列 (日期: {trade_date})")
        
        if '权重' not in df_ak.columns:
            raise ValueError(f"返回数据中缺少'权重'列 (日期: {trade_date})")
        
        # 提取数据
        df_ak['con_code'] = df_ak['成分券代码']
        df_ak['weight'] = df_ak['权重']
        
        # 验证权重数据
        if df_ak['weight'].isna().any():
            raise ValueError(f"权重数据包含空值 (日期: {trade_date})")
        
        if (df_ak['weight'] <= 0).any():
            raise ValueError(f"权重数据包含非正值 (日期: {trade_date})")
        
        # 标准化股票代码格式（转换为 SH688XXX 格式）
        def normalize_stock_code(code):
            """将股票代码转换为 SH688XXX 格式"""
            code_str = str(code)
            # 如果已经是 SH688XXX 格式，直接返回
            if code_str.startswith('SH') or code_str.startswith('SZ'):
                return code_str
            # 如果包含 .SH 或 .SZ，转换为 SH/SZ 前缀格式
            if '.' in code_str:
                parts = code_str.split('.')
                if len(parts) == 2:
                    market = parts[1].upper()
                    code_num = parts[0]
                    return f"{market}{code_num}"
            # 如果是纯数字（688开头），添加 SH 前缀
            if code_str.startswith('688'):
                return f"SH{code_str}"
            return code_str
        
        df_ak['con_code'] = df_ak['con_code'].apply(normalize_stock_code)
        
        # 添加其他必要列
        df_ak['index_code'] = INDEX_CODE
        df_ak['trade_date'] = trade_date
        
        # 选择需要的列
        result_df = df_ak[['index_code', 'con_code', 'weight', 'trade_date']].copy()
        
        # 验证结果
        if result_df.empty:
            raise ValueError(f"处理后的权重数据为空 (日期: {trade_date})")
        
        logger.info(f"获取到 {len(result_df)} 只成分股的真实权重数据 (日期: {trade_date})")
        return result_df
    
    def get_stock_prices(self, stock_codes: List[str], trade_date: str, target_time: str = None) -> Dict[str, float]:
        """
        获取指定日期和时间的股票价格（使用 TinySoft）
        
        Args:
            stock_codes: 股票代码列表（如 ['688008.SH', '688111.SH']）
            trade_date: 交易日期 (YYYYMMDD)
            target_time: 目标时间点 (HH:MM:SS)，如果为 None 则使用收盘价
        
        Returns:
            { 'stock_code': price } 字典
        
        Raises:
            ValueError: 如果无法获取价格数据
        """
        if self.ts_client is None:
            raise ValueError("TinySoft 客户端未初始化")
        
        prices = {}
        import time
        from datetime import datetime, timedelta
        
        # 标准化日期格式
        trade_date_formatted = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        
        # 构建时间范围（TinySoft 需要 begin_time 和 end_time）
        if target_time:
            # 解析目标时间
            time_parts = target_time.split(':')
            target_hour = int(time_parts[0])
            target_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            target_second = int(time_parts[2]) if len(time_parts) > 2 else 0
            
            # 构建目标时间点
            target_dt = datetime.strptime(trade_date_formatted, "%Y-%m-%d").replace(
                hour=target_hour, minute=target_minute, second=target_second
            )
            
            # 查询范围：从当天开盘到目标时间
            begin_time = target_dt.replace(hour=9, minute=30, second=0)
            end_time = target_dt
        else:
            # 使用收盘价，查询当天所有数据
            begin_time = datetime.strptime(trade_date_formatted, "%Y-%m-%d").replace(hour=9, minute=30, second=0)
            end_time = datetime.strptime(trade_date_formatted, "%Y-%m-%d").replace(hour=15, minute=0, second=0)
        
        # 逐个查询股票价格
        for code in stock_codes:
            try:
                # 股票代码已经是 SH688XXX 格式，直接使用
                # TinySoft 需要 SH/SZ 前缀格式，所以直接使用
                stock_symbol = code
                
                # 使用 TinySoft 获取小时线数据（60分钟线）
                r = self.ts_client.query(
                    stock=stock_symbol,
                    begin_time=begin_time,
                    end_time=end_time,
                    cycle='60分钟线',
                    fields='date, close'
                )
                
                if r.error() != 0:
                    raise ValueError(f"TinySoft 查询失败: {r.message()} (股票: {code})")
                
                df = r.dataframe()
                if df is None or df.empty:
                    raise ValueError(f"未获取到股票 {code} 的数据 (日期: {trade_date})")
                
                # 如果指定了时间点，查找最接近的数据点
                price = None
                if target_time:
                    # 查找 <= target_time 的最新数据点
                    for idx, row in df.iterrows():
                        row_time = row.get('date')
                        if row_time is None:
                            continue
                        
                        # 转换为 datetime
                        if isinstance(row_time, str):
                            try:
                                row_dt = datetime.strptime(row_time[:19], "%Y-%m-%d %H:%M:%S")
                            except:
                                continue
                        elif hasattr(row_time, 'to_pydatetime'):
                            row_dt = row_time.to_pydatetime()
                        else:
                            continue
                        
                        if row_dt <= target_dt:
                            close_price = row.get('close')
                            if close_price is not None:
                                price = float(close_price)
                                # 继续查找更接近的数据点（倒序遍历）
                                # 但这里我们已经正序遍历，找到最后一个 <= target_dt 的
                                # 为了更准确，应该倒序遍历
                
                # 如果没找到指定时间点的数据，使用最后一条数据（收盘价）
                if price is None:
                    last_row = df.iloc[-1]
                    close_price = last_row.get('close')
                    if close_price is None:
                        raise ValueError(f"股票 {code} 数据中无收盘价 (日期: {trade_date})")
                    price = float(close_price)
                
                if price is None or price <= 0:
                    raise ValueError(f"股票 {code} 价格无效: {price} (日期: {trade_date})")
                
                prices[code] = price
                
                # 避免请求过于频繁
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"获取股票 {code} 价格失败: {e}")
                raise ValueError(f"获取股票 {code} 价格失败: {e}")
        
        if not prices:
            raise ValueError(f"未能获取任何股票价格 (日期: {trade_date})")
        
        logger.info(f"成功获取 {len(prices)} 只股票的价格 (日期: {trade_date}, 时间: {target_time or '收盘价'})")
        return prices
    
    def calculate_nav(self, prices: Dict[str, float]) -> Tuple[float, float, float]:
        """计算当前资产净值"""
        stock_value = 0.0
        for code, shares in self.positions.items():
            if code in prices and shares > 0:
                stock_value += shares * prices[code]
        
        total_assets = self.cash + stock_value
        return total_assets, stock_value, self.cash
    
    def save_position_jsonl(self, trade_date: str, decision_time: str, decision_count: int, 
                           prices: Dict[str, float], action: Dict[str, Any] = None):
        """保存持仓记录到 position.jsonl 文件"""
        # 构建 positions 字典（与系统格式一致）
        positions_dict = {"CASH": round(self.cash, 2)}
        
        # 添加股票持仓（格式：{code: {"shares": int, "avg_price": float, "purchase_date": str}}）
        # 使用实际记录的平均成本价
        for code, shares in self.positions.items():
            if shares > 0:
                # 使用记录的平均成本价，如果没有则使用当前价格（兼容旧数据）
                avg_price = self.position_costs.get(code, prices.get(code, 0.0) if code in prices else 0.0)
                purchase_date = self.position_purchase_dates.get(code, trade_date)
                positions_dict[code] = {
                    "shares": shares,
                    "avg_price": round(avg_price, 2),
                    "purchase_date": purchase_date
                }
        
        # 构建记录
        record = {
            "id": len(self.nav_history),  # 使用 nav_history 长度作为 id
            "date": trade_date,
            "decision_time": decision_time,
            "decision_count": decision_count,
            "positions": positions_dict,
            "this_action": action or {"action": "rebalance", "symbol": "", "amount": 0}
        }
        
        # 追加写入 JSONL 文件
        try:
            with open(self.position_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存 position.jsonl 失败: {e}")
    
    def calculate_unrealized_pnl(self, trade_date: str, target_time: str, prices: Dict[str, float]) -> None:
        """计算未实现盈亏（不执行交易，只更新NAV和记录）"""
        # 计算当前 NAV
        total_assets, stock_value, cash = self.calculate_nav(prices)
        cash_ratio = cash / total_assets if total_assets > 0 else 0.0
        pnl_pct = ((total_assets / self.initial_capital) - 1.0) * 100.0
        
        # 记录 NAV
        trade_date_clean = trade_date.replace('-', '')
        nav_record = {
            'date': trade_date_clean,
            'total_assets': round(total_assets, 2),
            'stock_value': round(stock_value, 2),
            'cash': round(cash, 2),
            'cash_ratio': round(cash_ratio, 4),
            'pnl_pct': round(pnl_pct, 4),
            'positions_count': len([p for p in self.positions.values() if p > 0])
        }
        
        self.nav_history.append(nav_record)
        
        # 保存到 position.jsonl
        decision_time = f"{trade_date} {target_time}"
        decision_count = self.decision_times.index(target_time) + 1
        self.save_position_jsonl(
            trade_date,
            decision_time,
            decision_count,
            prices,
            {"action": "update", "symbol": "", "amount": 0}
        )
        
        # 保存股票价格和权重数据到历史记录
        self._save_stock_data_to_history(trade_date, target_time, self.index_weights_df, prices)
    
    def _save_stock_data_to_history(self, trade_date: str, time_str: str, 
                                    weights_df: pd.DataFrame, prices: Dict[str, float]):
        """保存股票价格和权重数据到历史记录"""
        decision_time = f"{trade_date} {time_str}"
        
        # 按股票代码组织数据（格式类似 ai_stock_data.json）
        for _, row in weights_df.iterrows():
            stock_code = row['con_code']
            weight = float(row['weight'])  # 权重（百分比）
            
            if stock_code in prices:
                # 股票代码已经是 SH688XXX 格式，直接使用
                if stock_code not in self.stock_data_dict:
                    self.stock_data_dict[stock_code] = []
                
                self.stock_data_dict[stock_code].append({
                    'date': decision_time,
                    'close': round(prices[stock_code], 2),
                    'weight': round(weight, 4)
                })
    
    def load_state(self):
        """从文件加载历史状态"""
        if not os.path.exists(self.state_file):
            return
        try:
            state = safe_read_json(self.state_file, default={})
            # 如果从状态文件加载，使用保存的现金值；否则使用初始资金
            saved_cash = state.get('cash')
            if saved_cash is not None:
                self.cash = saved_cash
            else:
                self.cash = self.initial_capital
            self.positions = state.get('positions', {})
            self.position_costs = state.get('position_costs', {})
            self.position_purchase_dates = state.get('position_purchase_dates', {})
            self.nav_history = state.get('nav_history', [])
            logger.info(f"加载状态: 现金={self.cash:.2f}, 持仓数={len(self.positions)}")
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
    
    def save_state(self):
        """保存状态到文件"""
        try:
            safe_write_json(self.state_file, {
                'cash': self.cash,
                'positions': self.positions,
                'position_costs': self.position_costs,
                'position_purchase_dates': self.position_purchase_dates,
                'nav_history': self.nav_history,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, backup=True)
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
    
    def _get_latest_decision_point(self) -> Tuple[Optional[str], Optional[str]]:
        """
        自动识别截止到当前时间点前的那个最新的决策时点
        
        Returns:
            (trade_date, time_str) 元组，如果当前不是交易日或还没到第一个时点，返回 (None, None)
        """
        now = datetime.now()
        current_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')
        
        # 检查是否是交易日（简单检查：周一到周五）
        if now.weekday() >= 5:  # 周末
            # 返回前一个交易日的最后一个时点
            last_trade_date = (now - timedelta(days=1 if now.weekday() == 5 else 2)).strftime('%Y-%m-%d')
            return last_trade_date, self.decision_times[-1]  # 14:00:00
        
        # 检查当前时间对应的决策时点
        # 如果还没到第一个时点（10:30），返回前一个交易日的最后一个时点
        if current_time < self.decision_times[0]:
            # 返回前一个交易日的最后一个时点
            last_trade_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            # 确保不是周末
            while datetime.strptime(last_trade_date, '%Y-%m-%d').weekday() >= 5:
                last_trade_date = (datetime.strptime(last_trade_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            return last_trade_date, self.decision_times[-1]  # 14:00:00
        
        # 找到当前时间对应的最新决策时点
        latest_time = None
        for time_str in reversed(self.decision_times):
            if current_time >= time_str:
                latest_time = time_str
                break
        
        if latest_time:
            return current_date, latest_time
        
        # 如果都不满足，返回前一个交易日的最后一个时点
        last_trade_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        while datetime.strptime(last_trade_date, '%Y-%m-%d').weekday() >= 5:
            last_trade_date = (datetime.strptime(last_trade_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        return last_trade_date, self.decision_times[-1]  # 14:00:00
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历"""
        try:
            df = self.ak.tool_trade_date_hist_sina()
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                start_dt = datetime.strptime(start_date, '%Y%m%d')
                end_dt = datetime.strptime(end_date, '%Y%m%d')
                mask = (df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)
                trading_days = [d.strftime('%Y-%m-%d') for d in df[mask]['trade_date'].tolist()]
                logger.info(f"获取到 {len(trading_days)} 个交易日")
                return trading_days
        except Exception as e:
            logger.debug(f"获取交易日历失败: {e}，使用简化逻辑")
        
        # 回退：排除周末
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        trading_days = [d.strftime('%Y-%m-%d') for d in pd.date_range(start_dt, end_dt, freq='B')]
        logger.info(f"生成 {len(trading_days)} 个交易日（排除周末）")
        return trading_days
    
    def run(self):
        """运行基准跟踪（从 init_date 到 end_date）"""
        logger.info(f"开始运行 STAR 50 基准跟踪")
        logger.info(f"日期范围: {self.init_date} 至 {self.end_date or '当前日期'}")
        logger.info(f"初始资金（占位值，将在步骤2中自动优化）: {self.initial_capital:.2f}")
        
        # 清空 position.jsonl 文件（重新生成）
        if os.path.exists(self.position_jsonl_path):
            # 备份旧文件
            backup_path = self.position_jsonl_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import shutil
                shutil.copy2(self.position_jsonl_path, backup_path)
                logger.info(f"已备份旧 position.jsonl 到: {backup_path}")
            except Exception as e:
                logger.warning(f"备份旧文件失败: {e}")
            
            # 删除旧文件，确保重新生成
            try:
                os.remove(self.position_jsonl_path)
                logger.info(f"已删除旧 position.jsonl，将重新生成")
            except Exception as e:
                logger.warning(f"删除旧文件失败: {e}")
        
        # 同时清空状态文件，确保从头开始
        if os.path.exists(self.state_file):
            backup_state_path = self.state_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import shutil
                shutil.copy2(self.state_file, backup_state_path)
                logger.info(f"已备份旧状态文件到: {backup_state_path}")
                os.remove(self.state_file)
                logger.info(f"已删除旧状态文件，将重新生成")
            except Exception as e:
                logger.warning(f"备份/删除状态文件失败: {e}")
        
        # 重置状态变量，确保从头开始
        self.cash = self.initial_capital
        self.positions = {}
        self.position_costs = {}
        self.position_purchase_dates = {}
        self.nav_history = []
        
        # 写入初始 seed 记录
        seed_record = {
            "id": 0,
            "date": self.init_date,
            "decision_time": f"{self.init_date} 00:00:00",
            "decision_count": 0,
            "seed": True,
            "positions": {"CASH": self.initial_capital},
            "this_action": {"action": "seed", "symbol": "", "amount": 0}
        }
        with open(self.position_jsonl_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(seed_record, ensure_ascii=False) + '\n')
        
        # 转换日期格式
        start_date_clean = self.init_date.replace('-', '')
        last_trade_date = None
        last_time = None
        
        if self.end_date:
            end_date_clean = self.end_date.replace('-', '')
            end_date_dt = datetime.strptime(end_date_clean, '%Y%m%d')
            today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # 如果 end_date 是今天或未来，自动识别当前时间点
            if end_date_dt >= today_dt:
                logger.info(f"end_date ({self.end_date}) 是今天或未来，自动识别最新决策时点")
                last_trade_date, last_time = self._get_latest_decision_point()
                if last_trade_date:
                    logger.info(f"自动识别最新决策时点: {last_trade_date} {last_time}")
                    # 如果识别出的日期早于 end_date，使用识别出的日期
                    last_date_dt = datetime.strptime(last_trade_date, '%Y-%m-%d')
                    if last_date_dt < end_date_dt:
                        end_date_clean = last_trade_date.replace('-', '')
                        logger.info(f"调整 end_date 为: {end_date_clean}")
        else:
            # 自动识别截止到当前时间点前的那个最新的时间点
            end_date_clean = datetime.now().strftime('%Y%m%d')
            last_trade_date, last_time = self._get_latest_decision_point()
            if last_trade_date:
                logger.info(f"自动识别最新决策时点: {last_trade_date} {last_time}")
                # 如果最新时点不是最后一个交易日，调整交易日历
                end_date_clean = last_trade_date.replace('-', '')
        
        # 获取交易日历
        trading_days = self.get_trading_calendar(start_date_clean, end_date_clean)
        
        # 如果自动识别了最新时点，需要调整交易日列表
        if last_trade_date and last_trade_date in trading_days:
            # 找到最后一个交易日的位置
            last_index = trading_days.index(last_trade_date)
            trading_days = trading_days[:last_index + 1]
        
        logger.info(f"共 {len(trading_days)} 个交易日需要处理")
        
        # 步骤1: 批量下载所有股票数据（支持断点续传）
        logger.info("=" * 60)
        logger.info("步骤1: 批量下载股票数据")
        logger.info("=" * 60)
        all_prices_data = self.download_all_stock_data(trading_days, last_trade_date, last_time)
        
        # 步骤2: 网格搜索优化初始资金（使用第一个交易日10:30的价格）
        logger.info("=" * 60)
        logger.info("步骤2: 网格搜索优化初始资金")
        logger.info("=" * 60)
        first_trade_date = trading_days[0]
        first_time = self.decision_times[0]  # 10:30:00
        
        logger.info(f"使用第一个交易日 {first_trade_date} {first_time} 的价格进行优化计算")
        
        # 从缓存或已下载数据获取价格
        if first_trade_date in all_prices_data and first_time in all_prices_data[first_trade_date]:
            prices = all_prices_data[first_trade_date][first_time]
            logger.info(f"从缓存加载价格数据: {len(prices)} 只股票")
        else:
            # 如果缓存中没有，实时获取
            first_trade_date_clean = first_trade_date.replace('-', '')
            prices = self.get_stock_prices(self.stock_codes_list, first_trade_date_clean, first_time)
            logger.info(f"实时获取价格数据: {len(prices)} 只股票")
        
        # 网格搜索找到最优初始资金（使用10:30的价格）
        optimal_capital = self.find_optimal_capital(prices)
        
        # 更新初始资金和现金
        original_capital = self.initial_capital
        self.initial_capital = optimal_capital
        self.cash = optimal_capital
        
        logger.info(f"✓ 初始资金已优化: {original_capital:,.2f} -> {optimal_capital:,.2f} 元")
        logger.info(f"✓ 当前初始资金: {self.initial_capital:,.2f} 元")
        
        # 步骤3: 计算初始持仓（基于优化后的资金，使用第一天10:30的价格）
        logger.info("=" * 60)
        logger.info("步骤3: 计算初始持仓（基于优化后的资金，使用第一天10:30的价格）")
        logger.info("=" * 60)
        self._calculate_initial_positions(prices)
        
        # 记录初始持仓到 position.jsonl（10:30时点）
        decision_time = f"{first_trade_date} {first_time}"
        self.save_position_jsonl(
            first_trade_date,
            decision_time,
            1,  # decision_count = 1
            prices,
            {"action": "initial_position", "symbol": "", "amount": 0}
        )
        
        # 保存股票价格和权重数据到历史记录（10:30时点）
        self._save_stock_data_to_history(first_trade_date, first_time, self.index_weights_df, prices)
        
        # 记录初始 NAV（10:30时点）
        total_assets, stock_value, cash = self.calculate_nav(prices)
        cash_ratio = cash / total_assets if total_assets > 0 else 0.0
        pnl_pct = ((total_assets / self.initial_capital) - 1.0) * 100.0
        
        nav_record = {
            'date': first_trade_date.replace('-', ''),
            'total_assets': round(total_assets, 2),
            'stock_value': round(stock_value, 2),
            'cash': round(cash, 2),
            'cash_ratio': round(cash_ratio, 4),
            'pnl_pct': round(pnl_pct, 4),
            'positions_count': len([p for p in self.positions.values() if p > 0])
        }
        self.nav_history.append(nav_record)
        
        # 步骤3.5: 处理第一天剩余时点（11:30, 14:00）
        logger.info("=" * 60)
        logger.info("步骤3.5: 处理第一天剩余时点（11:30, 14:00）")
        logger.info("=" * 60)
        
        # 处理第一天的其他时点（从第二个时点开始）
        for decision_count, time_str in enumerate(self.decision_times[1:], 2):  # 从11:30开始
            decision_time = f"{first_trade_date} {time_str}"
            logger.debug(f"  决策时点 {decision_count}/{len(self.decision_times)}: {decision_time}")
            
            try:
                # 从缓存或已下载数据获取价格
                if first_trade_date in all_prices_data and time_str in all_prices_data[first_trade_date]:
                    prices = all_prices_data[first_trade_date][time_str]
                else:
                    # 如果缓存中没有，尝试从本地缓存文件加载
                    prices = self._load_stock_prices_from_cache(first_trade_date, time_str)
                    if prices is None:
                        # 如果本地也没有，实时获取
                        first_trade_date_clean = first_trade_date.replace('-', '')
                        prices = self.get_stock_prices(self.stock_codes_list, first_trade_date_clean, time_str)
                        # 保存到缓存
                        self._save_stock_prices_to_cache(first_trade_date, time_str, prices)
                
                if not prices:
                    raise ValueError(f"无法获取股票价格数据 (日期: {first_trade_date})")
                
                # 计算未实现盈亏（不执行交易）
                self.calculate_unrealized_pnl(first_trade_date, time_str, prices)
                
            except Exception as e:
                logger.error(f"处理 {decision_time} 失败: {e}")
                raise  # 重新抛出异常，停止执行
        
        # 步骤4: 后续交易日只计算unrealized pnl
        logger.info("=" * 60)
        logger.info("步骤4: 计算未实现盈亏（后续交易日）")
        logger.info("=" * 60)
        
        for i, trade_date in enumerate(trading_days[1:], 2):  # 从第二个交易日开始
            logger.info(f"处理交易日 {i}/{len(trading_days)}: {trade_date}")
            
            # 确定该交易日需要处理的时点列表
            if last_trade_date and trade_date == last_trade_date:
                # 最后一个交易日，只处理到 last_time
                time_points = self.decision_times[:self.decision_times.index(last_time) + 1]
                logger.info(f"  最后一个交易日，只处理到 {last_time}")
            else:
                # 其他交易日，处理所有时点
                time_points = self.decision_times
            
            # 每天在决策时点记录状态
            for decision_count, time_str in enumerate(time_points, 1):
                decision_time = f"{trade_date} {time_str}"
                logger.debug(f"  决策时点 {decision_count}/{len(time_points)}: {decision_time}")
                
                try:
                    # 从缓存或已下载数据获取价格
                    if trade_date in all_prices_data and time_str in all_prices_data[trade_date]:
                        prices = all_prices_data[trade_date][time_str]
                    else:
                        # 如果缓存中没有，尝试从本地缓存文件加载
                        prices = self._load_stock_prices_from_cache(trade_date, time_str)
                        if prices is None:
                            # 如果本地也没有，实时获取
                            trade_date_clean = trade_date.replace('-', '')
                            prices = self.get_stock_prices(self.stock_codes_list, trade_date_clean, time_str)
                            # 保存到缓存
                            self._save_stock_prices_to_cache(trade_date, time_str, prices)
                    
                    if not prices:
                        raise ValueError(f"无法获取股票价格数据 (日期: {trade_date})")
                    
                    # 计算未实现盈亏（不执行交易）
                    self.calculate_unrealized_pnl(trade_date, time_str, prices)
                    
                except Exception as e:
                    logger.error(f"处理 {decision_time} 失败: {e}")
                    raise  # 重新抛出异常，停止执行
        
        # 保存最终状态和历史记录
        self.save_state()
        
        # 保存 NAV 历史到 CSV（简化版，不包含持仓明细）
        nav_csv_path = os.path.join(self.data_dir, "nav_history.csv")
        if self.nav_history:
            # 创建简化版 DataFrame（不包含 positions_detail）
            nav_records_simple = []
            for record in self.nav_history:
                simple_record = {k: v for k, v in record.items() if k != 'positions_detail'}
                nav_records_simple.append(simple_record)
            
            nav_df = pd.DataFrame(nav_records_simple)
            nav_df.to_csv(nav_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"NAV 历史已保存到: {nav_csv_path}")
        
        # 保存完整 NAV 历史到 JSON（包含持仓明细）
        nav_json_path = os.path.join(self.data_dir, "nav_history.json")
        safe_write_json(nav_json_path, {'nav_history': self.nav_history}, backup=False)
        logger.info(f"完整 NAV 历史（含持仓明细）已保存到: {nav_json_path}")
        
        # 保存股票价格和权重数据到 JSON（格式类似 ai_stock_data.json）
        stock_data_json_path = os.path.join(self.data_dir, "star50_stock_data.json")
        safe_write_json(stock_data_json_path, self.stock_data_dict, backup=False)
        total_records = sum(len(records) for records in self.stock_data_dict.values())
        logger.info(f"STAR 50 股票价格和权重数据已保存到: {stock_data_json_path}")
        logger.info(f"  共 {len(self.stock_data_dict)} 只股票，{total_records} 条记录")
        
        # 输出最终总结
        if self.nav_history:
            final_record = self.nav_history[-1]
            
            # 计算统计指标
            returns = [r['pnl_pct'] for r in self.nav_history]
            max_return = max(returns) if returns else 0
            min_return = min(returns) if returns else 0
            avg_return = sum(returns) / len(returns) if returns else 0
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("基准跟踪运行完成 - 最终统计")
            logger.info("=" * 80)
            logger.info(f"📊 资金状况:")
            logger.info(f"  初始资金: {self.initial_capital:,.2f} 元")
            logger.info(f"  最终总资产: {final_record['total_assets']:,.2f} 元")
            logger.info(f"  股票市值: {final_record['stock_value']:,.2f} 元")
            logger.info(f"  现金余额: {final_record['cash']:,.2f} 元")
            logger.info(f"  现金比例: {final_record['cash_ratio']:.2%}")
            logger.info("")
            logger.info(f"📈 收益统计:")
            logger.info(f"  最终收益率: {final_record['pnl_pct']:.2f}%")
            logger.info(f"  最高收益率: {max_return:.2f}%")
            logger.info(f"  最低收益率: {min_return:.2f}%")
            logger.info(f"  平均收益率: {avg_return:.2f}%")
            logger.info("")
            logger.info(f"📦 持仓信息:")
            logger.info(f"  持仓股票数: {final_record['positions_count']} 只")
            logger.info(f"  处理交易日数: {len(trading_days)} 天")
            logger.info(f"  记录总数: {len(self.nav_history)} 条")
            logger.info("")
            logger.info(f"💾 输出文件:")
            logger.info(f"  状态文件: {self.state_file}")
            logger.info(f"  持仓记录: {self.position_jsonl_path}")
            logger.info(f"  NAV历史CSV: {os.path.join(self.data_dir, 'nav_history.csv')}")
            logger.info(f"  NAV历史JSON: {os.path.join(self.data_dir, 'nav_history.json')}")
            logger.info(f"  股票数据JSON: {os.path.join(self.data_dir, 'star50_stock_data.json')}")
            logger.info("=" * 80)
        
        # 关闭 TinySoft 连接
        if self.ts_client:
            try:
                self.ts_client.logout()
                del self.ts_client
                import time
                time.sleep(1.5)
                logger.info("TinySoft 会话已安全关闭")
            except Exception as e:
                logger.warning(f"关闭 TinySoft 会话时出错: {e}")


def main():
    """主函数"""
    # 在脚本中设置日期范围（可以从配置文件读取）
    config_path = os.path.join(project_root, "settings", "default_config.json")
    try:
        config = safe_read_json(config_path, default={})
        date_range = config.get("date_range", {})
        init_date = date_range.get("init_date", "2026-01-12")
        end_date = date_range.get("end_date", None)  # 如果为 None，则运行到当前日期
    except Exception as e:
        logger.warning(f"无法从配置文件读取日期范围，使用默认值: {e}")
        init_date = "2026-01-12"
        end_date = None
    
    logger.info(f"从配置文件读取日期范围: {init_date} 至 {end_date or '当前日期'}")
    
    # 创建基准跟踪器并运行（初始资金将通过网格搜索自动优化）
    benchmark = Star50Benchmark(
        init_date=init_date,
        end_date=end_date,
        initial_capital=None,  # 使用 None，将通过网格搜索自动优化
        state_file='data_flow/star50_benchmark/benchmark_state.json',
        data_dir='data_flow/star50_benchmark'
    )
    
    benchmark.run()


if __name__ == "__main__":
    main()

