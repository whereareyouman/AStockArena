"""
Helper utilities to trigger the backup_data.py script from Python entrypoints.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = PROJECT_ROOT / "utilities" / "backup_data.py"


def run_backup_snapshot(reason: str = "manual", retain: Optional[int] = None) -> bool:
    """
    Execute the backup_data.py script.

    Args:
        reason: Human-readable context, stored in BACKUP_REASON env var for logs.
        retain: Optional override for --retain parameter.

    Returns:
        True if the backup script completed successfully, False otherwise.
    """
    if not BACKUP_SCRIPT.exists():
        try:
            print(f"[WARNING] Backup script not found at {BACKUP_SCRIPT}")
        except UnicodeEncodeError:
            print(f"WARNING: Backup script not found at {BACKUP_SCRIPT}")
        return False

    cmd = [sys.executable, str(BACKUP_SCRIPT)]
    if retain is not None:
        cmd.extend(["--retain", str(retain)])

    env = os.environ.copy()
    env["BACKUP_REASON"] = reason

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        try:
            print("[OK] Backup snapshot completed.")
        except UnicodeEncodeError:
            print("Backup snapshot completed.")
        if completed.stdout:
            print(completed.stdout.strip())
        if completed.stderr:
            print(completed.stderr.strip())
        return True
    except subprocess.CalledProcessError as exc:
        try:
            print(f"[WARNING] Backup script exited with non-zero status ({exc.returncode}).")
            print(f"[WARNING] Backup reason: {env.get('BACKUP_REASON', 'unknown')}")
        except UnicodeEncodeError:
            print(f"WARNING: Backup script exited with non-zero status ({exc.returncode}).")
            print(f"WARNING: Backup reason: {env.get('BACKUP_REASON', 'unknown')}")
        if exc.stdout:
            print(f"[WARNING] Backup stdout: {exc.stdout.strip()}")
        if exc.stderr:
            print(f"[WARNING] Backup stderr: {exc.stderr.strip()}")
        return False
    except Exception as exc:
        try:
            print(f"[WARNING] Failed to run backup script: {exc}")
            print(f"[WARNING] Backup reason: {env.get('BACKUP_REASON', 'unknown')}")
            import traceback
            print(f"[WARNING] Backup exception traceback: {traceback.format_exc()}")
        except UnicodeEncodeError:
            print(f"WARNING: Failed to run backup script: {exc}")
            print(f"WARNING: Backup reason: {env.get('BACKUP_REASON', 'unknown')}")
        return False

def save_pnl_snapshot(reason: str) -> bool:
    """Save PnL/equity curve snapshot to a separate file."""
    try:
        # --- 本地辅助函数定义 ---
        def _read_jsonl_tail(filepath: Path, limit: int = 1000) -> List[Dict]:
            items = []
            if not filepath.exists():
                return items
            try:
                import collections
                with open(filepath, 'r', encoding='utf-8') as f:
                    # 使用 deque 读取最后 N 行
                    lines = collections.deque(f, limit)
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                print(f"Error reading tail of {filepath}: {e}")
            return items

        def _load_initial_cash() -> float:
            try:
                config_path = PROJECT_ROOT / "settings" / "default_config.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        return float(config.get("agent_config", {}).get("initial_cash", 1000000.0))
            except Exception:
                pass
            return 1000000.0

        def _pick_position_file() -> Optional[Path]:
            """
            选择用于计算收益曲线的 position.jsonl。
            优先级：
            1) default_config.json 中 enabled=true 的模型 signature
            2) data_flow/trading_summary_each_agent/*/position/position.jsonl 中 mtime 最新的
            3) legacy: data_flow/trading_summary_each_agent/default/position.jsonl
            """
            agent_data_root = PROJECT_ROOT / "data_flow" / "trading_summary_each_agent"

            # 1) 从配置里拿 enabled 模型
            try:
                config_path = PROJECT_ROOT / "settings" / "default_config.json"
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    models = cfg.get("models") or []
                    enabled = [m for m in models if m.get("enabled") is True]
                    if enabled:
                        sig = str(enabled[0].get("signature") or enabled[0].get("name") or "").strip()
                        if sig:
                            candidate = agent_data_root / sig / "position" / "position.jsonl"
                            if candidate.exists():
                                return candidate
            except Exception:
                pass

            # 2) 选择最新的 position.jsonl
            try:
                if agent_data_root.exists():
                    candidates = [p for p in agent_data_root.glob("*/position/position.jsonl") if p.exists()]
                    if candidates:
                        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        return candidates[0]
            except Exception:
                pass

            # 3) legacy 默认路径
            legacy = agent_data_root / "default" / "position.jsonl"
            return legacy if legacy.exists() else None

        def _get_price_at_time(symbol: str, decision_time: str = None, date_str: str = None) -> float:
            """
            根据决策时点获取股票价格。
            从 ai_stock_data.json 中查找 <= decision_time 的最新价格。
            """
            stock_data_path = PROJECT_ROOT / "data_flow" / "ai_stock_data.json"
            if not stock_data_path.exists():
                return 0.0
            
            try:
                with open(stock_data_path, "r", encoding="utf-8") as f:
                    full_data = json.load(f)
                
                # 尝试多个可能的键名
                stock_entry = None
                symbol_upper = symbol.upper()
                for key in [symbol_upper, symbol_upper.replace("SH", "").replace("SZ", ""), f"SH{symbol_upper}", f"SZ{symbol_upper}"]:
                    if key in full_data:
                        stock_entry = full_data[key]
                        break
                
                if not stock_entry or not isinstance(stock_entry, dict):
                    return 0.0
                
                # 优先使用小时线行情，如果没有则使用日线行情
                hourly_data = stock_entry.get("小时线行情") or []
                daily_data = stock_entry.get("日线行情") or []
                
                data_list = hourly_data if hourly_data else daily_data
                if not data_list:
                    return 0.0
                
                # 如果没有指定决策时点，返回最新价格
                if not decision_time and not date_str:
                    last_item = data_list[-1]
                    return float(last_item.get("close") or last_item.get("buy1") or 0)
                
                # 根据 decision_time 查找 <= 目标时间的最新价格
                target_time = decision_time or (date_str + " 15:00:00" if date_str else None)
                if not target_time:
                    last_item = data_list[-1]
                    return float(last_item.get("close") or last_item.get("buy1") or 0)
                
                # 倒序遍历，找第一个 <= target_time 的记录
                best_match = None
                for item in reversed(data_list):
                    item_time = item.get("time") or item.get("date") or ""
                    if item_time and item_time <= target_time:
                        best_match = item
                        break
                
                if best_match:
                    return float(best_match.get("close") or best_match.get("buy1") or 0)
                
                # 如果找不到，返回第一条记录的价格（最早的价格）
                first_item = data_list[0]
                return float(first_item.get("close") or first_item.get("buy1") or 0)
                
            except Exception:
                return 0.0

        def _estimate_equity_for_positions(positions: Dict[str, Any], decision_time: str = None, date_str: str = None) -> Tuple[float, float, float, float]:
            """
            估算持仓权益。
            返回: (cash, equity, market_value, profit)
            根据决策时点获取对应时间的价格，不允许使用 avg_price 作为估算。
            """
            cash = float(positions.get("CASH", 0.0))
            market_value = 0.0

            for symbol, details in positions.items():
                if symbol == "CASH":
                    continue
                if isinstance(details, dict):
                    shares = float(details.get("shares", 0))
                    if shares == 0:
                        continue
                    
                    # 根据决策时点获取对应时间的价格
                    price = _get_price_at_time(symbol, decision_time, date_str)
                        
                    # 如果还是没找到价格，报错而不是使用 avg_price
                    if price == 0.0:
                        raise ValueError(
                            f"无法获取股票 {symbol} 在决策时点 {decision_time} 的价格来计算权益。"
                            f"日期: {date_str}。"
                            f"请确保 ai_stock_data.json 中包含该股票在该时点的价格数据。"
                        )
                        
                    market_value += shares * price

            equity = cash + market_value
            return cash, equity, market_value, 0.0

        # --- 主逻辑：为所有模型生成 PnL 快照 ---
        agent_data_root = PROJECT_ROOT / "data_flow" / "trading_summary_each_agent"
        if not agent_data_root.exists():
            return False
        
        # 获取所有模型的 position 文件
        position_files = list(agent_data_root.glob("*/position/position.jsonl"))
        if not position_files:
            return False
        
        initial_cash = _load_initial_cash()
        pnl_dir = Path(PROJECT_ROOT) / "data_flow" / "pnl_snapshots"
        pnl_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        for pos_file in position_files:
            # 从文件路径中提取模型 signature
            signature = "unknown"
            try:
                parts = pos_file.parts
                # 查找 signature（在 position 目录的上一级）
                if "position" in parts:
                    pos_idx = parts.index("position")
                    if pos_idx > 0:
                        signature = parts[pos_idx - 1]
            except Exception:
                continue
            
            if signature == "unknown":
                continue
             
            all_items = _read_jsonl_tail(pos_file, limit=100000)
            
            if not all_items:
                continue
        
            # 不再按日期去重，而是保留所有决策时点的记录
            # 使用 (date, decision_time) 作为唯一键，这样每个决策时点都有独立的 PnL 记录
            by_datetime = {}
            for it in all_items:
                d = it.get("date")
                decision_time = it.get("decision_time", "")
                if not d:
                    continue
                # 使用 date + decision_time 作为唯一键
                key = f"{d}_{decision_time}"
                # 如果同一决策时点有多条记录，保留 id 最大的（最新的）
                if key not in by_datetime or (it.get("id", -1) > by_datetime[key].get("id", -1)):
                    by_datetime[key] = it
            
            # 计算新的 PnL 数据（为每个决策时点生成独立记录）
            new_pnl_data = []
            for key, rec in sorted(by_datetime.items()):
                d = rec.get("date")
                decision_time = rec.get("decision_time", "")
                decision_count = rec.get("decision_count", 0)
                
                _, equity_val, _, _ = _estimate_equity_for_positions(
                    rec.get("positions", {}) or {}, decision_time, d
                )
                ret_pct = (equity_val / initial_cash - 1.0) * 100.0 if initial_cash > 0 else 0.0
                new_pnl_data.append({
                    "date": d,
                    "decision_time": decision_time,  # 添加决策时点
                    "decision_count": decision_count,  # 添加决策序号
                    "returnPct": ret_pct,
                    "equity": equity_val,
                    "id": rec.get("id"),
                })

            # 使用模型 signature 作为文件名
            # Windows 文件名清理：移除无效字符（: / \ < > " | ? *）
            safe_signature = signature.replace(":", "-").replace("/", "-").replace("\\", "-").replace("<", "_").replace(">", "_").replace('"', "_").replace("|", "_").replace("?", "_").replace("*", "_").replace(" ", "_")
            pnl_file = pnl_dir / f"pnl_{safe_signature}.json"
            
            # 读取现有数据（如果存在）
            existing_data = []
            if pnl_file.exists():
                try:
                    with open(pnl_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = []
            
            # 合并数据：按 (date, decision_time) 去重，保留最新的记录（通过 id 判断）
            merged_by_datetime = {}
            # 先添加现有数据（兼容旧格式：可能没有 decision_time 字段）
            for item in existing_data:
                d = item.get("date")
                decision_time = item.get("decision_time", "")
                if d:
                    key = f"{d}_{decision_time}"
                    existing_id = item.get("id", -1)
                    if key not in merged_by_datetime or existing_id > merged_by_datetime[key].get("id", -1):
                        merged_by_datetime[key] = item
            
            # 再添加新数据（会覆盖同决策时点的旧数据）
            for item in new_pnl_data:
                d = item.get("date")
                decision_time = item.get("decision_time", "")
                if d:
                    key = f"{d}_{decision_time}"
                    new_id = item.get("id", -1)
                    if key not in merged_by_datetime or new_id > merged_by_datetime[key].get("id", -1):
                        merged_by_datetime[key] = item
            
            # 按日期和时间排序并保存
            def sort_key(x):
                date_str = x.get("date", "")
                time_str = x.get("decision_time", "")
                return (date_str, time_str)
            
            final_data = sorted(merged_by_datetime.values(), key=sort_key)
            
            try:
                with open(pnl_file, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, indent=2, ensure_ascii=False)
                success_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to save PnL snapshot for {signature}: {e}")
        
        return success_count > 0
    except Exception as e:
        try:
            print(f"[WARNING] Error saving PnL snapshot: {e}")
        except UnicodeEncodeError:
            print(f"WARNING: Error saving PnL snapshot: {e}")
        return False

