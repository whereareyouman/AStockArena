#!/usr/bin/env python3
"""
预生成所有决策时点的 snapshot，避免进程运行时生成导致内存累积。

使用方法：
  # 生成指定日期所有时点的 snapshot
  TODAY_DATE=2026-01-14 python utilities/prefetch_snapshots.py

  # 或指定日期和时点
  TODAY_DATE=2026-01-14 CURRENT_TIME="2026-01-14 10:30:00" python utilities/prefetch_snapshots.py
"""

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_engine.agent.agent import AgenticWorkflow as BaseAgent
from utils.position_manager import normalize_symbol


def get_trading_hours(today_date: str) -> list[str]:
    """获取标准决策时点"""
    return [
        f"{today_date} 10:30:00",
        f"{today_date} 11:30:00",
        f"{today_date} 14:00:00",
    ]


def prefetch_snapshot(today_date: str, current_time: str, decision_count: int = 1) -> bool:
    """为指定时点生成 snapshot"""
    print(f"\n{'='*60}")
    print(f"📦 预生成 snapshot: {current_time}")
    print(f"{'='*60}")
    
    try:
        agent = BaseAgent(
            signature="snapshot-pregen",
            basemodel="noop",
            news_csv_path="./data_flow/news.csv",
            log_path="./data_flow/trading_summary_each_agent",
            init_date=today_date,
        )
        agent.runtime_context["TODAY_DATE"] = today_date
        agent.runtime_context["CURRENT_TIME"] = current_time
        agent.runtime_context["DECISION_COUNT"] = decision_count

        def build_snapshot():
            bundle = agent._collect_prefetch_bundle(today_date, current_time, decision_count)
            # 永远由 LLM 自己生成 Observation Summary：共享快照里不保存 observation_summary
            bundle.pop("observation_summary", None)
            return bundle

        snapshot_result = agent.prefetch_coordinator.ensure_snapshot(
            today_date=today_date,
            current_time=current_time,
            symbols_signature=agent._symbols_signature(),
            builder=build_snapshot,
        )
        bundle = snapshot_result.data

        if snapshot_result.created:
            print(f"✅ 生成新 snapshot: {snapshot_result.path}")
        else:
            print(f"📄 使用已存在的 snapshot: {snapshot_result.path}")

        # 验证数据完整性
        print(f"   snapshot_id: {bundle.get('snapshot_id')}")
        print(f"   股票数量: {len(bundle.get('symbols', []))}")
        
        # 检查每个股票的数据
        missing_data = []
        for sym in agent.stock_symbols:
            normalized = normalize_symbol(sym)
            news_count = len(bundle.get("news", {}).get(normalized, {}).get("news", []))
            has_price = bool(bundle.get("prices", {}).get(normalized))
            has_indicator = bool(bundle.get("indicators", {}).get(normalized))
            
            if not has_price or not has_indicator:
                missing_data.append(f"{normalized} (news:{news_count}, price:{has_price}, indicator:{has_indicator})")
        
        if missing_data:
            print(f"   ⚠️ 缺失数据的股票: {', '.join(missing_data)}")
        else:
            print(f"   ✅ 所有股票数据完整")

        # 清理 agentic workflow 以释放内存
        del agent
        return True

    except Exception as e:
        print(f"❌ 生成 snapshot 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """主函数"""
    today = os.environ.get("TODAY_DATE") or datetime.now().strftime("%Y-%m-%d")
    current_time_env = os.environ.get("CURRENT_TIME")
    
    print(f"\n{'='*60}")
    print(f"🚀 预生成 Snapshot 脚本")
    print(f"{'='*60}")
    print(f"日期: {today}")
    
    if current_time_env:
        # 如果指定了 CURRENT_TIME，只生成该时点的 snapshot
        decision_times = [current_time_env]
        print(f"时点: {current_time_env}")
    else:
        # 否则生成所有标准时点
        decision_times = get_trading_hours(today)
        print(f"时点: 10:30, 11:30, 14:00 (共 {len(decision_times)} 个)")
    
    print(f"{'='*60}\n")
    
    success_count = 0
    for idx, current_time in enumerate(decision_times, 1):
        decision_count = idx  # 1, 2, 3
        if prefetch_snapshot(today, current_time, decision_count):
            success_count += 1
        else:
            print(f"❌ 失败: {current_time}")
    
    print(f"\n{'='*60}")
    print(f"📊 总结")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(decision_times)}")
    if success_count == len(decision_times):
        print(f"✅ 所有 snapshot 已预生成，可以启动交易进程")
    else:
        print(f"⚠️ 部分 snapshot 生成失败，请检查错误信息")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

