#!/usr/bin/env python3
"""
修复持仓文件中现金为0的记录，将其设置为初始现金（1000000.0）
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.position_manager import _load_initial_cash


def fix_zero_cash_in_file(file_path: Path, initial_cash: float) -> tuple[int, int]:
    """
    修复单个持仓文件中的零现金记录
    
    Returns:
        (fixed_count, total_lines): 修复的记录数和总行数
    """
    if not file_path.exists():
        print(f"[WARN] 文件不存在: {file_path}")
        return 0, 0
    
    fixed_count = 0
    total_lines = 0
    fixed_lines = []
    
    # 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                fixed_lines.append(line)
                continue
            
            try:
                record = json.loads(line)
                positions = record.get("positions", {})
                cash = positions.get("CASH", 0.0)
                
                # 如果现金为0，修复它
                if cash == 0.0 or cash == 0:
                    positions["CASH"] = initial_cash
                    record["positions"] = positions
                    fixed_count += 1
                    print(f"  [FIX] 修复记录: date={record.get('date')}, decision_time={record.get('decision_time')}, id={record.get('id')}")
                
                fixed_lines.append(json.dumps(record, ensure_ascii=False))
            except Exception as e:
                print(f"  [WARN] 解析行失败: {e}")
                fixed_lines.append(line)
    
    # 写回文件
    if fixed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in fixed_lines:
                f.write(line + '\n')
    
    return fixed_count, total_lines


def main():
    """主函数：修复所有持仓文件中的零现金记录"""
    initial_cash = _load_initial_cash()
    print(f"[INFO] 初始现金设置为: {initial_cash:,.2f}")
    print()
    
    # 查找所有持仓文件
    base_dir = project_root / "data_flow" / "trading_summary_each_agent"
    
    if not base_dir.exists():
        print(f"[ERROR] 目录不存在: {base_dir}")
        return
    
    position_files = list(base_dir.glob("*/position/position.jsonl"))
    
    if not position_files:
        print(f"[WARN] 未找到任何持仓文件")
        return
    
    print(f"[INFO] 找到 {len(position_files)} 个持仓文件")
    print()
    
    total_fixed = 0
    total_processed = 0
    
    for pos_file in sorted(position_files):
        agent_name = pos_file.parent.parent.name
        print(f"[PROCESS] 处理: {agent_name}")
        fixed, total = fix_zero_cash_in_file(pos_file, initial_cash)
        total_fixed += fixed
        total_processed += total
        if fixed > 0:
            print(f"  [OK] 修复了 {fixed} 条记录（共 {total} 条）")
        else:
            print(f"  [OK] 无需修复（共 {total} 条）")
        print()
    
    print("=" * 60)
    print(f"[SUMMARY] 修复完成:")
    print(f"   - 处理文件数: {len(position_files)}")
    print(f"   - 修复记录数: {total_fixed}")
    print(f"   - 总记录数: {total_processed}")
    print()
    print("[SUCCESS] 所有现金为0的记录已修复为初始现金！")


if __name__ == "__main__":
    main()

