#!/usr/bin/env python3
"""
清理 data_flow/news.csv 中的 symbol 列，将其统一为带交易所前缀的格式。
"""

import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.position_manager import normalize_symbol


def _normalize_symbol_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    if raw.isdigit() and len(raw) < 6:
        raw = raw.zfill(6)
    return normalize_symbol(raw) or raw.upper()


def main():
    csv_path = Path("data_flow") / "news.csv"
    if not csv_path.exists():
        print(f"未找到 {csv_path}，无需清理。")
        return

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, encoding="utf-8")

    if df.empty:
        print("news.csv 内容为空，无需清理。")
        return

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].apply(_normalize_symbol_value)
        df = df[df["symbol"].astype(str).str.len() > 0]
    if "source" in df.columns:
        df = df[df["source"].notna()]
    if "title" in df.columns:
        df = df[df["title"].notna()]

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已清理 {csv_path}，共 {len(df)} 条记录。")


if __name__ == "__main__":
    main()

