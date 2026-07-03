#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from prompt_templates.prompts import DEFAULT_STOCK_SYMBOLS
from utils.position_manager import normalize_symbol, strip_exchange_prefix


NEWS_COLUMNS = ["symbol", "title", "content", "publish_time", "source", "url", "query", "search_time"]


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_dt(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
    return parsed


def _default_start() -> str:
    return (datetime.now() - timedelta(days=93)).strftime("%Y-%m-%d 00:00:00")


def _sina_symbol(symbol: str) -> str:
    plain = strip_exchange_prefix(symbol) or symbol
    return f"sh{plain}" if plain.startswith("6") else f"sz{plain}"


def fetch_sina_page(symbol: str, page: int, timeout: int) -> List[Dict[str, str]]:
    market_symbol = _sina_symbol(symbol)
    response = requests.get(
        "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php",
        params={"symbol": market_symbol, "Page": int(page)},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol={market_symbol}",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = "gb2312"
    pattern = r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2}).*?<a[^>]*href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>"
    rows: List[Dict[str, str]] = []
    for date_str, time_str, link, title in re.findall(pattern, response.text, flags=re.DOTALL):
        if "sina.com.cn" not in link and "sina.cn" not in link:
            continue
        rows.append(
            {
                "date": f"{date_str} {time_str}:00",
                "title": _clean_text(title),
                "url": link.strip(),
            }
        )
    return rows


def crawl_symbol(
    symbol: str,
    *,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_pages: int,
    sleep_seconds: float,
    timeout: int,
) -> List[Dict[str, str]]:
    normalized = normalize_symbol(symbol)
    if not normalized:
        raise ValueError(f"Invalid symbol: {symbol}")
    rows: List[Dict[str, str]] = []
    search_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    empty_pages = 0
    reached_start = False

    for page in range(1, max_pages + 1):
        try:
            page_rows = fetch_sina_page(normalized, page, timeout)
        except Exception as exc:
            print(f"{normalized} page={page}: failed: {exc}", flush=True)
            empty_pages += 1
            if empty_pages >= 3:
                break
            time.sleep(max(sleep_seconds, 0.0))
            continue

        if not page_rows:
            print(f"{normalized} page={page}: empty", flush=True)
            empty_pages += 1
            if empty_pages >= 2:
                break
            time.sleep(max(sleep_seconds, 0.0))
            continue

        page_times = [_parse_dt(item["date"]) for item in page_rows]
        valid_times = [ts for ts in page_times if not pd.isna(ts)]
        oldest = min(valid_times) if valid_times else pd.NaT
        newest = max(valid_times) if valid_times else pd.NaT
        in_range = 0

        for item in page_rows:
            ts = _parse_dt(item["date"])
            if pd.isna(ts):
                continue
            if ts < start_dt:
                reached_start = True
                continue
            if ts > end_dt:
                continue
            title = item["title"]
            if not title:
                continue
            rows.append(
                {
                    "symbol": normalized,
                    "title": title,
                    "content": title,
                    "publish_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "sina_deep",
                    "url": item["url"],
                    "query": f"{strip_exchange_prefix(normalized)} Page={page}",
                    "search_time": search_time,
                }
            )
            in_range += 1

        print(
            f"{normalized} page={page}: raw={len(page_rows)} in_range={in_range} "
            f"newest={newest} oldest={oldest}",
            flush=True,
        )
        if reached_start:
            print(f"{normalized}: reached start date {start_dt}", flush=True)
            break
        time.sleep(max(sleep_seconds, 0.0))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slowly crawl Sina stock news pages until a target start date.")
    parser.add_argument("--start-date", default=_default_start())
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_STOCK_SYMBOLS)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", default=(BASE_DIR / "jobs" / "web_news_deep_trial" / "sina_deep_news.csv").as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_dt = _parse_dt(args.start_date)
    end_dt = _parse_dt(args.end_date)
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("Invalid start/end date")

    all_rows: List[Dict[str, str]] = []
    for symbol in args.symbols:
        rows = crawl_symbol(
            symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            max_pages=args.max_pages,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=NEWS_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["symbol", "title", "publish_time"], keep="last")
        df = df.sort_values(["publish_time", "symbol"], ascending=[False, True]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print("\n=== SUMMARY ===")
    print(f"date_window: {start_dt} -> {end_dt}")
    print(f"rows: {len(df)}")
    if not df.empty:
        grouped = df.groupby("symbol")["publish_time"].agg(["count", "min", "max"]).sort_index()
        print(grouped.to_string())
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
