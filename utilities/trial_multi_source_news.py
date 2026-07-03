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
from typing import Any, Dict, List

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from prompt_templates.prompts import DEFAULT_STOCK_SYMBOLS
from utilities.prefetch_historical_news import fetch_eastmoney_notice_page, _frame_to_rows


NEWS_COLUMNS = ["symbol", "title", "content", "publish_time", "source", "url", "query", "search_time"]

STOCK_NAMES = {
    "SH688008": "澜起科技",
    "SH688111": "金山办公",
    "SH688009": "中国通号",
    "SH688981": "中芯国际",
    "SH688256": "寒武纪",
    "SH688271": "联影医疗",
    "SH688047": "龙芯中科",
    "SH688617": "惠泰医疗",
    "SH688303": "大全能源",
    "SH688180": "君实生物",
}


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


def _row(symbol: str, title: Any, publish_time: Any, source: str, url: str, query: str) -> Dict[str, str] | None:
    title_text = _clean_text(title)
    ts = _parse_dt(publish_time)
    if not title_text or pd.isna(ts):
        return None
    return {
        "symbol": symbol,
        "title": title_text,
        "content": title_text,
        "publish_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "url": url,
        "query": query,
        "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def fetch_eastmoney_notice(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, timeout: int) -> List[Dict[str, str]]:
    df = fetch_eastmoney_notice_page(symbol, 1, 100, timeout)
    return _frame_to_rows("eastmoney_notice", symbol, df, start_dt, end_dt, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def fetch_sse_announcement(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, timeout: int) -> List[Dict[str, str]]:
    code = symbol[-6:]
    rows: List[Dict[str, str]] = []
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/disclosure/listedinfo/announcement/"}
    page = 1
    while True:
        params = {
            "isPagination": "true",
            "pageHelp.pageSize": "25",
            "pageHelp.pageNo": str(page),
            "pageHelp.beginPage": str(page),
            "pageHelp.cacheSize": "1",
            "START_DATE": start_dt.strftime("%Y-%m-%d"),
            "END_DATE": end_dt.strftime("%Y-%m-%d"),
            "SECURITY_CODE": code,
            "TITLE": "",
            "BULLETIN_TYPE": "",
            "stockType": "2",
        }
        response = requests.get(
            "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        page_help = payload.get("pageHelp") or {}
        data = page_help.get("data") or []
        for group in data:
            items = group if isinstance(group, list) else [group]
            for item in items:
                url = str(item.get("URL") or "")
                if url.startswith("/"):
                    url = "https://www.sse.com.cn" + url
                row = _row(
                    symbol,
                    item.get("TITLE"),
                    item.get("SSEDATE"),
                    "sse_announcement",
                    url,
                    item.get("BULLETIN_TYPE_DESC") or "上交所公告",
                )
                if row:
                    rows.append(row)
        total = int(page_help.get("total") or len(rows))
        if page * 25 >= total or not data:
            break
        page += 1
        time.sleep(0.3)
    return rows


def fetch_cninfo_fulltext(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, timeout: int) -> List[Dict[str, str]]:
    name = STOCK_NAMES.get(symbol, symbol[-6:])
    rows: List[Dict[str, str]] = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    }
    page = 1
    while True:
        data = {
            "searchkey": name,
            "sdate": start_dt.strftime("%Y-%m-%d"),
            "edate": end_dt.strftime("%Y-%m-%d"),
            "isfulltext": "false",
            "sortName": "pubdate",
            "sortType": "desc",
            "pageNum": str(page),
        }
        response = requests.post("http://www.cninfo.com.cn/new/fulltextSearch/full", data=data, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        announcements = payload.get("announcements") or []
        for item in announcements:
            if str(item.get("secCode") or "") != symbol[-6:]:
                continue
            ts = pd.to_datetime(item.get("announcementTime"), unit="ms", errors="coerce")
            if pd.isna(ts) or ts < start_dt or ts > end_dt:
                continue
            url = str(item.get("adjunctUrl") or "")
            if url and not url.startswith("http"):
                url = "http://static.cninfo.com.cn/" + url.lstrip("/")
            row = _row(
                symbol,
                item.get("announcementTitle") or item.get("shortTitle"),
                ts,
                "cninfo_fulltext",
                url,
                name,
            )
            if row:
                rows.append(row)
        total = int(payload.get("totalRecordNum") or len(rows))
        if page * 10 >= total or not announcements:
            break
        page += 1
        time.sleep(0.3)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trial multiple free web news/disclosure sources.")
    parser.add_argument("--start-date", default=_default_start())
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_STOCK_SYMBOLS)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.8)
    parser.add_argument("--include-sina-csv", default=(BASE_DIR / "jobs/web_news_deep_trial/sina_deep_news_10symbols.csv").as_posix())
    parser.add_argument("--output", default=(BASE_DIR / "jobs/news_source_trials/multi_source_news_trial.csv").as_posix())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_dt = _parse_dt(args.start_date)
    end_dt = _parse_dt(args.end_date)
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("Invalid start/end date")

    rows: List[Dict[str, str]] = []
    sina_path = Path(args.include_sina_csv)
    if sina_path.exists():
        sina_df = pd.read_csv(sina_path, encoding="utf-8-sig")
        for record in sina_df.to_dict("records"):
            ts = _parse_dt(record.get("publish_time"))
            if not pd.isna(ts) and start_dt <= ts <= end_dt:
                rows.append({col: str(record.get(col) or "") for col in NEWS_COLUMNS})

    for symbol in args.symbols:
        for source_name, fetcher in (
            ("eastmoney_notice", fetch_eastmoney_notice),
            ("sse_announcement", fetch_sse_announcement),
            ("cninfo_fulltext", fetch_cninfo_fulltext),
        ):
            try:
                fetched = fetcher(symbol, start_dt, end_dt, args.timeout)
                rows.extend(fetched)
                print(f"{source_name} {symbol}: {len(fetched)}", flush=True)
            except Exception as exc:
                print(f"{source_name} {symbol}: failed: {exc}", flush=True)
            time.sleep(max(args.sleep, 0.0))

    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["symbol", "source", "title", "publish_time"], keep="last")
        df = df.sort_values(["publish_time", "symbol", "source"], ascending=[False, True, True]).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print("\n=== SUMMARY ===")
    print(f"rows={len(df)} output={out}")
    if not df.empty:
        print("\nby source:")
        print(df.groupby("source").size().sort_values(ascending=False).to_string())
        print("\nby symbol/source:")
        print(df.pivot_table(index="symbol", columns="source", values="title", aggfunc="count", fill_value=0).to_string())
        print("\nby symbol:")
        print(df.groupby("symbol")["publish_time"].agg(["count", "min", "max"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
