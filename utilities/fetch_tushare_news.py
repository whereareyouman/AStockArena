#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

from prompt_templates.prompts import DEFAULT_STOCK_SYMBOLS
from utils.news_cache_guard import validate_news_cache_integrity, write_news_manifest
from utils.position_manager import file_transaction_lock
from utils.tushare_config import TUSHARE_HTTP_URL, TUSHARE_TOKEN, get_tushare_module, get_tushare_pro


NEWS_COLUMNS = ["symbol", "title", "content", "publish_time", "source", "url", "query", "search_time"]

DEFAULT_SOURCES = ["sina", "10jqka", "eastmoney", "wallstreetcn", "cls", "yicai", "jinrongjie"]

STOCK_ALIASES: Dict[str, List[str]] = {
    "SH688008": ["688008", "澜起科技", "澜起", "Montage"],
    "SH688111": ["688111", "金山办公", "WPS"],
    "SH688009": ["688009", "中国通号", "通号"],
    "SH688981": ["688981", "中芯国际", "中芯", "SMIC"],
    "SH688256": ["688256", "寒武纪", "Cambricon"],
    "SH688271": ["688271", "联影医疗", "联影"],
    "SH688047": ["688047", "龙芯中科", "龙芯", "Loongson"],
    "SH688617": ["688617", "惠泰医疗", "惠泰"],
    "SH688303": ["688303", "大全能源"],
    "SH688180": ["688180", "君实生物", "君实"],
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_time(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
    return parsed


def _format_tushare_dt(value: str) -> str:
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _default_start_end(days: int) -> Tuple[str, str]:
    end = datetime.now()
    start = end - timedelta(days=max(int(days), 1))
    return start.strftime("%Y-%m-%d 00:00:00"), end.strftime("%Y-%m-%d %H:%M:%S")


def _import_tushare():
    try:
        import tushare as ts  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "未安装 tushare。请先运行：python -m pip install tushare"
        ) from exc
    return ts


def _extract_fields(row: pd.Series) -> Dict[str, Any]:
    def first_present(*keys: str) -> Any:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            if str(value).strip():
                return value
        return ""

    title = first_present("title", "headline", "content")
    content = first_present("content", "summary", "title", "headline")
    publish_time = first_present("datetime", "time", "publish_time", "date")
    source = first_present("src", "source")
    url = first_present("url", "link")
    return {
        "title": _clean_text(title),
        "content": _clean_text(content),
        "publish_time": publish_time,
        "source": _clean_text(source),
        "url": str(url or "").strip(),
    }


def _matched_symbols(title: str, content: str, symbols: Sequence[str]) -> List[Tuple[str, str]]:
    haystack = f"{title} {content}"
    matches: List[Tuple[str, str]] = []
    for symbol in symbols:
        aliases = STOCK_ALIASES.get(symbol, [symbol, symbol[-6:]])
        for alias in aliases:
            if alias and alias in haystack:
                matches.append((symbol, alias))
                break
    return matches


def _fetch_source(
    pro: Any,
    *,
    source: str,
    start_date: str,
    end_date: str,
    fields: Optional[str],
    retries: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            kwargs = {"src": source, "start_date": start_date, "end_date": end_date}
            if fields:
                kwargs["fields"] = fields
            df = pro.news(**kwargs)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"tushare pro.news returned {type(df)!r}")
            return df
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait = sleep_seconds * attempt
            print(f"{source}: attempt {attempt}/{retries} failed: {exc}; retry in {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{source}: failed after {retries} attempts: {last_exc}")


def _date_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int,
    chunk_hours: int = 0,
) -> List[Tuple[str, str]]:
    if chunk_hours > 0:
        start = pd.to_datetime(start_date, errors="raise")
        end = pd.to_datetime(end_date, errors="raise")
        chunks: List[Tuple[str, str]] = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + pd.Timedelta(hours=chunk_hours) - pd.Timedelta(seconds=1), end)
            chunks.append((cur.strftime("%Y-%m-%d %H:%M:%S"), chunk_end.strftime("%Y-%m-%d %H:%M:%S")))
            cur = chunk_end + pd.Timedelta(seconds=1)
        return chunks
    if chunk_days <= 0:
        return [(start_date, end_date)]
    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")
    chunks: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days) - pd.Timedelta(seconds=1), end)
        chunks.append((cur.strftime("%Y-%m-%d %H:%M:%S"), chunk_end.strftime("%Y-%m-%d %H:%M:%S")))
        cur = chunk_end + pd.Timedelta(seconds=1)
    return chunks


def _build_news_rows(
    raw_frames: Iterable[Tuple[str, pd.DataFrame]],
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    search_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_ts = _parse_time(start_date)
    end_ts = _parse_time(end_date)

    for source, df in raw_frames:
        for _, raw_row in df.iterrows():
            item = _extract_fields(raw_row)
            title = item["title"]
            content = item["content"]
            if not title:
                continue
            ts = _parse_time(item["publish_time"])
            if pd.isna(ts):
                continue
            if not pd.isna(start_ts) and ts < start_ts:
                continue
            if not pd.isna(end_ts) and ts > end_ts:
                continue
            for symbol, alias in _matched_symbols(title, content, symbols):
                rows.append(
                    {
                        "symbol": symbol,
                        "title": title,
                        "content": content or title,
                        "publish_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": f"tushare:{source}",
                        "url": item["url"],
                        "query": alias,
                        "search_time": search_time,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    out = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    out = out.drop_duplicates(subset=["symbol", "title", "publish_time"], keep="last")
    out = out.sort_values(["publish_time", "symbol"], ascending=[False, True])
    return out.reset_index(drop=True)


def _append_news_csv(output_df: pd.DataFrame, news_csv_path: Path) -> pd.DataFrame:
    with file_transaction_lock(news_csv_path, suffix=".news.lock"):
        if news_csv_path.exists():
            validate_news_cache_integrity(news_csv_path, strict=True)
        if news_csv_path.exists() and news_csv_path.stat().st_size > 0:
            existing = pd.read_csv(news_csv_path, encoding="utf-8-sig")
        else:
            existing = pd.DataFrame(columns=NEWS_COLUMNS)
        combined = pd.concat([existing, output_df], ignore_index=True)
        for col in NEWS_COLUMNS:
            if col not in combined.columns:
                combined[col] = ""
        combined = combined[NEWS_COLUMNS]
        combined = combined.drop_duplicates(subset=["symbol", "title", "publish_time"], keep="last")
        combined = combined.sort_values(["publish_time", "symbol"], ascending=[False, True], na_position="last")
        news_csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(news_csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        write_news_manifest(news_csv_path, combined)
        return combined


def parse_args() -> argparse.Namespace:
    default_start, default_end = _default_start_end(2)
    parser = argparse.ArgumentParser(
        description="Fetch Tushare news into an AStockArena-compatible news.csv trial file."
    )
    parser.add_argument("--token", default=os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN") or TUSHARE_TOKEN)
    parser.add_argument("--start-date", default=default_start, help="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end-date", default=default_end, help="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_STOCK_SYMBOLS)
    parser.add_argument("--output", default=(BASE_DIR / "jobs" / "tushare_news_trial" / "tushare_news.csv").as_posix())
    parser.add_argument("--raw-output", default=(BASE_DIR / "jobs" / "tushare_news_trial" / "tushare_news_raw.csv").as_posix())
    parser.add_argument("--append-news-csv", action="store_true", help="Merge matched rows into data_flow/news.csv")
    parser.add_argument("--news-csv", default=(BASE_DIR / "data_flow" / "news.csv").as_posix())
    parser.add_argument("--fields", default=None, help="Optional Tushare fields, e.g. datetime,title,content,src,url")
    parser.add_argument("--chunk-days", type=int, default=0, help="Split each source request into N-day chunks; use 1 for historical news.")
    parser.add_argument("--chunk-hours", type=int, default=6, help="Split each source request into N-hour chunks.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--limit-raw-rows", type=int, default=0, help="For quick local trials; 0 means no limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        print("Missing TUSHARE_TOKEN. Set it first, for example:", file=sys.stderr)
        print("  export TUSHARE_TOKEN='your_tushare_token'", file=sys.stderr)
        return 2

    start_date = _format_tushare_dt(args.start_date)
    end_date = _format_tushare_dt(args.end_date)
    if args.token == TUSHARE_TOKEN:
        ts = get_tushare_module()
        pro = get_tushare_pro()
    else:
        ts = _import_tushare()
        ts.set_token(args.token)
        pro = ts.pro_api()
        pro._DataApi__http_url = TUSHARE_HTTP_URL

    raw_frames: List[Tuple[str, pd.DataFrame]] = []
    for source in args.sources:
        source_frames: List[pd.DataFrame] = []
        for chunk_start, chunk_end in _date_chunks(start_date, end_date, args.chunk_days, args.chunk_hours):
            print(f"Fetching Tushare news source={source} {chunk_start} -> {chunk_end}", flush=True)
            try:
                df = _fetch_source(
                    pro,
                    source=source,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    fields=args.fields,
                    retries=args.retries,
                    sleep_seconds=args.sleep,
                )
            except Exception as exc:
                print(f"{source}: failed {chunk_start} -> {chunk_end}: {exc}", file=sys.stderr, flush=True)
                continue
            if args.limit_raw_rows and len(df) > args.limit_raw_rows:
                df = df.head(args.limit_raw_rows)
            print(f"{source}: raw rows={len(df)}", flush=True)
            source_frames.append(df)
            time.sleep(max(args.sleep, 0.0))
        if source_frames:
            raw_frames.append((source, pd.concat(source_frames, ignore_index=True)))

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_out = []
    for source, df in raw_frames:
        if df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "tushare_source", source)
        raw_out.append(tmp)
    if raw_out:
        pd.concat(raw_out, ignore_index=True).to_csv(raw_output, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(raw_output, index=False, encoding="utf-8-sig")

    matched = _build_news_rows(raw_frames, symbols=args.symbols, start_date=start_date, end_date=end_date)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print(f"Matched rows: {len(matched)}")
    if not matched.empty:
        print(matched.groupby("symbol").size().sort_values(ascending=False).to_string())
    print(f"Wrote matched CSV: {output}")
    print(f"Wrote raw CSV: {raw_output}")

    if args.append_news_csv:
        combined = _append_news_csv(matched, Path(args.news_csv))
        print(f"Merged into {args.news_csv}; combined rows={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
