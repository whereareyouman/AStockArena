#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from prompt_templates.prompts import DEFAULT_STOCK_SYMBOLS
from utils.eastmoney_news import EASTMONEY_SEARCH_URL, _default_headers, _parse_json_or_jsonp
from utils.position_manager import file_transaction_lock, normalize_symbol, strip_exchange_prefix
from utils.tushare_config import get_tushare_pro


NEWS_COLUMNS = ["symbol", "title", "content", "publish_time", "source", "url", "query", "search_time"]
TUSHARE_NEWS_LIMIT_GUARD = 1490
TUSHARE_MIN_SPLIT_SECONDS = 3600

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

TUSHARE_NEWS_SOURCES = ["sina", "10jqka", "eastmoney", "wallstreetcn", "cls", "yicai", "jinrongjie"]

SOURCE_PRIORITY = {
    "sse_announcement": 0,
    "cninfo_fulltext": 1,
    "东方财富公告": 2,
    "tushare:sina": 3,
    "tushare:10jqka": 4,
    "tushare:eastmoney": 5,
    "tushare:wallstreetcn": 6,
    "tushare:cls": 7,
    "tushare:yicai": 8,
    "tushare:jinrongjie": 9,
    "sina_deep": 10,
    "新浪财经": 11,
    "东方财富新闻(AkShare)": 12,
    "东方财富公司动态(AkShare)": 13,
}


def _load_config() -> Dict[str, Any]:
    config_path = BASE_DIR / "settings" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_dt(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
    return parsed


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


def _normalize_title_key(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[《》“”\"'‘’（）()【】\[\]：:；;，,。.!！?？、·\-—_]+", "", text)
    return text


def _source_priority(value: Any) -> int:
    text = _clean_text(value)
    return SOURCE_PRIORITY.get(text, 50)


def _normalize_symbol_value(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if not normalized:
        raise ValueError(f"Invalid symbol: {symbol}")
    return normalized


def _polite_sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(float(seconds))


def _retry_backoff_seconds(attempt: int, base_seconds: float) -> float:
    """Exponential backoff with a small jitter between retries."""
    return float(base_seconds) * (2 ** max(attempt - 1, 0)) + random.uniform(0.0, 0.5)


def _fetch_with_retry(
    label: str,
    fetcher,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Run a fetch callable; on failure wait and retry before returning empty."""
    attempts = max(int(max_retries), 1)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            df = fetcher()
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{label} returned {type(df)!r}, expected DataFrame")
            if attempt > 1:
                print(f"{label}: success on attempt {attempt}/{attempts}", flush=True)
            return df
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                print(f"{label}: failed after {attempts} attempts: {exc}", flush=True)
                break
            wait = _retry_backoff_seconds(attempt, retry_backoff_seconds)
            print(
                f"{label}: attempt {attempt}/{attempts} error {exc}; retry in {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
    if last_exc is not None:
        return pd.DataFrame()
    return pd.DataFrame()


def _row(
    *,
    symbol: str,
    title: Any,
    content: Any,
    publish_time: Any,
    source: Any,
    url: Any,
    query: str,
    search_time: str,
) -> Optional[Dict[str, str]]:
    normalized = _normalize_symbol_value(symbol)
    title_text = _clean_text(title)
    if not title_text:
        return None
    ts = _parse_dt(publish_time)
    if pd.isna(ts):
        return None
    content_text = _clean_text(content) or title_text
    return {
        "symbol": normalized,
        "title": title_text,
        "content": content_text,
        "publish_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source": _clean_text(source),
        "url": str(url or "").strip(),
        "query": query,
        "search_time": search_time,
    }


def _extract_search_items(result_obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(result_obj, dict):
        return []
    for key in ("cmsArticleWebOld", "cmsArticleWeb"):
        value = result_obj.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for sub_key in ("data", "data_pipeline", "list", "items"):
                sub_value = value.get(sub_key)
                if isinstance(sub_value, list):
                    return sub_value
    return []


def fetch_eastmoney_search_page(symbol: str, page_index: int, page_size: int, timeout: int) -> pd.DataFrame:
    plain = strip_exchange_prefix(symbol) or symbol
    inner_param: Dict[str, Any] = {
        "uid": "",
        "keyword": plain,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": int(page_index),
                "pageSize": int(page_size),
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    ts_ms = int(time.time() * 1000)
    cb = f"jQuery{random.randint(10000000000000000000, 99999999999999999999)}_{ts_ms}"
    params = {"cb": cb, "param": json.dumps(inner_param, ensure_ascii=False), "_": str(ts_ms)}
    response = requests.get(
        EASTMONEY_SEARCH_URL,
        params=params,
        headers=_default_headers(plain),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = _parse_json_or_jsonp(response.text)
    items = _extract_search_items(payload.get("result") if isinstance(payload, dict) else None)
    return pd.DataFrame(items)


def fetch_eastmoney_notice_page(symbol: str, page_index: int, page_size: int, timeout: int) -> pd.DataFrame:
    plain = strip_exchange_prefix(symbol) or symbol
    response = requests.get(
        "https://np-anotice-stock.eastmoney.com/api/security/ann",
        params={
            "page_size": int(page_size),
            "page_index": int(page_index),
            "ann_type": "A",
            "client_source": "web",
            "stock_list": plain,
            "f_node": 1,
            "s_node": 1,
        },
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://quote.eastmoney.com/{plain}.html",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    items = (payload.get("data") or {}).get("list") or (payload.get("data_pipeline") or {}).get("list") or []
    return pd.DataFrame(items)


def fetch_sina_stock_news(symbol: str, timeout: int) -> pd.DataFrame:
    plain = strip_exchange_prefix(symbol) or symbol
    market_symbol = f"sh{plain}" if plain.startswith("6") else f"sz{plain}"
    response = requests.get(
        f"https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol={market_symbol}",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = "gb2312"
    pattern = r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2}).*?<a[^>]*href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>"
    rows = [
        {
            "date": f"{date_str} {time_str}:00",
            "title": title,
            "content": title,
            "source": "新浪财经",
            "url": link,
        }
        for date_str, time_str, link, title in re.findall(pattern, response.text)
        if "sina.com.cn" in link or "sina.cn" in link
    ]
    return pd.DataFrame(rows)


def _sina_market_symbol(symbol: str) -> str:
    plain = strip_exchange_prefix(symbol) or symbol
    return f"sh{plain}" if plain.startswith("6") else f"sz{plain}"


def fetch_sina_stock_news_deep(
    symbol: str,
    timeout: int,
    *,
    max_pages: int,
    start_dt: pd.Timestamp,
    page_sleep_seconds: float = 0.8,
) -> pd.DataFrame:
    """Slowly page Sina stock news until the requested start date is reached."""
    market_symbol = _sina_market_symbol(symbol)
    rows: List[Dict[str, str]] = []
    empty_pages = 0
    reached_start = False
    pattern = r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2}).*?<a[^>]*href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>"

    for page in range(1, max(int(max_pages), 1) + 1):
        response = requests.get(
            "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php",
            params={"symbol": market_symbol, "Page": int(page)},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": (
                    "https://vip.stock.finance.sina.com.cn/corp/view/"
                    f"vCB_AllNewsStock.php?symbol={market_symbol}"
                ),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        response.encoding = "gb2312"
        page_rows = re.findall(pattern, response.text, flags=re.DOTALL)
        if not page_rows:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue
        empty_pages = 0

        for date_str, time_str, link, title in page_rows:
            if "sina.com.cn" not in link and "sina.cn" not in link:
                continue
            ts = _parse_dt(f"{date_str} {time_str}:00")
            if pd.isna(ts):
                continue
            if ts < start_dt:
                reached_start = True
                continue
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "title": _clean_text(title),
                    "content": _clean_text(title),
                    "source": "sina_deep",
                    "url": link.strip(),
                }
        )
        if reached_start:
            break
        _polite_sleep(page_sleep_seconds)

    return pd.DataFrame(rows)


def fetch_sse_announcements(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, timeout: int) -> pd.DataFrame:
    code = strip_exchange_prefix(symbol) or symbol[-6:]
    rows: List[Dict[str, Any]] = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
    }
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
                rows.append(
                    {
                        "date": item.get("SSEDATE"),
                        "title": item.get("TITLE"),
                        "content": item.get("TITLE"),
                        "source": "sse_announcement",
                        "url": url,
                        "query": item.get("BULLETIN_TYPE_DESC") or "上交所公告",
                    }
                )
        total = int(page_help.get("total") or len(rows))
        if not data or page * 25 >= total:
            break
        page += 1
        time.sleep(0.3)
    return pd.DataFrame(rows)


def fetch_cninfo_fulltext(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, timeout: int) -> pd.DataFrame:
    normalized = _normalize_symbol_value(symbol)
    name = STOCK_NAMES.get(normalized, strip_exchange_prefix(normalized) or normalized)
    rows: List[Dict[str, Any]] = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    }
    page = 1
    while True:
        response = requests.post(
            "http://www.cninfo.com.cn/new/fulltextSearch/full",
            data={
                "searchkey": name,
                "sdate": start_dt.strftime("%Y-%m-%d"),
                "edate": end_dt.strftime("%Y-%m-%d"),
                "isfulltext": "false",
                "sortName": "pubdate",
                "sortType": "desc",
                "pageNum": str(page),
            },
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        announcements = payload.get("announcements") or []
        for item in announcements:
            if str(item.get("secCode") or "") != normalized[-6:]:
                continue
            ts = pd.to_datetime(item.get("announcementTime"), unit="ms", errors="coerce")
            if pd.isna(ts) or ts < start_dt or ts > end_dt:
                continue
            url = str(item.get("adjunctUrl") or "")
            if url and not url.startswith("http"):
                url = "http://static.cninfo.com.cn/" + url.lstrip("/")
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "title": item.get("announcementTitle") or item.get("shortTitle"),
                    "content": item.get("announcementTitle") or item.get("shortTitle"),
                    "source": "cninfo_fulltext",
                    "url": url,
                    "query": name,
                }
            )
        total = int(payload.get("totalRecordNum") or len(rows))
        if not announcements or page * 10 >= total:
            break
        page += 1
        time.sleep(0.3)
    return pd.DataFrame(rows)


def _date_chunks(start_dt: pd.Timestamp, end_dt: pd.Timestamp, chunk_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk_days <= 0:
        return [(start_dt, end_dt)]
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days) - pd.Timedelta(seconds=1), end_dt)
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(seconds=1)
    return chunks


def _time_chunks(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    chunk_days: int,
    chunk_hours: Optional[int] = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk_hours is not None and chunk_hours > 0:
        chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = start_dt
        delta = pd.Timedelta(hours=chunk_hours)
        while cur <= end_dt:
            chunk_end = min(cur + delta - pd.Timedelta(seconds=1), end_dt)
            chunks.append((cur, chunk_end))
            cur = chunk_end + pd.Timedelta(seconds=1)
        return chunks
    return _date_chunks(start_dt, end_dt, chunk_days)


def _tushare_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    def first_present(*keys: str) -> Any:
        for key in keys:
            value = record.get(key)
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

    return {
        "title": _clean_text(first_present("title", "headline", "content")),
        "content": _clean_text(first_present("content", "summary", "title", "headline")),
        "publish_time": first_present("datetime", "time", "publish_time", "date"),
        "source": _clean_text(first_present("src", "source")),
        "url": str(first_present("url", "link") or "").strip(),
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


def fetch_tushare_news_rows(
    symbols: Sequence[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    sources: Sequence[str],
    chunk_days: int,
    chunk_hours: Optional[int] = None,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_seconds: float,
    search_time: str,
) -> List[Dict[str, str]]:
    if not symbols:
        return []
    try:
        pro = get_tushare_pro()
    except Exception as exc:
        print(f"tushare_news: init failed: {exc}", flush=True)
        return []

    rows: List[Dict[str, str]] = []
    chunks = _time_chunks(start_dt, end_dt, chunk_days=chunk_days, chunk_hours=chunk_hours)
    for source in sources:
        def fetch_chunk_frames(
            chunk_start: pd.Timestamp,
            chunk_end: pd.Timestamp,
            *,
            depth: int = 0,
        ) -> List[Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]]:
            label = f"tushare_news {source} {chunk_start} -> {chunk_end}"

            def fetcher(source=source, chunk_start=chunk_start, chunk_end=chunk_end):
                return pro.news(
                    src=source,
                    start_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df = _fetch_with_retry(
                label,
                fetcher,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            span_seconds = max(0, int((chunk_end - chunk_start).total_seconds()))
            if len(df) >= TUSHARE_NEWS_LIMIT_GUARD and span_seconds > TUSHARE_MIN_SPLIT_SECONDS:
                mid = chunk_start + pd.Timedelta(seconds=max(1, span_seconds // 2))
                left_end = min(mid, chunk_end)
                right_start = left_end + pd.Timedelta(seconds=1)
                if right_start <= chunk_end:
                    print(f"{label}: raw={len(df)} reached limit guard; split", flush=True)
                    return [
                        *fetch_chunk_frames(chunk_start, left_end, depth=depth + 1),
                        *fetch_chunk_frames(right_start, chunk_end, depth=depth + 1),
                    ]
            return [(df, chunk_start, chunk_end)]

        for chunk_start, chunk_end in chunks:
            for df, final_start, final_end in fetch_chunk_frames(chunk_start, chunk_end):
                label = f"tushare_news {source} {final_start} -> {final_end}"
                in_range = 0
                for record in df.to_dict("records"):
                    item = _tushare_fields(record)
                    if not item["title"]:
                        continue
                    ts = _parse_dt(item["publish_time"])
                    if pd.isna(ts) or ts < start_dt or ts > end_dt:
                        continue
                    for symbol, alias in _matched_symbols(item["title"], item["content"], symbols):
                        rows.append(
                            {
                                "symbol": symbol,
                                "title": item["title"],
                                "content": item["content"] or item["title"],
                                "publish_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                                "source": f"tushare:{source}",
                                "url": item["url"],
                                "query": alias,
                                "search_time": search_time,
                            }
                        )
                        in_range += 1
                print(f"{label}: raw={len(df)} matched={in_range}", flush=True)
                _polite_sleep(sleep_seconds)
    return rows


def fetch_akshare_stock_news(symbol: str) -> pd.DataFrame:
    """AkShare 东方财富个股新闻，通常可返回最近 100 条。"""
    import akshare as ak

    plain = strip_exchange_prefix(symbol) or symbol
    return ak.stock_news_em(symbol=plain)


def fetch_akshare_calendar_events(date_value: str) -> pd.DataFrame:
    """AkShare 东方财富股市日历-公司动态，按日期返回全市场事件。"""
    import akshare as ak

    return ak.stock_gsrl_gsdt_em(date=date_value)


def _frame_to_rows(
    source_name: str,
    symbol: str,
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    search_time: str,
) -> List[Dict[str, str]]:
    if df.empty:
        return []
    plain = strip_exchange_prefix(symbol) or symbol
    rows: List[Dict[str, str]] = []
    for record in df.to_dict("records"):
        if source_name == "eastmoney_notice":
            publish_time = record.get("notice_date") or record.get("display_time")
            art_code = record.get("art_code") or ""
            url = f"https://data.eastmoney.com/notices/detail/{art_code}/{plain}.html" if art_code else ""
            source = "东方财富公告"
        elif source_name == "eastmoney_search":
            publish_time = record.get("date") or record.get("publishTime") or record.get("showTime")
            url = record.get("url") or (f"http://finance.eastmoney.com/a/{record.get('code')}.html" if record.get("code") else "")
            source = record.get("mediaName") or "东方财富"
        elif source_name == "akshare_stock_news":
            publish_time = record.get("发布时间")
            url = record.get("新闻链接")
            source = record.get("文章来源") or "东方财富新闻(AkShare)"
        else:
            publish_time = record.get("date")
            url = record.get("url")
            source = record.get("source") or "新浪财经"

        ts = _parse_dt(publish_time)
        if pd.isna(ts) or ts < start_dt or ts > end_dt:
            continue
        item = _row(
            symbol=symbol,
            title=record.get("title") or record.get("新闻标题"),
            content=record.get("content") or record.get("新闻内容") or record.get("title") or record.get("新闻标题"),
            publish_time=ts,
            source=source,
            url=url,
            query=_clean_text(record.get("query")) or f"{plain} 最新消息",
            search_time=search_time,
        )
        if item:
            rows.append(item)
    return rows


def _calendar_events_to_rows(
    df: pd.DataFrame,
    allowed_symbols: set[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    search_time: str,
) -> List[Dict[str, str]]:
    if df.empty:
        return []
    rows: List[Dict[str, str]] = []
    for record in df.to_dict("records"):
        normalized = normalize_symbol(str(record.get("代码") or ""))
        if not normalized or normalized not in allowed_symbols:
            continue
        event_date = _parse_dt(record.get("交易日"))
        if pd.isna(event_date):
            continue
        publish_time = pd.Timestamp(f"{event_date.date()} 00:00:00")
        if publish_time < start_dt or publish_time > end_dt:
            continue
        short_name = _clean_text(record.get("简称") or "")
        event_type = _clean_text(record.get("事件类型") or record.get("事项类型") or "")
        detail = _clean_text(record.get("具体事项") or "")
        title_parts = [part for part in (short_name, event_type) if part]
        title = " ".join(title_parts) or detail[:80]
        item = _row(
            symbol=normalized,
            title=title,
            content=detail or title,
            publish_time=publish_time,
            source="东方财富公司动态(AkShare)",
            url="",
            query="股市日历公司动态",
            search_time=search_time,
        )
        if item:
            rows.append(item)
    return rows


def _read_existing_news(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except Exception:
            df = None
    if df is None:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    for col in NEWS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[NEWS_COLUMNS]


def _save_news(path: Path, old_df: pd.DataFrame, new_rows: List[Dict[str, str]]) -> pd.DataFrame:
    with file_transaction_lock(path, suffix=".news.lock"):
        current_old = old_df
        if path.exists() and path.stat().st_size > 0:
            for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
                try:
                    current_old = pd.read_csv(path, encoding=encoding)
                    break
                except Exception:
                    continue
        if not new_rows and not current_old.empty:
            print(f"new_rows=0; keep existing news cache unchanged: {path}", flush=True)
            return current_old
        new_df = pd.DataFrame(new_rows, columns=NEWS_COLUMNS)
        combined = pd.concat([current_old, new_df], ignore_index=True) if not current_old.empty else new_df
        if combined.empty:
            return combined
        combined["symbol"] = combined["symbol"].map(lambda value: normalize_symbol(str(value)) or str(value).upper())
        combined["title"] = combined["title"].map(_clean_text)
        combined["content"] = combined["content"].map(_clean_text)
        combined["source"] = combined["source"].map(_clean_text)
        combined["parsed_time"] = pd.to_datetime(combined["publish_time"], errors="coerce")
        combined = combined[combined["title"].astype(str).str.len() > 0]
        combined = combined[combined["parsed_time"].notna()]
        combined["_title_key"] = combined["title"].map(_normalize_title_key)
        combined["_date_key"] = combined["parsed_time"].dt.strftime("%Y-%m-%d")
        combined["_source_priority"] = combined["source"].map(_source_priority)
        combined = combined.sort_values(
            ["symbol", "_title_key", "_date_key", "_source_priority", "parsed_time"],
            ascending=[True, True, True, True, False],
        )
        combined = combined.drop_duplicates(subset=["symbol", "_title_key", "_date_key"], keep="first")
        combined = combined.sort_values(["parsed_time", "symbol"], ascending=[False, True])
        combined = combined.drop(columns=["parsed_time", "_title_key", "_date_key", "_source_priority"])
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        return combined


def prefetch_historical_news(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    output_path: Path,
    page_size: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
    max_retries: int = 3,
    retry_backoff_seconds: float = 3.0,
    use_akshare_stock_news: bool = True,
    use_akshare_calendar_events: bool = True,
    use_tushare_news: bool = True,
    tushare_sources: Optional[Sequence[str]] = None,
    tushare_chunk_days: int = 1,
    tushare_chunk_hours: Optional[int] = None,
    use_sina_deep: bool = True,
    sina_max_pages: Optional[int] = None,
    sina_page_sleep_seconds: float = 0.8,
    use_sse_announcements: bool = True,
    use_cninfo_fulltext: bool = True,
) -> pd.DataFrame:
    start_dt = pd.Timestamp(f"{start_date} 00:00:00")
    end_dt = pd.Timestamp(f"{end_date} 23:59:59")
    search_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, str]] = []

    normalized_symbols = [_normalize_symbol_value(symbol) for symbol in symbols]
    allowed_symbols = set(normalized_symbols)
    if use_tushare_news and normalized_symbols:
        tushare_rows = fetch_tushare_news_rows(
            normalized_symbols,
            start_dt,
            end_dt,
            sources=list(tushare_sources or TUSHARE_NEWS_SOURCES),
            chunk_days=tushare_chunk_days,
            chunk_hours=tushare_chunk_hours,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep_seconds=sleep_seconds,
            search_time=search_time,
        )
        rows.extend(tushare_rows)

    for normalized in normalized_symbols:
        print(f"=== {normalized} ===", flush=True)

        for source_name, fetcher in (
            ("eastmoney_notice", fetch_eastmoney_notice_page),
            ("eastmoney_search", fetch_eastmoney_search_page),
        ):
            previous_signatures = set()
            for page in range(1, max_pages + 1):
                df = _fetch_with_retry(
                    f"{source_name} {normalized} page {page}",
                    lambda normalized=normalized, page=page, fetcher=fetcher: fetcher(
                        normalized, page, page_size, timeout
                    ),
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                source_rows = _frame_to_rows(source_name, normalized, df, start_dt, end_dt, search_time)
                signatures = {(r["title"], r["publish_time"]) for r in source_rows}
                print(f"{source_name} page {page}: raw={len(df)} in_range={len(source_rows)}", flush=True)
                rows.extend(source_rows)
                _polite_sleep(sleep_seconds)
                if df.empty or (page > 1 and signatures and signatures == previous_signatures):
                    break
                previous_signatures = signatures

        if use_akshare_stock_news:
            ak_df = _fetch_with_retry(
                f"akshare_stock_news {normalized}",
                lambda normalized=normalized: fetch_akshare_stock_news(normalized),
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            ak_rows = _frame_to_rows("akshare_stock_news", normalized, ak_df, start_dt, end_dt, search_time)
            print(f"akshare_stock_news: raw={len(ak_df)} in_range={len(ak_rows)}", flush=True)
            rows.extend(ak_rows)
            _polite_sleep(sleep_seconds)

        if use_sina_deep:
            effective_sina_pages = int(sina_max_pages or max_pages)
            sina_label = "sina_deep"
            sina_fetcher = lambda normalized=normalized: fetch_sina_stock_news_deep(
                normalized,
                timeout,
                max_pages=effective_sina_pages,
                start_dt=start_dt,
                page_sleep_seconds=sina_page_sleep_seconds,
            )
        else:
            sina_label = "sina"
            sina_fetcher = lambda normalized=normalized: fetch_sina_stock_news(normalized, timeout)

        sina_df = _fetch_with_retry(
            f"{sina_label} {normalized}",
            sina_fetcher,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        sina_rows = _frame_to_rows(sina_label, normalized, sina_df, start_dt, end_dt, search_time)
        print(f"{sina_label}: raw={len(sina_df)} in_range={len(sina_rows)}", flush=True)
        rows.extend(sina_rows)
        _polite_sleep(sleep_seconds)

        if use_sse_announcements:
            sse_df = _fetch_with_retry(
                f"sse_announcement {normalized}",
                lambda normalized=normalized: fetch_sse_announcements(normalized, start_dt, end_dt, timeout),
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            sse_rows = _frame_to_rows("sse_announcement", normalized, sse_df, start_dt, end_dt, search_time)
            print(f"sse_announcement: raw={len(sse_df)} in_range={len(sse_rows)}", flush=True)
            rows.extend(sse_rows)
            _polite_sleep(sleep_seconds)

        if use_cninfo_fulltext:
            cninfo_df = _fetch_with_retry(
                f"cninfo_fulltext {normalized}",
                lambda normalized=normalized: fetch_cninfo_fulltext(normalized, start_dt, end_dt, timeout),
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            cninfo_rows = _frame_to_rows("cninfo_fulltext", normalized, cninfo_df, start_dt, end_dt, search_time)
            print(f"cninfo_fulltext: raw={len(cninfo_df)} in_range={len(cninfo_rows)}", flush=True)
            rows.extend(cninfo_rows)
            _polite_sleep(sleep_seconds)

    if use_akshare_calendar_events:
        for current_date in pd.date_range(start_dt.date(), end_dt.date(), freq="D"):
            if current_date.weekday() >= 5:
                print(
                    f"akshare_calendar_events {current_date.strftime('%Y%m%d')}: skipped weekend",
                    flush=True,
                )
                continue
            date_arg = current_date.strftime("%Y%m%d")
            calendar_df = _fetch_with_retry(
                f"akshare_calendar_events {date_arg}",
                lambda date_arg=date_arg: fetch_akshare_calendar_events(date_arg),
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            calendar_rows = _calendar_events_to_rows(calendar_df, allowed_symbols, start_dt, end_dt, search_time)
            print(f"akshare_calendar_events {date_arg}: raw={len(calendar_df)} in_range={len(calendar_rows)}", flush=True)
            rows.extend(calendar_rows)
            _polite_sleep(sleep_seconds)

    old_df = _read_existing_news(output_path)
    combined = _save_news(output_path, old_df, rows)
    scoped = combined.copy()
    scoped["parsed_time"] = pd.to_datetime(scoped["publish_time"], errors="coerce")
    scoped = scoped[(scoped["parsed_time"] >= start_dt) & (scoped["parsed_time"] <= end_dt)]
    print(f"saved_rows={len(combined)} path={output_path}")
    print(f"in_range_rows={len(scoped)} start={start_dt} end={end_dt}")
    if not scoped.empty:
        print(scoped.groupby("symbol").size().sort_index().to_string())
    return scoped


def main() -> None:
    config = _load_config()
    date_range = config.get("date_range", {})
    parser = argparse.ArgumentParser(description="Prefetch historical stock news into data_flow/news.csv")
    parser.add_argument("--start-date", default=date_range.get("init_date"), help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=date_range.get("end_date"), help="YYYY-MM-DD")
    parser.add_argument("--output", default=(BASE_DIR / "data_flow" / "news.csv").as_posix())
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_STOCK_SYMBOLS)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds per request")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count after timeout/failure")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=3.0,
        help="Base seconds to wait before retry; doubles each attempt with jitter",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep after every source request")
    parser.add_argument("--no-akshare-stock-news", action="store_true", help="Disable AkShare stock_news_em source")
    parser.add_argument("--no-akshare-calendar-events", action="store_true", help="Disable AkShare stock_gsrl_gsdt_em source")
    parser.add_argument("--no-tushare-news", action="store_true", help="Disable Tushare pro.news source")
    parser.add_argument("--tushare-sources", nargs="+", default=TUSHARE_NEWS_SOURCES)
    parser.add_argument("--tushare-chunk-days", type=int, default=1)
    parser.add_argument("--tushare-chunk-hours", type=int, default=0)
    parser.add_argument("--no-sina-deep", action="store_true", help="Use only Sina first page instead of deep pagination")
    parser.add_argument("--sina-max-pages", type=int, default=None, help="Override Sina deep pagination pages")
    parser.add_argument("--sina-page-sleep", type=float, default=0.8, help="Seconds to sleep between Sina deep pages")
    parser.add_argument("--no-sse-announcements", action="store_true", help="Disable SSE official disclosures")
    parser.add_argument("--no-cninfo-fulltext", action="store_true", help="Disable CNINFO full-text disclosure search")
    args = parser.parse_args()

    if not args.start_date or not args.end_date:
        raise ValueError("start-date/end-date is required")
    prefetch_historical_news(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=Path(args.output),
        page_size=args.page_size,
        max_pages=args.max_pages,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff,
        use_akshare_stock_news=not args.no_akshare_stock_news,
        use_akshare_calendar_events=not args.no_akshare_calendar_events,
        use_tushare_news=not args.no_tushare_news,
        tushare_sources=args.tushare_sources,
        tushare_chunk_days=args.tushare_chunk_days,
        tushare_chunk_hours=args.tushare_chunk_hours,
        use_sina_deep=not args.no_sina_deep,
        sina_max_pages=args.sina_max_pages,
        sina_page_sleep_seconds=args.sina_page_sleep,
        use_sse_announcements=not args.no_sse_announcements,
        use_cninfo_fulltext=not args.no_cninfo_fulltext,
    )


if __name__ == "__main__":
    main()
