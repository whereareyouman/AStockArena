from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests


EASTMONEY_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"


def _parse_json_or_jsonp(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        raise json.JSONDecodeError("Empty response", s, 0)

    if s[0] in "{[":
        return json.loads(s)

    s = s.rstrip().rstrip(";").strip()

    lpar = s.find("(")
    rpar = s.rfind(")")
    if lpar == -1 or rpar == -1 or rpar <= lpar:
        return json.loads(s)

    payload = s[lpar + 1 : rpar].strip()
    if not payload:
        raise json.JSONDecodeError("Empty JSONP payload", s, lpar + 1)
    return json.loads(payload)


def _default_headers(symbol: str) -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": f"https://so.eastmoney.com/news/s?keyword={symbol}",
        "Origin": "https://so.eastmoney.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def stock_notice_em(
    symbol: str,
    *,
    page_index: int = 1,
    page_size: int = 10,
    timeout: int = 30
) -> pd.DataFrame:
    """
    东方财富-个股公告（np-anotice-stock 接口）。
    作为 stock_news_em_safe 的主要数据源。
    """
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "page_size": page_size,
        "page_index": page_index,
        "ann_type": "A",
        "client_source": "web",
        "stock_list": symbol,  # 6位代码，无需市场前缀
        "f_node": 1,
        "s_node": 1
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"http://quote.eastmoney.com/{symbol}.html"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        data = r.json()
        
        items = []
        if data.get("data_pipeline") and data["data_pipeline"].get("list"):
            items = data["data_pipeline"]["list"]
            
        required_cols = ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
        if not items:
            return pd.DataFrame(columns=required_cols)
            
        df = pd.DataFrame(items)
        
        # 映射字段：title -> 新闻标题, notice_date -> 发布时间
        # content 暂无，用 title 填充
        # url 构造: https://data.eastmoney.com/notices/detail/{art_code}/{codes}.html
        
        df["关键词"] = symbol
        df["新闻标题"] = df["title"]
        df["新闻内容"] = df["title"]  # 公告通常只有标题，内容需要点进去
        df["发布时间"] = df["notice_date"]
        df["文章来源"] = "东方财富公告"
        
        # 构造 URL
        # art_code: AN202412291612345678
        # codes: 688981
        def make_url(row):
            art_code = row.get("art_code", "")
            return f"https://data.eastmoney.com/notices/detail/{art_code}/{symbol}.html"
            
        df["新闻链接"] = df.apply(make_url, axis=1)
        
        # 填充缺失列
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
                
        return df[required_cols]
        
    except Exception as e:
        print(f"⚠️ 获取公告失败: {e}")
        return pd.DataFrame(columns=["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"])


import re

def stock_news_sina(symbol: str) -> pd.DataFrame:
    """
    新浪财经-个股新闻（HTML Parsing）。
    作为 stock_news_em_safe 的补充数据源。
    URL: https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=sh688981
    """
    market_symbol = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
    url = f"https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol={market_symbol}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    required_cols = ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = 'gb2312' # 新浪通常用 gb2312
        html = r.text
        
        # 提取新闻块：通常在 class="datelist" 里
        # 格式: &nbsp;&nbsp;&nbsp;&nbsp;2025-12-29&nbsp;22:46&nbsp;&nbsp;<a target='_blank' href='...'>标题</a>
        
        # 正则表达式说明：
        # (\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2}) : 提取日期和时间
        # .*?<a[^>]*href=['"]([^'"]+)['"][^>]*> : 提取链接
        # ([^<]+)</a> : 提取标题
        
        pattern = r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2}).*?<a[^>]*href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>"
        matches = re.findall(pattern, html)
        
        items = []
        for date_str, time_str, link, title in matches:
            # 过滤掉非新闻链接（如广告或无关链接）
            if "sina.com.cn" not in link and "sina.cn" not in link:
                continue
                
            items.append({
                "关键词": symbol,
                "新闻标题": title.strip(),
                "新闻内容": title.strip(), # 暂时用标题填充内容
                "发布时间": f"{date_str} {time_str}:00",
                "文章来源": "新浪财经",
                "新闻链接": link
            })
            
        if not items:
            return pd.DataFrame(columns=required_cols)
            
        return pd.DataFrame(items)
        
    except Exception as e:
        # print(f"⚠️ 获取新浪新闻失败: {e}")
        return pd.DataFrame(columns=required_cols)


def stock_news_em_safe(
    symbol: str,
    *,
    page_index: int = 1,
    page_size: int = 10,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
    url: str = EASTMONEY_SEARCH_URL,
) -> pd.DataFrame:
    """
    东方财富-个股新闻-最近新闻（项目内稳健实现）。
    
    策略变更：
    1. 优先尝试获取官方公告 (stock_notice_em)，因为该接口更稳定且价值高。
    2. 尝试获取新浪财经新闻 (stock_news_sina)，作为媒体新闻补充。
    3. 如果上述都为空，尝试原有的 search/jsonp 接口（兜底）。
    4. 合并所有结果并按时间排序。
    
    返回列名与 akshare.stock_news_em 对齐：
    ["关键词","新闻标题","新闻内容","发布时间","文章来源","新闻链接"]
    """
    
    # 1. 获取公告
    df_notice = stock_notice_em(symbol, page_index=page_index, page_size=page_size, timeout=timeout)
    
    # 2. 获取新浪新闻
    df_sina = stock_news_sina(symbol)
    
    # 3. 获取原有新闻接口 (search/jsonp)
    df_news = pd.DataFrame()
    try:
        inner_param: Dict[str, Any] = {
            "uid": "",
            "keyword": symbol,
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
        params = {
            "cb": cb,
            "param": json.dumps(inner_param, ensure_ascii=False),
            "_": str(ts_ms + random.randint(0, 999)),
        }

        r = requests.get(url, params=params, headers=headers or _default_headers(symbol), timeout=timeout)
        data_json = _parse_json_or_jsonp(r.text)

        def _extract_article_list(result_obj: Any) -> Optional[list]:
            if not isinstance(result_obj, dict):
                return None
            for key in ("cmsArticleWebOld", "cmsArticleWeb"):
                v = result_obj.get(key)
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    for sub_key in ("data_pipeline", "list", "items"):
                        sub_v = v.get(sub_key)
                        if isinstance(sub_v, list):
                            return sub_v
            return None

        items = None
        if isinstance(data_json, dict):
            items = _extract_article_list(data_json.get("result"))

        required_cols = ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
        
        if items:
            temp_df = pd.DataFrame(items)
            if "code" in temp_df.columns:
                temp_df["url"] = "http://finance.eastmoney.com/a/" + temp_df["code"].astype(str) + ".html"
            else:
                temp_df["url"] = ""

            temp_df.rename(
                columns={
                    "date": "发布时间",
                    "mediaName": "文章来源",
                    "title": "新闻标题",
                    "content": "新闻内容",
                    "url": "新闻链接",
                },
                inplace=True,
            )
            temp_df["关键词"] = symbol

            for col in required_cols:
                if col not in temp_df.columns:
                    temp_df[col] = ""

            temp_df = temp_df[required_cols]
            for col in ["新闻标题", "新闻内容"]:
                temp_df[col] = (
                    temp_df[col]
                    .astype(str)
                    .str.replace(r"\(<em>", "", regex=True)
                    .str.replace(r"</em>\)", "", regex=True)
                    .str.replace(r"<em>", "", regex=True)
                    .str.replace(r"</em>", "", regex=True)
                    .str.replace(r"\u3000", "", regex=True)
                    .str.replace(r"\r\n", " ", regex=True)
                )
            df_news = temp_df
            
    except Exception as e:
        # print(f"⚠️ search/jsonp 接口失败: {e}")
        pass

    # 4. 合并结果
    frames = []
    if not df_notice.empty:
        frames.append(df_notice)
    if not df_sina.empty:
        frames.append(df_sina)
    if not df_news.empty:
        frames.append(df_news)
        
    if not frames:
        return pd.DataFrame(columns=["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"])
        
    final_df = pd.concat(frames, ignore_index=True)
    
    # 去重 (按标题和发布时间)
    final_df.drop_duplicates(subset=["新闻标题", "发布时间"], keep="first", inplace=True)
    
    # 按时间倒序
    if "发布时间" in final_df.columns:
        final_df.sort_values(by="发布时间", ascending=False, inplace=True)
        
    # 限制返回数量 (比如最多返回 20 条，避免 token 消耗过大)
    return final_df.head(20)


