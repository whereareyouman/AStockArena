from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


NEWS_MANIFEST_FILENAME = "news_manifest.json"


class NewsCacheIntegrityError(RuntimeError):
    """Raised when news.csv no longer matches its manifest."""


def default_news_manifest_path(news_csv_path: Path) -> Path:
    return Path(news_csv_path).parent / NEWS_MANIFEST_FILENAME


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise NewsCacheIntegrityError(f"无法读取新闻缓存 {path}: {last_error}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_news_manifest(news_csv_path: Path, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    path = Path(news_csv_path)
    if df is None:
        df = _read_csv_with_fallback(path) if path.exists() else pd.DataFrame()

    manifest: Dict[str, Any] = {
        "path": str(path),
        "rows": int(len(df)),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256": _sha256(path) if path.exists() else "",
    }

    if not df.empty:
        scoped = df.copy()
        if "publish_time" in scoped.columns:
            parsed_time = pd.to_datetime(scoped["publish_time"], errors="coerce")
            parsed_time = parsed_time.dropna()
            if not parsed_time.empty:
                manifest["date_start"] = parsed_time.min().strftime("%Y-%m-%d %H:%M:%S")
                manifest["date_end"] = parsed_time.max().strftime("%Y-%m-%d %H:%M:%S")
        if "symbol" in scoped.columns:
            manifest["by_symbol"] = {
                str(key): int(value)
                for key, value in scoped["symbol"].astype(str).value_counts().sort_index().items()
            }
        if "source" in scoped.columns:
            manifest["by_source"] = {
                str(key): int(value)
                for key, value in scoped["source"].astype(str).value_counts().items()
            }
        if {"symbol", "title", "publish_time"}.issubset(scoped.columns):
            manifest["duplicate_key_count"] = int(
                scoped.duplicated(subset=["symbol", "title", "publish_time"]).sum()
            )
        if "title" in scoped.columns:
            manifest["empty_title_count"] = int(scoped["title"].fillna("").astype(str).str.len().eq(0).sum())
        if "content" in scoped.columns:
            manifest["empty_content_count"] = int(scoped["content"].fillna("").astype(str).str.len().eq(0).sum())
    return manifest


def write_news_manifest(
    news_csv_path: Path,
    df: Optional[pd.DataFrame] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    csv_path = Path(news_csv_path)
    target = Path(manifest_path) if manifest_path is not None else default_news_manifest_path(csv_path)
    manifest = build_news_manifest(csv_path, df)
    manifest["path"] = str(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def read_news_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise NewsCacheIntegrityError(f"无法读取新闻 manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NewsCacheIntegrityError(f"新闻 manifest 格式错误: {path}")
    return payload


def validate_news_cache_integrity(
    news_csv_path: Path,
    manifest_path: Optional[Path] = None,
    *,
    strict: bool = False,
    verify_hash: bool = True,
) -> Dict[str, Any]:
    csv_path = Path(news_csv_path)
    manifest_file = Path(manifest_path) if manifest_path is not None else default_news_manifest_path(csv_path)
    manifest = read_news_manifest(manifest_file)
    result: Dict[str, Any] = {
        "ok": True,
        "news_csv_path": str(csv_path),
        "manifest_path": str(manifest_file),
        "manifest_exists": manifest is not None,
        "errors": [],
    }

    if not csv_path.exists():
        result["ok"] = False
        result["errors"].append(f"news.csv 不存在: {csv_path}")
        if strict:
            raise NewsCacheIntegrityError("; ".join(result["errors"]))
        return result

    result["size_bytes"] = int(csv_path.stat().st_size)
    try:
        df = _read_csv_with_fallback(csv_path)
        result["rows"] = int(len(df))
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(str(exc))
        if strict:
            raise NewsCacheIntegrityError("; ".join(result["errors"]))
        return result

    if manifest:
        expected_rows = manifest.get("rows")
        expected_size = manifest.get("size_bytes")
        expected_sha = manifest.get("sha256")
        result["expected_rows"] = expected_rows
        result["expected_size_bytes"] = expected_size
        result["expected_sha256"] = expected_sha
        if isinstance(expected_rows, int) and expected_rows > 0 and result["rows"] == 0:
            result["ok"] = False
            result["errors"].append(
                f"news.csv 只有表头/空数据，但 manifest 记录应有 {expected_rows} 行"
            )
        if isinstance(expected_rows, int) and expected_rows != result["rows"]:
            result["ok"] = False
            result["errors"].append(f"news.csv 行数 {result['rows']} 与 manifest {expected_rows} 不一致")
        if isinstance(expected_size, int) and expected_size != result["size_bytes"]:
            result["ok"] = False
            result["errors"].append(
                f"news.csv 大小 {result['size_bytes']} 与 manifest {expected_size} 不一致"
            )
        if verify_hash and expected_sha:
            actual_sha = _sha256(csv_path)
            result["sha256"] = actual_sha
            if actual_sha != expected_sha:
                result["ok"] = False
                result["errors"].append("news.csv sha256 与 manifest 不一致")
    elif result["rows"] == 0:
        result["warnings"] = ["news.csv 当前为空；没有 manifest 可用于判断是否被覆盖"]

    if strict and not result["ok"]:
        raise NewsCacheIntegrityError("; ".join(result["errors"]))
    return result
