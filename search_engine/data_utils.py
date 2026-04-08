from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ASPECT_PATTERNS


def parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1)


def detect_aspects(text: str) -> str:
    lowered = (text or "").casefold()
    found: list[str] = []
    for aspect, keywords in ASPECT_PATTERNS.items():
        if any(keyword in lowered for keyword in keywords):
            found.append(aspect)
    return "|".join(found) if found else "general"


def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["like_count"] = pd.to_numeric(out.get("like_count", 0), errors="coerce").fillna(0).astype(int)
    out["relevance_score"] = pd.to_numeric(out.get("relevance_score", 0), errors="coerce").fillna(0).astype(int)
    out["is_reply"] = out.get("is_reply", False).astype(str).str.lower().eq("true")

    if "suggested_sentiment_label" not in out.columns:
        out["suggested_sentiment_label"] = "neutral"
    else:
        out["suggested_sentiment_label"] = out["suggested_sentiment_label"].fillna("neutral").astype(str)

    out["published_at"] = pd.to_datetime(out.get("published_at"), errors="coerce")
    out["aspects"] = out["text"].astype(str).apply(detect_aspects)
    return out


def merge_final_sentiment(df: pd.DataFrame, sentiment_csv_path: str) -> pd.DataFrame:
    out = df.copy()
    path = Path(sentiment_csv_path)
    if not path.exists():
        return out

    try:
        sentiment_df = pd.read_csv(path, usecols=["comment_id", "final_sentiment"])
    except Exception:
        return out

    if sentiment_df.empty or "comment_id" not in out.columns:
        return out

    sentiment_map = (
        sentiment_df.dropna(subset=["comment_id"])
        .assign(comment_id=lambda d: d["comment_id"].astype(str))
        .drop_duplicates(subset=["comment_id"], keep="last")
        .set_index("comment_id")["final_sentiment"]
    )

    existing = out.get("suggested_sentiment_label", pd.Series([None] * len(out), index=out.index, dtype=object))
    existing = existing.astype(object)
    merged = out["comment_id"].astype(str).map(sentiment_map)
    out["suggested_sentiment_label"] = merged.combine_first(existing)
    return out
