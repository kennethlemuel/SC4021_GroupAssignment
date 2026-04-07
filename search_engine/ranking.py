from __future__ import annotations

import math
from datetime import date

import pandas as pd

from .config import RANK_WEIGHTS


def minmax_norm(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(numeric.min())
    hi = float(numeric.max())
    if hi <= lo:
        return pd.Series([0.0] * len(numeric), index=numeric.index, dtype=float)
    return (numeric - lo) / (hi - lo)


def apply_fixed_reranking(df: pd.DataFrame, query_text: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    text_component = minmax_norm(out.get("score", 0.0))
    relevance_component = minmax_norm(out.get("relevance_score", 0.0))

    likes = pd.to_numeric(out.get("like_count", 0), errors="coerce").fillna(0).astype(float)
    engagement_component = minmax_norm(likes.map(lambda v: math.log1p(v)))

    is_reply = out.get("is_reply", False)
    is_reply = pd.Series(is_reply, index=out.index).astype(bool)
    quality_component = (~is_reply).astype(float)

    out["weighted_score"] = (
        RANK_WEIGHTS["text_relevance"] * text_component
        + RANK_WEIGHTS["relevance_label"] * relevance_component
        + RANK_WEIGHTS["engagement"] * engagement_component
        + RANK_WEIGHTS["quality"] * quality_component
    )

    out = out.sort_values(
        by=["weighted_score", "score", "relevance_score", "like_count"],
        ascending=False,
    )
    return out


def apply_filters(
    df: pd.DataFrame,
    family: str,
    bucket: str,
    category: str,
    sentiment: str,
    aspect: str,
    include_replies: bool,
    enable_date_filter: bool,
    start_date: date | None,
    end_date: date | None,
    query_text: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    if family != "All":
        mask &= df["family"].eq(family)
    if bucket != "All":
        mask &= df["bucket"].eq(bucket)
    if category != "All":
        mask &= df["comment_category"].eq(category)
    if sentiment != "All":
        mask &= df["suggested_sentiment_label"].eq(sentiment)
    if aspect != "All":
        mask &= df["aspects"].astype(str).str.contains(aspect, na=False)
    if not include_replies:
        mask &= ~df["is_reply"].astype(bool)

    if enable_date_filter and start_date and end_date:
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        published = pd.to_datetime(df.get("published_at"), errors="coerce")
        mask &= (published.dt.date >= start_date) & (published.dt.date <= end_date)

    out = df.loc[mask]
    return apply_fixed_reranking(out, query_text)


def get_brand_model_mappings(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    pairs = (
        df[["family", "bucket"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    bucket_to_family: dict[str, str] = {}
    family_to_buckets: dict[str, list[str]] = {}
    ambiguous_buckets: list[str] = []

    for bucket, group in pairs.groupby("bucket"):
        families = sorted(group["family"].unique().tolist())
        if len(families) == 1:
            bucket_to_family[bucket] = families[0]
        else:
            ambiguous_buckets.append(bucket)

    for family, group in pairs.groupby("family"):
        family_to_buckets[family] = sorted(group["bucket"].unique().tolist())

    return bucket_to_family, family_to_buckets, ambiguous_buckets
