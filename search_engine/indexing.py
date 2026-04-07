from __future__ import annotations

import os
import shutil
import time
from datetime import date, datetime

import pandas as pd
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import BOOLEAN, DATETIME, ID, NUMERIC, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import And, DateRange, Every, Term

from .config import INDEX_DIR, SEARCH_FIELDS
from .data_utils import parse_dt


def build_schema() -> Schema:
    return Schema(
        comment_id=ID(stored=True, unique=True),
        parent_id=ID(stored=True),
        video_id=ID(stored=True),
        video_url=ID(stored=True),
        bucket=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        family=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        variant=ID(stored=True),
        comment_category=ID(stored=True),
        video_title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        channel_title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        search_query=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        aspects=ID(stored=True),
        like_count=NUMERIC(stored=True, sortable=True),
        relevance_score=NUMERIC(stored=True, sortable=True),
        is_reply=BOOLEAN(stored=True),
        published_at=DATETIME(stored=True, sortable=True),
        suggested_sentiment_label=ID(stored=True),
    )


def build_index(df: pd.DataFrame, force_rebuild: bool = False):
    if force_rebuild and os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)

    schema = build_schema()
    if index.exists_in(INDEX_DIR):
        ix = index.open_dir(INDEX_DIR)
    else:
        ix = index.create_in(INDEX_DIR, schema)

    writer = ix.writer(limitmb=256, multisegment=True)
    writer.commit(mergetype=None)

    shutil.rmtree(INDEX_DIR)
    os.mkdir(INDEX_DIR)
    ix = index.create_in(INDEX_DIR, schema)
    writer = ix.writer(limitmb=256, multisegment=True)

    for _, row in df.iterrows():
        writer.add_document(
            comment_id=str(row.get("comment_id", "")),
            parent_id=str(row.get("parent_id", "")),
            video_id=str(row.get("video_id", "")),
            video_url=str(row.get("video_url", "")),
            bucket=str(row.get("bucket", "")),
            family=str(row.get("family", "")),
            variant=str(row.get("variant", "")),
            comment_category=str(row.get("comment_category", "")),
            video_title=str(row.get("video_title", "")),
            channel_title=str(row.get("channel_title", "")),
            search_query=str(row.get("search_query", "")),
            text=str(row.get("text", "")),
            aspects=str(row.get("aspects", "general")),
            like_count=int(row.get("like_count", 0)),
            relevance_score=int(row.get("relevance_score", 0)),
            is_reply=bool(row.get("is_reply", False)),
            published_at=parse_dt(str(row.get("published_at", ""))),
            suggested_sentiment_label=str(row.get("suggested_sentiment_label", "neutral")),
        )

    writer.commit()
    return ix


def get_index(df: pd.DataFrame, force_rebuild: bool = False):
    if force_rebuild or not index.exists_in(INDEX_DIR):
        return build_index(df, force_rebuild=force_rebuild)
    ix = index.open_dir(INDEX_DIR)
    if "parent_id" not in ix.schema.names():
        return build_index(df, force_rebuild=True)
    return ix


def search(
    ix,
    query_text: str,
    limit: int,
    category: str = "All",
    sentiment: str = "All",
    include_replies: bool = True,
    enable_date_filter: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
):
    start = time.perf_counter()
    with ix.searcher() as searcher:
        parser = MultifieldParser(SEARCH_FIELDS, schema=ix.schema, group=OrGroup.factory(0.9))
        base_query = Every() if not query_text.strip() else parser.parse(query_text.strip())

        query_parts = [base_query]
        if category != "All":
            query_parts.append(Term("comment_category", str(category)))
        if sentiment != "All":
            query_parts.append(Term("suggested_sentiment_label", str(sentiment)))
        if not include_replies:
            query_parts.append(Term("is_reply", False))
        if enable_date_filter and start_date and end_date:
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            query_parts.append(
                DateRange(
                    "published_at",
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                )
            )

        query = query_parts[0] if len(query_parts) == 1 else And(query_parts)
        hits = searcher.search(query, limit=limit)

        rows = []
        for hit in hits:
            rows.append(
                {
                    "score": float(hit.score),
                    "comment_id": hit.get("comment_id"),
                    "parent_id": hit.get("parent_id"),
                    "video_id": hit.get("video_id"),
                    "video_url": hit.get("video_url"),
                    "bucket": hit.get("bucket"),
                    "family": hit.get("family"),
                    "variant": hit.get("variant"),
                    "comment_category": hit.get("comment_category"),
                    "video_title": hit.get("video_title"),
                    "channel_title": hit.get("channel_title"),
                    "search_query": hit.get("search_query"),
                    "text": hit.get("text"),
                    "aspects": hit.get("aspects"),
                    "like_count": hit.get("like_count"),
                    "relevance_score": hit.get("relevance_score"),
                    "is_reply": hit.get("is_reply"),
                    "published_at": hit.get("published_at"),
                    "suggested_sentiment_label": hit.get("suggested_sentiment_label"),
                }
            )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return pd.DataFrame(rows), latency_ms
