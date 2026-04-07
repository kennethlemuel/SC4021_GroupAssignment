from __future__ import annotations

import os
import re
import shutil
import time
import math
import html
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import BOOLEAN, DATETIME, ID, NUMERIC, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import And, DateRange, Every, Term

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

DATA_PATH = "data/comments_relevant.csv"
INDEX_DIR = "indexdir"
SEARCH_FIELDS = ["text", "bucket", "family", "comment_category", "video_title", "search_query"]
SEARCH_LIMIT = 10000
COMMENTS_PAGE_SIZE = 20
RESULTS_PAGE_SIZE = 10


ASPECT_PATTERNS = {
    "battery": ("battery", "charging", "mah", "magsafe"),
    "display": ("display", "screen", "brightness", "hz", "refresh"),
    "camera": ("camera", "photo", "video", "zoom", "sensor"),
    "performance": ("performance", "chip", "cpu", "exynos", "snapdragon", "lag", "smooth"),
    "design": ("design", "size", "weight", "build", "ergonomic", "form factor", "titanium", "aluminum"),
    "price": ("price", "cost", "expensive", "cheap", "value", "msrp", "usd", "$"),
    "ai": ("ai", "artificial intelligence"),
}


# Fixed reranking weights (sum to 1.0) chosen for assignment-aligned opinion retrieval.
RANK_WEIGHTS = {
    "text_relevance": 0.55,
    "relevance_label": 0.20,
    "engagement": 0.15,
    "quality": 0.10,
}


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

    # Recreate fresh index each run of indexing action for deterministic behavior.
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
    limit: int = SEARCH_LIMIT,
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

    # Tie-break by BM25 score and useful metadata for deterministic ordering.
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


def bucket_pattern(bucket: str, all_buckets: list[str]) -> re.Pattern[str]:
    lowered = (bucket or "").casefold()
    parts = re.findall(r"[a-z0-9]+", lowered)
    if not parts:
        return re.compile(r"a^", flags=re.IGNORECASE)

    # Match model tokens with flexible separators, e.g. "iphone17pro" / "iphone 17 pro" / "iphone-17-pro".
    token_expr = r"\W*".join(re.escape(p) for p in parts)
    premium_suffixes = r"(pro|max|plus|ultra|mini|fold|flip|edge|fe)"

    # If this bucket is a base prefix of another model, block premium suffix continuation.
    is_base_prefix = any(
        other.casefold().startswith(lowered + " ") for other in all_buckets if other != bucket
    )
    if is_base_prefix:
        return re.compile(
            rf"\b{token_expr}\b(?!\W*{premium_suffixes}\b)",
            flags=re.IGNORECASE,
        )

    return re.compile(rf"\b{token_expr}\b", flags=re.IGNORECASE)


def infer_bucket_from_query(query_text: str, all_buckets: list[str]) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    # Prefer longer bucket names first (e.g., iPhone 17 Pro over iPhone 17).
    ordered = sorted(all_buckets, key=lambda x: len(x), reverse=True)
    matches: list[str] = []
    for bucket in ordered:
        pattern = bucket_pattern(bucket, all_buckets)
        if pattern.search(q):
            matches.append(bucket)

    # Strict model inference: if we detect a model phrase, lock query to the most specific match.
    if matches:
        return matches[0]
    return None


def infer_category_from_query(query_text: str) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    category_patterns: dict[str, tuple[str, ...]] = {
        "camera": (r"\bcamera\b", r"\bphoto\b", r"\bvideo\b", r"\bzoom\b", r"\bsensor\b"),
        "battery": (r"\bbattery\b", r"\bcharging\b", r"\bmah\b", r"\bmagsafe\b"),
        "issue": (
            r"\bissue\b",
            r"\bproblem\b",
            r"\boverheat(?:ing)?\b",
            r"\bbug\b",
            r"\bfault\b",
            r"\bdefect\b",
        ),
        "review": (r"\breview\b", r"\bimpression\b", r"\bthoughts\b", r"\bopinion\b"),
    }

    matches: list[str] = []
    for category, patterns in category_patterns.items():
        if any(re.search(p, q) for p in patterns):
            matches.append(category)

    if len(matches) == 1:
        return matches[0]
    return None


def infer_family_from_query(query_text: str, all_families: list[str]) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    matches: list[str] = []
    for family in all_families:
        # Include family alias cues for stronger brand intent detection.
        aliases = [family.casefold()]
        if family.casefold() == "apple":
            aliases.extend(["iphone", "ios"])
        elif family.casefold() == "samsung":
            aliases.extend(["galaxy"])
        elif family.casefold() == "google":
            aliases.extend(["pixel"])
        elif family.casefold() == "xiaomi":
            aliases.extend(["mi"])

        if any(re.search(rf"\b{re.escape(alias)}\b", q) for alias in aliases):
            matches.append(family)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def infer_unknown_model_from_query(
    query_text: str,
    all_buckets: list[str],
    all_families: list[str],
) -> str | None:
    q = (query_text or "").strip()
    if not q:
        return None

    # If we already matched a known model, the query is in-scope.
    if infer_bucket_from_query(q, all_buckets) is not None:
        return None

    inferred_family = infer_family_from_query(q, all_families)
    if inferred_family is None:
        return None

    lowered = q.casefold()

    # Model intent heuristic: recognized phone family + model-number token in query.
    has_model_number = bool(re.search(r"\b(?:\d{1,3}|[a-z]\d{1,3})\b", lowered))
    if not has_model_number:
        return None

    alias_map = {
        "apple": ["iphone", "ios", "apple"],
        "samsung": ["galaxy", "samsung"],
        "google": ["pixel", "google"],
        "xiaomi": ["xiaomi", "mi", "redmi"],
    }
    aliases = alias_map.get(inferred_family.casefold(), [inferred_family.casefold()])
    alias_expr = "|".join(re.escape(a) for a in aliases)

    candidate_patterns = [
        rf"\b(?:{alias_expr})\W*[a-z]?\W*\d{{1,3}}(?:\W*(?:pro|max|plus|ultra|mini|fold|flip|edge|fe))?\b",
        r"\b[a-z]\d{1,3}(?:\W*(?:pro|max|plus|ultra|mini|fold|flip|edge|fe))?\b",
    ]
    for pattern in candidate_patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return q[match.start():match.end()].strip()

    return q


def reset_filters() -> None:
    st.session_state["family_filter"] = "All"
    st.session_state["bucket_filter"] = "All"
    st.session_state["category_filter"] = "All"
    st.session_state["sentiment_filter"] = "All"
    st.session_state["aspect_filter"] = "All"
    st.session_state["enable_date_filter"] = False


def init_session_state() -> None:
    defaults = {
        "family_filter": "All",
        "bucket_filter": "All",
        "category_filter": "All",
        "sentiment_filter": "All",
        "aspect_filter": "All",
        "enable_date_filter": False,
        "start_date_filter": None,
        "end_date_filter": None,
        "active_query": "",
        "last_submitted_query": "",
        "query_mode": "all",
        "query_input": "",
        "clear_query_input_next_run": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def on_facet_change() -> None:
    # Facet edits take precedence over previously submitted text query.
    st.session_state["query_mode"] = "facet"
    st.session_state["active_query"] = ""
    st.session_state["last_submitted_query"] = ""
    st.session_state["clear_query_input_next_run"] = True


def sentiment_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No result rows for sentiment summary.")
        return
    chart_df = (
        df.assign(suggested_sentiment_label=df["suggested_sentiment_label"].astype(str).str.lower())
        .groupby("suggested_sentiment_label")
        .size()
        .reset_index(name="count")
    )
    sentiment_palette = {
        "positive": "#5f9f7a",
        "neutral": "#8b93a2",
        "negative": "#b66a72",
    }
    fig = px.pie(
        chart_df,
        names="suggested_sentiment_label",
        values="count",
        title="Sentiment Distribution",
        color="suggested_sentiment_label",
        color_discrete_map=sentiment_palette,
    )
    panel_height_px = 420
    fig.update_layout(height=panel_height_px)

    counts = sentiment_counts(df)
    total = sum(counts.values())
    ordered_labels = ["positive", "neutral", "negative"]
    dominant = max(ordered_labels, key=lambda label: counts.get(label, 0))

    indicator_style = {
        "positive": {"accent": "#5f9f7a", "badge_bg": "#1a3427", "icon": "↗"},
        "neutral": {"accent": "#8b93a2", "badge_bg": "#2b313d", "icon": "→"},
        "negative": {"accent": "#b66a72", "badge_bg": "#40242a", "icon": "↘"},
    }
    style = indicator_style[dominant]
    dominant_pct = (counts.get(dominant, 0) / total * 100.0) if total else 0.0

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown(
            f"""
                        <div style=\"background:transparent; border-radius:14px; padding:26px 16px; text-align:center; height:{panel_height_px}px; display:flex; flex-direction:column; justify-content:center; box-sizing:border-box;\">
              <div style=\"width:96px; height:96px; margin:0 auto 16px auto; border-radius:50%; background:{style['badge_bg']}; display:flex; align-items:center; justify-content:center;\">
                <span style=\"font-size:42px; color:{style['accent']}; line-height:1; font-weight:700;\">{style['icon']}</span>
              </div>
              <div style=\"font-size:34px; font-weight:800; color:{style['accent']}; line-height:1.1; margin-bottom:8px;\">{dominant.upper()}</div>
              <div style=\"font-size:13px; color:#a5b1c1; margin-bottom:10px;\">Platform sentiment analysis</div>
              <div style=\"font-size:12px; color:#7f8da3;\">{counts.get(dominant, 0)} comments ({dominant_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def category_chart(df: pd.DataFrame):
    if df.empty:
        return
    chart_df = df.groupby("comment_category").size().reset_index(name="count").sort_values("count", ascending=False)
    category_palette = ["#4c78a8", "#f58518", "#7a5195", "#2f4b7c", "#bc5090", "#ffa600", "#3b8ea5"]
    fig = px.bar(
        chart_df,
        x="comment_category",
        y="count",
        title="Category Distribution",
        color="comment_category",
        color_discrete_sequence=category_palette,
    )
    st.plotly_chart(fig, use_container_width=True)


def word_cloud_chart(df: pd.DataFrame):
    if df.empty:
        return
    if not HAS_WORDCLOUD:
        st.info("Install wordcloud and matplotlib to enable the word cloud visualization.")
        return

    all_text = " ".join(df.get("text", pd.Series([], dtype=str)).fillna("").astype(str).head(300).tolist())
    if not all_text.strip():
        return

    wc = WordCloud(
        width=1200,
        height=420,
        background_color="#12161e",
        max_words=100,
        colormap="viridis",
    ).generate(all_text)

    fig_wc, ax_wc = plt.subplots(figsize=(12, 4.2), facecolor="#12161e")
    ax_wc.set_facecolor("#12161e")
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title("Word Cloud of Search Results", color="#e8edf3")
    st.pyplot(fig_wc)
    plt.close(fig_wc)


def results_over_time_chart(df: pd.DataFrame):
    if df.empty:
        return

    dated = df.copy()
    dated["published_at"] = pd.to_datetime(dated.get("published_at"), errors="coerce")
    dated = dated.dropna(subset=["published_at"])
    if dated.empty:
        return

    dated["date"] = dated["published_at"].dt.date
    timeline_df = dated.groupby("date").size().reset_index(name="count")

    fig_timeline = px.line(
        timeline_df,
        x="date",
        y="count",
        title="Results Over Time",
        labels={"date": "Date", "count": "Number of Comments"},
    )
    fig_timeline.update_layout(height=320)
    st.plotly_chart(fig_timeline, use_container_width=True)


def sentiment_counts(df: pd.DataFrame) -> dict[str, int]:
    labels = ["positive", "neutral", "negative"]
    if df.empty:
        return {label: 0 for label in labels}
    series = df.get("suggested_sentiment_label", pd.Series([], dtype=str)).fillna("neutral").astype(str).str.lower()
    return {label: int((series == label).sum()) for label in labels}


def render_sentiment_comparison(
    ix,
    query_a: str,
    query_b: str,
    family: str,
    bucket: str,
    category: str,
    sentiment: str,
    aspect: str,
    include_replies: bool,
    enable_date_filter: bool,
    start_date: date | None,
    end_date: date | None,
    all_buckets: list[str],
    all_families: list[str],
):
    unknown_a = infer_unknown_model_from_query(query_a, all_buckets, all_families)
    unknown_b = infer_unknown_model_from_query(query_b, all_buckets, all_families)
    if unknown_a or unknown_b:
        messages: list[str] = []
        if unknown_a:
            messages.append(f"Query A is outside dataset scope (unknown model: {unknown_a}).")
        if unknown_b:
            messages.append(f"Query B is outside dataset scope (unknown model: {unknown_b}).")
        messages.append("This dataset only supports comparisons for phone models present in the indexed records.")
        st.warning(" ".join(messages))
        return

    q1_df, _ = search(
        ix,
        query_a,
        limit=SEARCH_LIMIT,
        category=category,
        sentiment=sentiment,
        include_replies=include_replies,
        enable_date_filter=enable_date_filter,
        start_date=start_date,
        end_date=end_date,
    )
    q2_df, _ = search(
        ix,
        query_b,
        limit=SEARCH_LIMIT,
        category=category,
        sentiment=sentiment,
        include_replies=include_replies,
        enable_date_filter=enable_date_filter,
        start_date=start_date,
        end_date=end_date,
    )

    q1_filtered = apply_filters(
        q1_df,
        family,
        bucket,
        category,
        sentiment,
        aspect,
        include_replies,
        enable_date_filter,
        start_date,
        end_date,
        query_a,
    )
    q2_filtered = apply_filters(
        q2_df,
        family,
        bucket,
        category,
        sentiment,
        aspect,
        include_replies,
        enable_date_filter,
        start_date,
        end_date,
        query_b,
    )

    q1_total = len(q1_filtered)
    q2_total = len(q2_filtered)
    if q1_total == 0 and q2_total == 0:
        st.warning("Both queries returned no results under current filters.")
        return

    q1_counts = sentiment_counts(q1_filtered)
    q2_counts = sentiment_counts(q2_filtered)

    query_a_label = (query_a or "").strip() or "Query A"
    query_b_label = (query_b or "").strip() or "Query B"
    if query_a_label == query_b_label:
        query_b_label = f"{query_b_label} (B)"

    labels = ["positive", "neutral", "negative"]
    compare_long = pd.DataFrame(
        [
            {"sentiment": label, "query": query_a_label, "count": q1_counts[label]}
            for label in labels
        ]
        + [
            {"sentiment": label, "query": query_b_label, "count": q2_counts[label]}
            for label in labels
        ]
    )

    fig_compare = px.bar(
        compare_long,
        x="sentiment",
        y="count",
        color="query",
        barmode="group",
        title="Sentiment Comparison",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    q1_pos = (q1_counts["positive"] / q1_total * 100.0) if q1_total else 0.0
    q2_pos = (q2_counts["positive"] / q2_total * 100.0) if q2_total else 0.0
    q1_neg = (q1_counts["negative"] / q1_total * 100.0) if q1_total else 0.0
    q2_neg = (q2_counts["negative"] / q2_total * 100.0) if q2_total else 0.0
    net_a = q1_pos - q1_neg
    net_b = q2_pos - q2_neg

    m1, m2, m3 = st.columns(3)
    m1.metric("Query A Results", q1_total)
    m2.metric("Query B Results", q2_total)
    m3.metric("Net Sentiment Delta (B - A)", f"{(net_b - net_a):.2f} pp")

    delta_df = pd.DataFrame(
        {
            "metric": ["Positive %", "Negative %", "Net Sentiment %"],
            query_a_label: [q1_pos, q1_neg, net_a],
            query_b_label: [q2_pos, q2_neg, net_b],
            "Difference (B - A)": [q2_pos - q1_pos, q2_neg - q1_neg, net_b - net_a],
        }
    )
    st.dataframe(delta_df, width="stretch", hide_index=True)


def format_comment_date(value) -> str:
    if pd.isna(value):
        return "Unknown date"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return "Unknown date"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "Unknown date"


def sentiment_style(label: str) -> tuple[str, str]:
    lowered = (label or "").casefold()
    if lowered == "positive":
        return "#123826", "#9cf7c6"
    if lowered == "negative":
        return "#4a1d24", "#ffb4bf"
    return "#2d3340", "#d6dbe5"


def render_comment_card(row: pd.Series, rank: int):
    text = html.escape(str(row.get("text", "")))
    model = html.escape(str(row.get("bucket", "")))
    published = format_comment_date(row.get("published_at"))
    likes = int(pd.to_numeric(row.get("like_count", 0), errors="coerce") or 0)
    weighted = float(pd.to_numeric(row.get("weighted_score", 0.0), errors="coerce") or 0.0)
    bm25 = float(pd.to_numeric(row.get("score", 0.0), errors="coerce") or 0.0)
    category = html.escape(str(row.get("category", "")))
    channel = html.escape(str(row.get("channel_title", "")))
    video_title = html.escape(str(row.get("video_title", "")))
    video_url_raw = str(row.get("video_url", "") or "").strip()
    video_url = html.escape(video_url_raw)
    query_hint = html.escape(str(row.get("search_query", "")))
    sentiment_raw = str(row.get("suggested_sentiment_label", "neutral"))
    sentiment = html.escape(sentiment_raw)
    chip_bg, chip_fg = sentiment_style(sentiment_raw)
    is_reply_value = bool(row.get("is_reply", False))
    message_type = "reply" if is_reply_value else "comment"
    type_chip_bg = "#143a5f" if is_reply_value else "#123826"
    type_chip_fg = "#a8d6ff" if is_reply_value else "#9cf7c6"

    title_line = video_title if video_title else model
    meta_parts = [
        f"{model}",
        f"📺 {channel}",
        f"{published}",
        f"category: {category}",
        f"weighted {weighted:.3f}",
        f"bm25 {bm25:.3f}",
        f"👍 {likes}",
    ]
    meta = " • ".join([part for part in meta_parts if part.strip()])

    if video_url_raw:
        title_html = f'<a href="{video_url}" target="_blank" style="color:#8ec8ff; text-decoration:none;">{title_line}</a>'
    else:
        title_html = f'<span style="color:#f2f5f9;">{title_line}</span>'

    st.markdown(
        f"""
        <div style=\"margin:8px 0 12px 0; padding:12px 14px; background:#12161e; border:1px solid #2b313d; border-radius:12px;\">
          <div style=\"display:flex; align-items:center; gap:8px; margin-bottom:6px;\">
            <span style=\"font-size:12px; color:#8fa0b7;\">#{rank}</span>
            <span style=\"font-size:16px; font-weight:600;\">{title_html}</span>
                        <span style=\"margin-left:auto; padding:2px 8px; border-radius:999px; background:{type_chip_bg}; color:{type_chip_fg}; font-size:12px;\">{message_type}</span>
                        <span style=\"padding:2px 8px; border-radius:999px; background:{chip_bg}; color:{chip_fg}; font-size:12px;\">{sentiment}</span>
          </div>
          <div style=\"font-size:12px; color:#9ca8ba; margin-bottom:8px;\">{meta}</div>
          <div style=\"font-size:14px; line-height:1.45; color:#e8edf3; white-space:pre-wrap; margin-bottom:8px;\">{text}</div>
          <div style=\"font-size:12px; color:#7f8da3;\">query seed: {query_hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_pagination(total_pages: int, current_page: int, page_state_key: str, widget_scope: str) -> int:
    if total_pages <= 1:
        st.markdown(
            "<div style='text-align:center; color:#9ca8ba; font-size:12px; margin:6px 0;'>Page 1 of 1</div>",
            unsafe_allow_html=True,
        )
        return current_page

    max_visible = 7
    half = max_visible // 2
    start_page = max(1, current_page - half)
    end_page = min(total_pages, start_page + max_visible - 1)
    if end_page - start_page + 1 < max_visible:
        start_page = max(1, end_page - max_visible + 1)

    display_tokens: list[str] = []
    if start_page > 1:
        display_tokens.append("1")
        if start_page > 2:
            display_tokens.append("...")
    for p in range(start_page, end_page + 1):
        display_tokens.append(str(p))
    if end_page < total_pages:
        if end_page < total_pages - 1:
            display_tokens.append("...")
        display_tokens.append(str(total_pages))

    token_widths: list[float] = []
    for token in display_tokens:
        if token == "...":
            token_widths.append(0.8)
        else:
            # Dynamically size each page slot by label length (e.g., "93" wider than "7").
            token_widths.append(max(1.0, 0.7 + 0.35 * len(token)))

    # Center a compact dynamic-width row: Prev | tokens | Next.
    prev_next_width = 1.8
    row_width = (2 * prev_next_width) + sum(token_widths)
    outer = st.columns([1.0, row_width, 1.0])
    with outer[1]:
        row_cols = st.columns([prev_next_width] + token_widths + [prev_next_width])

        with row_cols[0]:
            if st.button("Prev", key=f"{widget_scope}_prev", disabled=current_page <= 1, use_container_width=True):
                next_page = current_page - 1
                st.session_state[page_state_key] = next_page
                st.rerun()

        for idx, token in enumerate(display_tokens, start=1):
            with row_cols[idx]:
                if token == "...":
                    st.markdown("<div style='text-align:center; color:#6f7f96; padding-top:6px;'>...</div>", unsafe_allow_html=True)
                else:
                    page_num = int(token)
                    if page_num == current_page:
                        st.button(
                            token,
                            key=f"{widget_scope}_num_{token}",
                            type="primary",
                            disabled=True,
                            use_container_width=True,
                        )
                    else:
                        if st.button(token, key=f"{widget_scope}_num_{token}", use_container_width=True):
                            st.session_state[page_state_key] = page_num
                            st.rerun()

        with row_cols[-1]:
            if st.button("Next", key=f"{widget_scope}_next", disabled=current_page >= total_pages, use_container_width=True):
                next_page = current_page + 1
                st.session_state[page_state_key] = next_page
                st.rerun()

    st.markdown(
        f"<div style='text-align:center; color:#9ca8ba; font-size:12px; margin-top:2px;'>Page {current_page} of {total_pages}</div>",
        unsafe_allow_html=True,
    )
    return current_page


def order_parent_before_replies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required_cols = {"comment_id", "parent_id", "is_reply"}
    if not required_cols.issubset(df.columns):
        return df

    ordered = df.reset_index(drop=True).copy()
    ordered["_row_pos"] = range(len(ordered))
    ordered["_is_reply"] = ordered["is_reply"].astype(bool)

    root_rows = ordered[~ordered["_is_reply"]]
    reply_rows = ordered[ordered["_is_reply"]]

    replies_by_parent: dict[str, list[int]] = {}
    for idx, row in reply_rows.iterrows():
        parent_id = str(row.get("parent_id", "") or "")
        if parent_id:
            replies_by_parent.setdefault(parent_id, []).append(idx)

    final_indices: list[int] = []
    used_indices: set[int] = set()

    for root_idx, root_row in root_rows.iterrows():
        final_indices.append(root_idx)
        used_indices.add(root_idx)

        root_comment_id = str(root_row.get("comment_id", "") or "")
        child_indices = replies_by_parent.pop(root_comment_id, [])
        for child_idx in child_indices:
            if child_idx not in used_indices:
                final_indices.append(child_idx)
                used_indices.add(child_idx)

    for orphan_idx in reply_rows.index.tolist():
        if orphan_idx not in used_indices:
            final_indices.append(orphan_idx)
            used_indices.add(orphan_idx)

    final_df = ordered.loc[final_indices].drop(columns=["_row_pos", "_is_reply"], errors="ignore")
    return final_df.reset_index(drop=True)


def render_comments_section(df: pd.DataFrame):
    if df.empty:
        st.warning("No results found. Try broadening query or removing filters.")
        return

    out = order_parent_before_replies(df)

    signature = (
        len(out),
        tuple(out.get("comment_id", pd.Series([], dtype=str)).fillna("").astype(str).head(50).tolist()),
    )
    if st.session_state.get("ranked_results_signature") != signature:
        st.session_state["ranked_results_signature"] = signature
        st.session_state["ranked_results_page"] = 1

    total_results = len(out)
    total_pages = max(1, math.ceil(total_results / RESULTS_PAGE_SIZE))
    current_page = int(st.session_state.get("ranked_results_page", 1))
    current_page = max(1, min(current_page, total_pages))
    st.session_state["ranked_results_page"] = current_page

    start_idx = (current_page - 1) * RESULTS_PAGE_SIZE
    end_idx = min(start_idx + RESULTS_PAGE_SIZE, total_results)
    page_df = out.iloc[start_idx:end_idx]

    st.caption(f"About {total_results} results")
    render_results_pagination(
        total_pages,
        current_page,
        page_state_key="ranked_results_page",
        widget_scope="ranked_results_top",
    )

    for local_idx, (_, row) in enumerate(page_df.iterrows(), start=1):
        global_rank = start_idx + local_idx
        render_comment_card(row, rank=global_rank)

    render_results_pagination(
        total_pages,
        current_page,
        page_state_key="ranked_results_page",
        widget_scope="ranked_results_bottom",
    )


def main():
    st.set_page_config(page_title="SC4021 Opinion Search Engine", layout="wide")
    st.title("SC4021 Opinion Search Engine")

    if not os.path.exists(DATA_PATH):
        st.error("Missing data/comments_relevant.csv. Please ensure the file exists.")
        return

    raw_df = pd.read_csv(DATA_PATH)
    df = ensure_types(raw_df)
    init_session_state()

    # Clear query input only before the widget is instantiated.
    if st.session_state.get("clear_query_input_next_run", False):
        st.session_state["query_input"] = ""
        st.session_state["clear_query_input_next_run"] = False

    with st.sidebar:
        st.header("Index Controls")
        rebuild = st.button("Rebuild Index", use_container_width=True)
        st.markdown("---")

    ix = get_index(df, force_rebuild=rebuild)

    st.subheader("Search")
    with st.form("query_form"):
        st.text_input("Query", key="query_input", placeholder="e.g., iphone 17 battery")
        query_submitted = st.form_submit_button("Run Query")

    if query_submitted:
        new_query = st.session_state["query_input"].strip()
        if new_query and new_query != st.session_state["last_submitted_query"]:
            reset_filters()
        st.session_state["active_query"] = new_query
        st.session_state["last_submitted_query"] = new_query
        st.session_state["query_mode"] = "text"

    query_text = st.session_state.get("active_query", "")

    valid_dates = df["published_at"].dropna().dt.date if "published_at" in df.columns else pd.Series([], dtype=object)
    min_date = valid_dates.min() if not valid_dates.empty else None
    max_date = valid_dates.max() if not valid_dates.empty else None

    if min_date is not None and st.session_state.get("start_date_filter") is None:
        st.session_state["start_date_filter"] = min_date
    if max_date is not None and st.session_state.get("end_date_filter") is None:
        st.session_state["end_date_filter"] = max_date

    bucket_to_family, family_to_buckets, ambiguous_buckets = get_brand_model_mappings(df)
    all_families = sorted(df["family"].dropna().astype(str).unique().tolist())
    all_buckets = sorted(df["bucket"].dropna().astype(str).unique().tolist())

    with st.sidebar:
        st.header("Facets")
        if st.button("Reset Filters"):
            reset_filters()
            st.session_state["query_mode"] = "all"

        family_options = ["All"] + sorted(df["family"].dropna().astype(str).unique().tolist())
        category_options = ["All"] + sorted(df["comment_category"].dropna().astype(str).unique().tolist())

        selected_family = st.selectbox("Brand", family_options, key="family_filter", on_change=on_facet_change)

        if selected_family == "All":
            available_buckets = sorted(df["bucket"].dropna().astype(str).unique().tolist())
        else:
            available_buckets = family_to_buckets.get(selected_family, [])
        bucket_options = ["All"] + available_buckets

        # Keep bucket selection valid if available options changed after family selection.
        if st.session_state.get("bucket_filter") not in bucket_options:
            st.session_state["bucket_filter"] = "All"

        selected_bucket = st.selectbox("Phone Model", bucket_options, key="bucket_filter", on_change=on_facet_change)

        # If a model is selected, auto-align brand for filtering consistency.
        effective_family = selected_family
        if selected_bucket != "All":
            mapped_family = bucket_to_family.get(selected_bucket)
            if mapped_family and mapped_family != selected_family:
                effective_family = mapped_family
                st.caption(f"Brand auto-aligned to {mapped_family} for model {selected_bucket}.")

        selected_category = st.selectbox("Category", category_options, key="category_filter", on_change=on_facet_change)

        # Advanced facets removed for simpler interaction flow.
        selected_sentiment = "All"
        selected_aspect = "All"

        include_replies = True

        enable_date_filter = st.checkbox("Enable Date Range", key="enable_date_filter", on_change=on_facet_change)
        if min_date is not None and max_date is not None:
            if enable_date_filter:
                selected_start_date = st.date_input(
                    "From",
                    min_value=min_date,
                    max_value=max_date,
                    key="start_date_filter",
                    on_change=on_facet_change,
                )
                selected_end_date = st.date_input(
                    "To",
                    min_value=min_date,
                    max_value=max_date,
                    key="end_date_filter",
                    on_change=on_facet_change,
                )
            else:
                selected_start_date = st.session_state.get("start_date_filter")
                selected_end_date = st.session_state.get("end_date_filter")
        else:
            selected_start_date = None
            selected_end_date = None
            st.caption("No valid published dates found for date filtering.")

        if ambiguous_buckets:
            st.caption("Warning: some models map to multiple brands in data. Check source rows for consistency.")

    query_mode = st.session_state.get("query_mode", "all")
    effective_query_text = query_text if query_mode == "text" else ""

    inferred_bucket = infer_bucket_from_query(effective_query_text, all_buckets)
    inferred_family = infer_family_from_query(effective_query_text, all_families)
    inferred_category = infer_category_from_query(effective_query_text)
    inferred_unknown_model = infer_unknown_model_from_query(effective_query_text, all_buckets, all_families)
    applied_rules: list[str] = []

    effective_bucket = selected_bucket
    if inferred_bucket is not None:
        effective_bucket = inferred_bucket
        applied_rules.append(f"Model detected -> bucket fixed to {inferred_bucket}")

    # Family rule: query-mentioned brand should constrain results.
    effective_family = selected_family
    if inferred_family is not None:
        effective_family = inferred_family
        applied_rules.append(f"Brand detected -> family fixed to {inferred_family}")

    # Model rule is more specific than family rule; keep them consistent.
    if inferred_bucket is not None:
        mapped_family = bucket_to_family.get(inferred_bucket)
        if mapped_family is not None:
            effective_family = mapped_family

    effective_category = selected_category
    if inferred_category is not None:
        effective_category = inferred_category
        applied_rules.append(f"Topic detected -> category fixed to {inferred_category}")

    if inferred_unknown_model is not None:
        applied_rules.append(f"Out-of-scope model detected -> {inferred_unknown_model}")

    if applied_rules:
        lines = "".join(f"<div style='margin:1px 0;'>{html.escape(rule)}</div>" for rule in applied_rules)
        st.markdown(
            f"""
            <div style="margin:4px 0 8px 0; padding:6px 10px; border:1px solid #2b313d; border-radius:8px; background:#141923; color:#b7c2d4; font-size:12px; line-height:1.2;">
                {lines}
            </div>
            """,
            unsafe_allow_html=True,
        )

    query_start = time.perf_counter()

    query_out_of_scope = bool(inferred_unknown_model) and bool(effective_query_text.strip())

    # Queryless browsing should still work: use full dataset baseline when query is empty.
    if query_out_of_scope:
        st.warning(
            f"The query references a phone model outside this dataset scope ({inferred_unknown_model}). "
            "No results are returned."
        )
        search_df = df.iloc[0:0].copy()
        search_df["score"] = pd.Series(dtype=float)
    elif effective_query_text.strip():
        search_df, _search_latency_ms = search(
            ix,
            effective_query_text,
            limit=SEARCH_LIMIT,
            category=effective_category,
            sentiment=selected_sentiment,
            include_replies=include_replies,
            enable_date_filter=enable_date_filter,
            start_date=selected_start_date,
            end_date=selected_end_date,
        )
    else:
        search_df = df.copy()
        search_df["score"] = 0.0

    result_df = apply_filters(
        search_df,
        effective_family,
        effective_bucket,
        effective_category,
        selected_sentiment,
        selected_aspect,
        include_replies,
        enable_date_filter,
        selected_start_date,
        selected_end_date,
        effective_query_text,
    )

    latency_ms = (time.perf_counter() - query_start) * 1000.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div style="border:1px solid #2b313d; border-radius:10px; padding:8px 10px; background:#12161e;">
              <div style="font-size:12px; color:#9ca8ba; margin-bottom:4px;">Result Count</div>
              <div style="font-size:16px; line-height:1.25; color:#e8edf3; font-weight:700;">{int(len(result_df))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div style="border:1px solid #2b313d; border-radius:10px; padding:8px 10px; background:#12161e;">
              <div style="font-size:12px; color:#9ca8ba; margin-bottom:4px;">Latency (ms)</div>
              <div style="font-size:16px; line-height:1.25; color:#e8edf3; font-weight:700;">{latency_ms:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if effective_query_text.strip():
        query_label = effective_query_text
    else:
        facet_parts: list[str] = []
        if selected_family != "All":
            facet_parts.append(f"brand={selected_family}")
        if selected_bucket != "All":
            facet_parts.append(f"model={selected_bucket}")
        if selected_category != "All":
            facet_parts.append(f"category={selected_category}")
        query_label = " | ".join(facet_parts) if facet_parts else "<all>"
    with m3:
        st.markdown(
            f"""
            <div style="border:1px solid #2b313d; border-radius:10px; padding:8px 10px; background:#12161e;">
              <div style="font-size:12px; color:#9ca8ba; margin-bottom:4px;">Query</div>
                            <div style="font-size:16px; line-height:1.25; color:#e8edf3; font-weight:700; white-space:normal; overflow-wrap:anywhere; word-break:break-word;">{html.escape(query_label)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Summary is the first tab so it is the default visible section.
    tab_summary, tab_results, tab_compare = st.tabs(["Summary", "Ranked Results", "Compare Sentiment"])

    with tab_summary:
        sentiment_chart(result_df)
        category_chart(result_df)
        results_over_time_chart(result_df)
        word_cloud_chart(result_df)

    with tab_results:
        render_comments_section(result_df)

    with tab_compare:
        st.subheader("Compare Two Queries")
        st.caption("Compares sentiment distribution for two search queries under the current sidebar filters.")

        with st.form("compare_sentiment_form"):
            compare_query_a = st.text_input("Query A", placeholder="e.g., iphone 17 battery")
            compare_query_b = st.text_input("Query B", placeholder="e.g., iphone 17 camera")
            compare_submitted = st.form_submit_button("Compare Sentiment")

        if compare_submitted:
            q_a = compare_query_a.strip()
            q_b = compare_query_b.strip()
            if not q_a or not q_b:
                st.warning("Please enter both Query A and Query B.")
            else:
                render_sentiment_comparison(
                    ix,
                    q_a,
                    q_b,
                    effective_family,
                    effective_bucket,
                    effective_category,
                    selected_sentiment,
                    selected_aspect,
                    include_replies,
                    enable_date_filter,
                    selected_start_date,
                    selected_end_date,
                    all_buckets,
                    all_families,
                )


if __name__ == "__main__":
    main()
