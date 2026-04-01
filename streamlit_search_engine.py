from __future__ import annotations

import os
import re
import shutil
import time
import math
import html
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import BOOLEAN, DATETIME, ID, NUMERIC, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import Every

DATA_PATH = "data/comments_relevant.csv"
INDEX_DIR = "indexdir"
SEARCH_FIELDS = ["text", "bucket", "family", "category", "video_title", "search_query"]
SEARCH_LIMIT = 10000
COMMENTS_PAGE_SIZE = 20


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
        category=ID(stored=True),
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
            category=str(row.get("category", "")),
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


def search(ix, query_text: str, limit: int = SEARCH_LIMIT):
    start = time.perf_counter()
    with ix.searcher() as searcher:
        parser = MultifieldParser(SEARCH_FIELDS, schema=ix.schema, group=OrGroup.factory(0.9))
        query = Every() if not query_text.strip() else parser.parse(query_text.strip())
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
                    "category": hit.get("category"),
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
    query_text: str,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if family != "All":
        out = out[out["family"] == family]
    if bucket != "All":
        out = out[out["bucket"] == bucket]
    if category != "All":
        out = out[out["category"] == category]
    if sentiment != "All":
        out = out[out["suggested_sentiment_label"] == sentiment]
    if aspect != "All":
        out = out[out["aspects"].astype(str).str.contains(aspect, na=False)]
    if not include_replies:
        out = out[out["is_reply"] == False]

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
    lowered = bucket.casefold()
    escaped = re.escape(lowered).replace("\\ ", r"\\s+")

    # For base models that are substrings of premium variants, exclude common premium suffixes.
    is_prefix_of_other = any(
        other.casefold().startswith(lowered + " ") for other in all_buckets if other != bucket
    )
    if is_prefix_of_other:
        return re.compile(
            rf"\\b{escaped}\\b(?!\\s+(pro|max|plus|ultra|mini|fold|flip|edge))",
            flags=re.IGNORECASE,
        )

    return re.compile(rf"\\b{escaped}\\b", flags=re.IGNORECASE)


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

    if len(matches) == 1:
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


def reset_filters() -> None:
    st.session_state["family_filter"] = "All"
    st.session_state["bucket_filter"] = "All"
    st.session_state["category_filter"] = "All"
    st.session_state["sentiment_filter"] = "All"
    st.session_state["aspect_filter"] = "All"
    st.session_state["include_replies_filter"] = True


def init_session_state() -> None:
    defaults = {
        "family_filter": "All",
        "bucket_filter": "All",
        "category_filter": "All",
        "sentiment_filter": "All",
        "aspect_filter": "All",
        "include_replies_filter": True,
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
    chart_df = df.groupby("suggested_sentiment_label").size().reset_index(name="count")
    fig = px.pie(chart_df, names="suggested_sentiment_label", values="count", title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)


def category_chart(df: pd.DataFrame):
    if df.empty:
        return
    chart_df = df.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
    fig = px.bar(chart_df, x="category", y="count", title="Category Distribution")
    st.plotly_chart(fig, use_container_width=True)


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


def render_comment_card(row: pd.Series, depth: int = 0):
    text = html.escape(str(row.get("text", "")))
    model = html.escape(str(row.get("bucket", "")))
    published = format_comment_date(row.get("published_at"))
    likes = int(pd.to_numeric(row.get("like_count", 0), errors="coerce") or 0)
    weighted = float(pd.to_numeric(row.get("weighted_score", 0.0), errors="coerce") or 0.0)
    sentiment_raw = str(row.get("suggested_sentiment_label", "neutral"))
    sentiment = html.escape(sentiment_raw)
    chip_bg, chip_fg = sentiment_style(sentiment_raw)
    left = f"{min(depth, 6) * 28}px"

    st.markdown(
        f"""
        <div style=\"margin:8px 0 10px {left}; padding:10px 12px; background:#12161e; border:1px solid #2b313d; border-radius:12px;\">
          <div style=\"display:flex; gap:8px; align-items:center; color:#c7ced9; font-size:12px; margin-bottom:8px;\">
            <span style=\"font-weight:600; color:#f2f5f9;\">{model}</span>
            <span>• {published}</span>
            <span>• score {weighted:.3f}</span>
            <span>• 👍 {likes}</span>
            <span style=\"margin-left:auto; padding:2px 8px; border-radius:999px; background:{chip_bg}; color:{chip_fg};\">{sentiment}</span>
          </div>
          <div style=\"font-size:14px; line-height:1.45; color:#e8edf3; white-space:pre-wrap;\">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comments_section(df: pd.DataFrame):
    if df.empty:
        st.warning("No results found. Try broadening query or removing filters.")
        return

    out = df.copy()
    out["comment_id"] = out.get("comment_id", "").fillna("").astype(str)
    out["parent_id"] = out.get("parent_id", "").fillna("").astype(str)
    out["is_reply"] = out.get("is_reply", False).astype(bool)

    top_level = out[out["is_reply"] == False].copy()
    replies = out[out["is_reply"] == True].copy()

    replies_by_parent: dict[str, pd.DataFrame] = {
        pid: grp.sort_values(by=["weighted_score", "like_count"], ascending=False)
        for pid, grp in replies.groupby("parent_id", dropna=False)
    }

    # Paginate parent comments so initial render stays lightweight.
    top_level_ids = top_level.get("comment_id", pd.Series([], dtype=str)).astype(str).tolist()
    signature = (len(top_level_ids), tuple(top_level_ids[:50]))
    if st.session_state.get("comments_signature") != signature:
        st.session_state["comments_signature"] = signature
        st.session_state["comments_page"] = 1

    total_parents = len(top_level)
    total_pages = max(1, math.ceil(total_parents / COMMENTS_PAGE_SIZE))
    current_page = int(st.session_state.get("comments_page", 1))
    current_page = max(1, min(current_page, total_pages))
    st.session_state["comments_page"] = current_page

    start_idx = (current_page - 1) * COMMENTS_PAGE_SIZE
    end_idx = start_idx + COMMENTS_PAGE_SIZE
    page_parents = top_level.iloc[start_idx:end_idx]

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("Previous", key="comments_prev", disabled=current_page <= 1):
            st.session_state["comments_page"] = current_page - 1
            st.rerun()
    with c2:
        shown_start = 0 if total_parents == 0 else start_idx + 1
        shown_end = min(end_idx, total_parents)
        st.caption(f"Showing parent comments {shown_start}-{shown_end} of {total_parents} (Page {current_page}/{total_pages})")
    with c3:
        if st.button("Next", key="comments_next", disabled=current_page >= total_pages):
            st.session_state["comments_page"] = current_page + 1
            st.rerun()

    with st.container(height=680):
        for _, parent in page_parents.iterrows():
            with st.container(border=True):
                render_comment_card(parent, depth=0)
                pid = str(parent.get("comment_id", ""))
                direct_replies = replies_by_parent.get(pid)
                if direct_replies is not None and not direct_replies.empty:
                    with st.expander(f"Show replies ({len(direct_replies)})", expanded=False):
                        for _, reply in direct_replies.iterrows():
                            render_comment_card(reply, depth=1)


def main():
    st.set_page_config(page_title="SC4021 Opinion Search Engine", layout="wide")
    st.title("SC4021 Opinion Search Engine")
    st.caption("Question 2 + Question 3 implementation (Whoosh + Streamlit)")

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
        rebuild = st.button("Rebuild Index")

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

    bucket_to_family, family_to_buckets, ambiguous_buckets = get_brand_model_mappings(df)
    all_families = sorted(df["family"].dropna().astype(str).unique().tolist())
    all_buckets = sorted(df["bucket"].dropna().astype(str).unique().tolist())

    with st.sidebar:
        st.header("Facets")
        if st.button("Reset Filters"):
            reset_filters()
            st.session_state["query_mode"] = "all"

        family_options = ["All"] + sorted(df["family"].dropna().astype(str).unique().tolist())
        category_options = ["All"] + sorted(df["category"].dropna().astype(str).unique().tolist())
        sentiment_options = ["All"] + sorted(
            df["suggested_sentiment_label"].dropna().astype(str).unique().tolist()
        )

        aspect_values = sorted({a for v in df["aspects"].astype(str).tolist() for a in v.split("|")})
        aspect_options = ["All"] + aspect_values

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
        show_advanced = st.checkbox("Show Advanced Facets", value=False, on_change=on_facet_change)

        if show_advanced:
            selected_sentiment = st.selectbox("Sentiment", sentiment_options, key="sentiment_filter", on_change=on_facet_change)
            selected_aspect = st.selectbox("Aspect", aspect_options, key="aspect_filter", on_change=on_facet_change)
        else:
            selected_sentiment = "All"
            selected_aspect = "All"

        include_replies = st.checkbox("Include Replies", key="include_replies_filter", on_change=on_facet_change)
        auto_model_lock = st.checkbox("Auto-lock detected model from query", value=True)

        if ambiguous_buckets:
            st.caption("Warning: some models map to multiple brands in data. Check source rows for consistency.")

    query_mode = st.session_state.get("query_mode", "all")
    effective_query_text = query_text if query_mode == "text" else ""

    inferred_bucket = infer_bucket_from_query(effective_query_text, all_buckets) if auto_model_lock else None
    inferred_family = infer_family_from_query(effective_query_text, all_families)
    inferred_category = infer_category_from_query(effective_query_text)

    query_start = time.perf_counter()

    # Queryless browsing should still work: use full dataset baseline when query is empty.
    if effective_query_text.strip():
        search_df, _search_latency_ms = search(ix, effective_query_text, limit=SEARCH_LIMIT)
    else:
        search_df = df.copy()
        search_df["score"] = 0.0

    effective_bucket = selected_bucket
    if inferred_bucket is not None:
        effective_bucket = inferred_bucket
        st.caption(f"Rule applied: detected model in query -> bucket fixed to {inferred_bucket}")

    # Family rule: query-mentioned brand should constrain results.
    effective_family = selected_family
    if inferred_family is not None:
        effective_family = inferred_family
        st.caption(f"Rule applied: detected brand in query -> family fixed to {inferred_family}")

    # Model rule is more specific than family rule; keep them consistent.
    if inferred_bucket is not None:
        mapped_family = bucket_to_family.get(inferred_bucket)
        if mapped_family is not None:
            effective_family = mapped_family

    effective_category = selected_category
    if inferred_category is not None:
        effective_category = inferred_category
        st.caption(f"Rule applied: detected topic in query -> category fixed to {inferred_category}")

    result_df = apply_filters(
        search_df,
        effective_family,
        effective_bucket,
        effective_category,
        selected_sentiment,
        selected_aspect,
        include_replies,
        effective_query_text,
    )

    latency_ms = (time.perf_counter() - query_start) * 1000.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Result Count", int(len(result_df)))
    m2.metric("Latency (ms)", f"{latency_ms:.2f}")
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
    m3.metric("Query", query_label)

    # Summary is the first tab so it is the default visible section.
    tab_summary, tab_results = st.tabs(["Summary", "Ranked Results"])

    with tab_summary:
        sentiment_chart(result_df)
        category_chart(result_df)

    with tab_results:
        render_comments_section(result_df)


if __name__ == "__main__":
    main()
