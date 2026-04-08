from __future__ import annotations

import html
import os
import time

import pandas as pd
import streamlit as st

from .charts import category_chart, results_over_time_chart, sentiment_chart, word_cloud_chart
from .compare import render_sentiment_comparison
from .config import DATA_PATH, SEARCH_LIMIT, SENTIMENT_RESULTS_PATH
from .data_utils import ensure_types, merge_final_sentiment
from .inference import (
    infer_bucket_from_query,
    infer_category_from_query,
    infer_family_from_query,
    infer_unknown_model_from_query,
)
from .indexing import get_index, search
from .ranking import apply_filters, get_brand_model_mappings
from .state import init_session_state, on_facet_change, reset_filters
from .components import render_comments_section


def main():
    st.set_page_config(page_title="SC4021 Opinion Search Engine", layout="wide")
    st.title("SC4021 Opinion Search Engine")

    if not os.path.exists(DATA_PATH):
        st.error("Missing data/comments_relevant.csv. Please ensure the file exists.")
        return

    raw_df = pd.read_csv(DATA_PATH)
    raw_df = merge_final_sentiment(raw_df, SENTIMENT_RESULTS_PATH)
    df = ensure_types(raw_df)
    init_session_state()

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

        if st.session_state.get("bucket_filter") not in bucket_options:
            st.session_state["bucket_filter"] = "All"

        selected_bucket = st.selectbox("Phone Model", bucket_options, key="bucket_filter", on_change=on_facet_change)

        if selected_bucket != "All":
            mapped_family = bucket_to_family.get(selected_bucket)
            if mapped_family and mapped_family != selected_family:
                st.caption(f"Brand auto-aligned to {mapped_family} for model {selected_bucket}.")

        selected_category = st.selectbox("Category", category_options, key="category_filter", on_change=on_facet_change)

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

    effective_family = selected_family
    if inferred_family is not None:
        effective_family = inferred_family
        applied_rules.append(f"Brand detected -> family fixed to {inferred_family}")

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
