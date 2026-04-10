from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from .charts import sentiment_counts
from .config import SEARCH_LIMIT
from .inference import (
    infer_bucket_from_query,
    infer_category_from_query,
    infer_family_from_query,
    infer_unknown_model_from_query,
    normalize_query_to_canonical_model,
)
from .indexing import search
from .ranking import apply_filters


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
    bucket_to_family: dict[str, str],
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

    inferred_bucket_a = infer_bucket_from_query(query_a, all_buckets)
    inferred_bucket_b = infer_bucket_from_query(query_b, all_buckets)
    inferred_family_a = infer_family_from_query(query_a, all_families)
    inferred_family_b = infer_family_from_query(query_b, all_families)
    inferred_category_a = infer_category_from_query(query_a)
    inferred_category_b = infer_category_from_query(query_b)

    normalized_query_a = normalize_query_to_canonical_model(
        query_a,
        inferred_bucket_a or "",
        all_buckets,
        bucket_to_family.get(inferred_bucket_a or "") or inferred_family_a,
    )
    normalized_query_b = normalize_query_to_canonical_model(
        query_b,
        inferred_bucket_b or "",
        all_buckets,
        bucket_to_family.get(inferred_bucket_b or "") or inferred_family_b,
    )

    effective_family_a = family
    effective_family_b = family
    effective_bucket_a = bucket
    effective_bucket_b = bucket
    effective_category_a = category
    effective_category_b = category

    if inferred_bucket_a is not None:
        effective_bucket_a = inferred_bucket_a
        mapped_family_a = bucket_to_family.get(inferred_bucket_a)
        if mapped_family_a is not None:
            effective_family_a = mapped_family_a
    if inferred_bucket_b is not None:
        effective_bucket_b = inferred_bucket_b
        mapped_family_b = bucket_to_family.get(inferred_bucket_b)
        if mapped_family_b is not None:
            effective_family_b = mapped_family_b

    if inferred_family_a is not None:
        effective_family_a = inferred_family_a
    if inferred_family_b is not None:
        effective_family_b = inferred_family_b

    if inferred_category_a is not None:
        effective_category_a = inferred_category_a
    if inferred_category_b is not None:
        effective_category_b = inferred_category_b

    q1_df, _ = search(
        ix,
        normalized_query_a,
        limit=SEARCH_LIMIT,
        category=effective_category_a,
        sentiment=sentiment,
        include_replies=include_replies,
        enable_date_filter=enable_date_filter,
        start_date=start_date,
        end_date=end_date,
    )
    q2_df, _ = search(
        ix,
        normalized_query_b,
        limit=SEARCH_LIMIT,
        category=effective_category_b,
        sentiment=sentiment,
        include_replies=include_replies,
        enable_date_filter=enable_date_filter,
        start_date=start_date,
        end_date=end_date,
    )

    q1_filtered = apply_filters(
        q1_df,
        effective_family_a,
        effective_bucket_a,
        effective_category_a,
        sentiment,
        aspect,
        include_replies,
        enable_date_filter,
        start_date,
        end_date,
        normalized_query_a,
    )
    q2_filtered = apply_filters(
        q2_df,
        effective_family_b,
        effective_bucket_b,
        effective_category_b,
        sentiment,
        aspect,
        include_replies,
        enable_date_filter,
        start_date,
        end_date,
        normalized_query_b,
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
        [{"sentiment": label, "query": query_a_label, "count": q1_counts[label]} for label in labels]
        + [{"sentiment": label, "query": query_b_label, "count": q2_counts[label]} for label in labels]
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
    st.dataframe(delta_df, use_container_width=True, hide_index=True)
