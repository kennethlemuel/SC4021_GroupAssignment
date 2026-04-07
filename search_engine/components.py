from __future__ import annotations

import html
import math
from datetime import datetime

import pandas as pd
import streamlit as st

from .config import RESULTS_PAGE_SIZE


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
    category = html.escape(str(row.get("comment_category", "")))
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
            token_widths.append(max(1.0, 0.7 + 0.35 * len(token)))

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
