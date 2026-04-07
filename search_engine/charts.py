from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False


def sentiment_counts(df: pd.DataFrame) -> dict[str, int]:
    labels = ["positive", "neutral", "negative"]
    if df.empty:
        return {label: 0 for label in labels}
    series = df.get("suggested_sentiment_label", pd.Series([], dtype=str)).fillna("neutral").astype(str).str.lower()
    return {label: int((series == label).sum()) for label in labels}


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
