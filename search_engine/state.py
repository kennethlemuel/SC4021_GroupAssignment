from __future__ import annotations

import streamlit as st


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
    st.session_state["query_mode"] = "facet"
    st.session_state["active_query"] = ""
    st.session_state["last_submitted_query"] = ""
    st.session_state["clear_query_input_next_run"] = True
