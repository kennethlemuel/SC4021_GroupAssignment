from __future__ import annotations

from search_engine.app import main
from search_engine.charts import (
    category_chart,
    results_over_time_chart,
    sentiment_chart,
    sentiment_counts,
    word_cloud_chart,
)
from search_engine.compare import render_sentiment_comparison
from search_engine.data_utils import detect_aspects, ensure_types, parse_dt
from search_engine.inference import (
    bucket_pattern,
    infer_bucket_from_query,
    infer_category_from_query,
    infer_family_from_query,
    infer_unknown_model_from_query,
)
from search_engine.indexing import build_index, build_schema, get_index, search
from search_engine.ranking import apply_filters, apply_fixed_reranking, get_brand_model_mappings, minmax_norm
from search_engine.state import init_session_state, on_facet_change, reset_filters
from search_engine.components import (
    format_comment_date,
    order_parent_before_replies,
    render_comment_card,
    render_comments_section,
    render_results_pagination,
    sentiment_style,
)
from search_engine.config import (
    ASPECT_PATTERNS,
    COMMENTS_PAGE_SIZE,
    DATA_PATH,
    INDEX_DIR,
    RANK_WEIGHTS,
    RESULTS_PAGE_SIZE,
    SEARCH_FIELDS,
    SEARCH_LIMIT,
)


if __name__ == "__main__":
    main()
