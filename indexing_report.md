# SC4021 Information Retrieval 2026
## Indexing and Querying Report (Question 2 and Question 3)

## 1. System Objective
This project implements an opinion-focused search engine over smartphone comments using an inverted index (Whoosh) and a Streamlit interface.

The system supports:
1. Text retrieval over user comments.
2. Faceted filtering (brand, model, category, date range).
3. Deterministic reranking for more useful opinion ordering.
4. Sentiment and category visual analytics.
5. Side-by-side sentiment comparison between two user queries.

## 2. Indexed Dataset and Fields
Primary dataset:
- `data/comments_relevant.csv`

Index unit:
- One comment row = one indexed document.

Core indexed/searched fields:
- `text`
- `bucket` (phone model)
- `family` (brand)
- `comment_category`
- `video_title`
- `search_query`

Stored metadata fields for filtering and presentation:
- `variant`
- `aspects` (rule-derived)
- `like_count`
- `relevance_score`
- `is_reply`
- `published_at`
- `suggested_sentiment_label`

Important update:
- Category logic now uses `comment_category` (not `category`) across indexing, filtering, and visualizations.

## 3. Indexing and Query Pipeline
### 3.1 Inverted Index Construction
The system builds a Whoosh schema and writes each row as a document. Rebuild can be triggered from UI (`Rebuild Index`).

### 3.2 Querying
Base retrieval uses `MultifieldParser` over:
- `text`, `bucket`, `family`, `comment_category`, `video_title`, `search_query`

Additional constraints are added as query terms/ranges:
- Category filter (`comment_category`)
- Sentiment label
- Reply inclusion
- Date range (`published_at`)

### 3.3 Post-Retrieval Filtering
Further deterministic filtering occurs in DataFrame stage for:
- Family
- Model bucket
- Category (`comment_category`)
- Sentiment/aspect flags
- Date range

### 3.4 Fixed Reranking (Assignment-aligned)
Final ranking score:

$$
	ext{FinalScore} = 0.55\cdot T + 0.20\cdot R + 0.15\cdot E + 0.10\cdot Q
$$

Where:
- $T$: normalized lexical relevance score
- $R$: normalized `relevance_score`
- $E$: normalized $\log(1 + like\_count)$
- $Q$: reply quality prior (`1` non-reply, `0` reply)

Tie-break order:
1. lexical score
2. `relevance_score`
3. `like_count`

## 4. Implemented Product Features
### 4.1 Query Mode Control (last-edited-wins)
- Text query mode and facet mode are mutually exclusive.
- Editing facets clears existing text input and switches to facet mode.
- Submitting a new text query resets facets.

### 4.2 Automatic Query Inference Rules
From typed query text, the app can infer and lock:
1. Model (`bucket`)
2. Brand (`family`)
3. Topic (`comment_category`)

Rule status is shown to user in a compact panel.

### 4.3 Strict Base/Premium Model Disambiguation
Model detection handles flexible separators and specific variants:
- `iphone 17 pro`, `iphone17pro`, `iphone-17-pro`

Longest-match and suffix-guard logic prevent mixing base and premium models.

### 4.4 Out-of-Scope Model Guard
To avoid misleading retrieval (e.g., `iphone 16` mapping to existing Apple models):
1. If query contains clear model intent (recognized family + model token),
2. but no known dataset model (`bucket`) is matched,
3. then query is marked out-of-scope.

Behavior:
- Main search: shows warning and returns 0 results.
- Compare Sentiment: validates Query A and Query B independently; if either is out-of-scope, shows warning and aborts comparison render.

### 4.5 Ranked Results Pagination
- Ranked comments are paginated with top and bottom controls.
- Navigation uses dynamic-width button layout to avoid compression with large page numbers.

### 4.6 Summary Analytics
The Summary tab includes:
1. Sentiment distribution pie + platform sentiment indicator panel
2. Category distribution bar chart (from `comment_category`)
3. Results-over-time chart
4. Word cloud (if optional dependencies are installed)

### 4.7 Compare Sentiment
- Two query inputs under current sidebar filters.
- Grouped sentiment comparison bar chart.
- Dynamic legend labels now reflect actual Query A/Query B text.
- Delta table reports positive%, negative%, and net sentiment difference.

## 5. UI/UX and Reporting Metrics
Implemented UI characteristics:
1. Unified metric cards: Result Count, Latency, Query.
2. Actual measured end-to-end latency (including facet-only flow).
3. Dark-theme visual consistency with muted sentiment palette.
4. Clear warnings for empty results and out-of-scope model queries.

## 6. Assignment Questions Mapping
### Question 2 (Simple UI + Query Performance)
Delivered:
1. Streamlit query interface with faceted retrieval.
2. Query speed display in milliseconds.
3. Ranked results and visual summaries.

### Question 3 (Innovations)
Delivered innovations:
1. Multifaceted retrieval beyond plain keyword search.
2. Deterministic hybrid reranking combining lexical + metadata signals.
3. Comparative sentiment analytics for two queries.
4. Out-of-scope model guard to enforce dataset validity and reduce false interpretation.

## 7. How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start app:

```bash
streamlit run streamlit_search_engine.py
```

3. If schema/field changes were made, click `Rebuild Index` once after launch.

## 8. Main Artifacts
- `streamlit_search_engine.py` (thin entrypoint + backwards-compatible exports)
- `indexing_report.md`
- `search_engine/config.py` (constants, weights, field lists)
- `search_engine/data_utils.py` (type normalization, datetime parsing, aspect detection)
- `search_engine/indexing.py` (Whoosh schema, indexing, query execution)
- `search_engine/ranking.py` (post-retrieval filtering and fixed reranking)
- `search_engine/inference.py` (model/brand/category inference + out-of-scope detection)
- `search_engine/state.py` (Streamlit session-state defaults and callbacks)
- `search_engine/charts.py` (summary visualizations)
- `search_engine/components.py` (result cards, pagination, comment ordering)
- `search_engine/compare.py` (Compare Sentiment workflow)
- `search_engine/app.py` (main Streamlit page orchestration)

## 9. Refactoring Notes
The original single-file implementation was reorganized into a package to improve readability, change isolation, and maintainability.

Design principles used:
1. Separation of concerns: indexing, inference, ranking, UI state, and rendering are in separate modules.
2. Stable entrypoint: existing command `streamlit run streamlit_search_engine.py` remains unchanged.
3. Backward compatibility: key functions are still importable from `streamlit_search_engine.py` via re-export.
4. Behavior parity: all previously implemented features (query-mode switching, strict model matching, out-of-scope guard, dynamic pagination, compare sentiment) are preserved.
