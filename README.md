# SC4021 Project

## Data Files

- data/comments_clean.csv: cleaned full corpus
- data/comments_relevant.csv: filtered corpus used by the search engine
- data/annotation_candidates.csv: annotation set from corpus for manual annotation
- data/annotated_evaluation_set.csv: final annotated dataset for evaluation
- data/seed_videos.csv: selected seed videos
- logs/crawl_errors.csv: crawl errors for debugging purposes

## Search Engine Files

- streamlit_search_engine.py: app entry point
- search_engine/app.py: main Streamlit flow and UI orchestration
- search_engine/indexing.py: Whoosh schema, indexing, and search
- search_engine/inference.py: query inference and out-of-scope model checks
- search_engine/ranking.py: filtering and reranking logic
- search_engine/charts.py: summary visualizations
- search_engine/components.py: reusable UI components (cards, pagination, ranked results)
- search_engine/compare.py: compare sentiment workflow
- search_engine/state.py: session state defaults and callbacks
- search_engine/config.py: shared constants/configuration
- indexing_report.md: indexing and querying report

## Quick Start

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_search_engine.py
```

4. Open the local URL shown by Streamlit (usually `http://localhost:8501`).
5. If schema/field changes were made, click `Rebuild Index` once in the sidebar.
