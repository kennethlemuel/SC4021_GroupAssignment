# SC4021 Information Retrieval 2026
## Indexing and Querying Report (Question 2 and Question 3)

## Objective Alignment
This component implements an opinion search engine over smartphone-related user comments.
The indexed corpus is `data/comments_relevant.csv`, which contains opinion-centric comments and supporting metadata.
The goal is to provide:
1. Text retrieval over opinions for a given phone model/topic.
2. Sentiment-oriented exploration of retrieved opinions.
3. Multifaceted access to refine and compare results.

## Data Used for Indexing
Primary file:
- `data/comments_relevant.csv`

Key indexed information:
- Text content: `text`, `video_title`, `search_query`
- Phone facets: `family`, `bucket`, `variant`
- Context facets: `category`, `is_reply`, `published_at`
- Sentiment fields: `suggested_sentiment_label`, `relevance_score`
- Engagement signal: `like_count`

## Indexing Method
We use an inverted index implementation based on Whoosh.
This satisfies the assignment requirement to use a text search engine approach and avoids SQL-only text retrieval.

### Indexed Unit
- One comment = one indexed document

### Searchable Fields
- `text`
- `bucket`
- `family`
- `category`
- `video_title`
- `search_query`

### Filter/Facet Fields
- `family`
- `bucket`
- `variant`
- `category`
- `suggested_sentiment_label`
- `is_reply`
- `aspects` (derived rule-based aspect tags)

### Ranking Strategy
- Ranking is fixed in backend (no user-adjustable weighting) to keep UI simple and retrieval behavior consistent.
- Final score uses a deterministic weighted combination:

$$
	ext{FinalScore} = 0.55\cdot T + 0.20\cdot R + 0.15\cdot E + 0.10\cdot Q
$$

Where:
- $T$: normalized Whoosh text relevance score (BM25-style signal)
- $R$: normalized `relevance_score` from preprocessing
- $E$: normalized $\log(1 + \text{like_count})$ engagement signal
- $Q$: quality indicator (`1` for top-level comment, `0` for reply)

Weight justification:
- `0.55` text relevance: primary IR objective is textual topical match.
- `0.20` relevance label: incorporates dataset-level opinion relevance without overpowering lexical retrieval.
- `0.15` engagement: rewards comments with stronger community interaction while dampening outliers via log scaling.
- `0.10` quality: slight preference for top-level comments because they are usually more self-contained opinions.

Tie-break order after `FinalScore`: Whoosh score, then `relevance_score`, then `like_count`.

## Question 2: Simple UI, 5 Queries, and Query Speed
A simple Streamlit UI was implemented to support querying and exploration.

UI features:
1. Query text box.
2. Sidebar facets (brand/model/category/sentiment/aspect/reply).
3. Ranked result table.
4. Query latency metric (milliseconds).
5. Summary and comparison tabs.
6. Filter reset mechanisms:
	- Manual `Reset Filters` button.
	- Automatic filter reset when a new submitted query is different from the previous submitted query.

Facet behavior notes:
- Core facets (brand/model/category) are always shown.
- Sentiment and aspect are exposed as advanced facets (optional) to reduce over-filtering and preserve sentiment spread analysis by default.

### Five Queries for Evaluation
Use the following five queries and record speed/results in your final submission:
1. `iphone 17 battery life`
2. `galaxy s26 camera quality`
3. `pixel 10 overheating issue`
4. `samsung vs iphone charging speed`
5. `xiaomi 15 ultra display`

### Query Speed Table Template
| Query | Filters | Results | Latency (ms) |
|---|---|---:|---:|
| iphone 17 battery life | None | [fill] | [fill] |
| galaxy s26 camera quality | category=camera | [fill] | [fill] |
| pixel 10 overheating issue | category=issue | [fill] | [fill] |
| samsung vs iphone charging speed | family=Samsung | [fill] | [fill] |
| xiaomi 15 ultra display | family=Xiaomi | [fill] | [fill] |

## Question 3: Innovations for Indexing and Ranking

### Innovation 1: Multifaceted Search
Problem:
- Keyword-only search yields mixed and noisy result sets.

Solution:
- Added facets for brand, phone model, category, sentiment, aspect, and reply mode.

Impact:
- Users can rapidly narrow results to a specific intent.

Example:
- Query `battery`
- Refinement: `bucket=Galaxy S26`, `category=battery`, `sentiment=negative`
- Outcome: focused complaint/opinion set instead of broad mixed comments.

### Innovation 2: Enhanced Search (Visual Summary)
Problem:
- Users cannot quickly infer opinion distribution from a raw list of comments.

Solution:
- Added summary charts (sentiment pie chart and category distribution chart).

Impact:
- Supports instant macro-level interpretation for the current query and active filters.

### Innovation 3: Comparative Opinion View
Problem:
- Users often need comparative insights between phone models.

Solution:
- Added side-by-side comparison for two phone models by sentiment and aspect mention frequency.

Impact:
- Enables decision-oriented retrieval based on user opinions rather than static specs.

## Version Improvement Narrative
Initial baseline:
- Simple text query returning ranked comments.

Improved system:
1. Added faceted filtering.
2. Added sentiment/category visual summaries.
3. Added phone-to-phone comparative view.
4. Added aspect-level filtering and visualization.

These improvements solve discoverability and interpretability issues present in the baseline.

## Run Instructions
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch app:

```bash
streamlit run streamlit_search_engine.py
```

3. Open local URL shown by Streamlit (usually `http://localhost:8501`).

## Files Produced
- `streamlit_search_engine.py` (indexing + querying UI)
- `indexing_report.md` (report-ready markdown for Question 2 and 3)
