DATA_PATH = "data/comments_relevant.csv"
INDEX_DIR = "indexdir"
SEARCH_FIELDS = ["text", "bucket", "family", "comment_category", "video_title", "search_query"]
SEARCH_LIMIT = 10000
COMMENTS_PAGE_SIZE = 20
RESULTS_PAGE_SIZE = 10

ASPECT_PATTERNS = {
    "battery": ("battery", "charging", "mah", "magsafe"),
    "display": ("display", "screen", "brightness", "hz", "refresh"),
    "camera": ("camera", "photo", "video", "zoom", "sensor"),
    "performance": ("performance", "chip", "cpu", "exynos", "snapdragon", "lag", "smooth"),
    "design": ("design", "size", "weight", "build", "ergonomic", "form factor", "titanium", "aluminum"),
    "price": ("price", "cost", "expensive", "cheap", "value", "msrp", "usd", "$"),
    "ai": ("ai", "artificial intelligence"),
}

RANK_WEIGHTS = {
    "text_relevance": 0.55,
    "relevance_label": 0.20,
    "engagement": 0.15,
    "quality": 0.10,
}
