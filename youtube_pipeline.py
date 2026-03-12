from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
from dotenv import load_dotenv

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
PUBLISHED_AFTER = "2024-01-01T00:00:00Z"
MIN_DURATION_SECONDS = 120

DEFAULT_SEARCH_RESULTS_PER_QUERY = 15
DEFAULT_KEEP_PER_QUERY = 2
DEFAULT_MAX_PER_CHANNEL_PER_BUCKET = 2
DEFAULT_MAX_THREAD_PAGES_PER_VIDEO = 5

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
SEED_VIDEOS_PATH = DATA_DIR / "seed_videos.csv"
SEED_SUMMARY_PATH = DATA_DIR / "seed_summary.csv"
COMMENTS_RAW_PATH = DATA_DIR / "comments_raw.csv"
COMMENTS_CLEAN_PATH = DATA_DIR / "comments_clean.csv"
COMMENTS_SCORED_PATH = DATA_DIR / "comments_scored.csv"
COMMENTS_RELEVANT_PATH = DATA_DIR / "comments_relevant.csv"
ANNOTATION_TEMPLATE_PATH = DATA_DIR / "annotation_candidates.csv"
RELEVANCE_SUMMARY_PATH = DATA_DIR / "relevance_summary.csv"
CRAWL_ERRORS_PATH = LOG_DIR / "crawl_errors.csv"

QUERY_TEMPLATES = (
    ("review", "{term} review"),
    ("camera", "{term} camera"),
    ("battery", "{term} battery"),
    ("issue", "{term} overheating issue problem bug"),
)


@dataclass(frozen=True)
class BucketRule:
    bucket: str
    family: str
    variant: str
    query_term: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()

    def title_in_scope(self, title: str) -> bool:
        lowered = title.casefold()
        includes = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in self.include_patterns)
        excludes = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in self.exclude_patterns)
        return includes and not excludes


BUCKET_RULES = (
    BucketRule(
        bucket="iPhone 17",
        family="Apple",
        variant="base",
        query_term="iphone 17",
        include_patterns=(r"\biphone 17\b",),
        exclude_patterns=(r"\biphone 17 pro\b", r"\biphone 17 pro max\b"),
    ),
    BucketRule(
        bucket="iPhone 17 Pro",
        family="Apple",
        variant="premium",
        query_term="iphone 17 pro",
        include_patterns=(r"\biphone 17 pro\b", r"\biphone 17 pro max\b"),
    ),
    BucketRule(
        bucket="Galaxy S26",
        family="Samsung",
        variant="base",
        query_term="galaxy s26",
        include_patterns=(r"\bgalaxy s26\b", r"\bs26\b"),
        exclude_patterns=(r"\bultra\b", r"\bplus\b", r"\+\b"),
    ),
    BucketRule(
        bucket="Galaxy S26 Ultra",
        family="Samsung",
        variant="premium",
        query_term="s26 ultra",
        include_patterns=(r"\bgalaxy s26 ultra\b", r"\bs26 ultra\b"),
    ),
    BucketRule(
        bucket="Pixel 10",
        family="Google",
        variant="base",
        query_term="pixel 10",
        include_patterns=(r"\bpixel 10\b",),
        exclude_patterns=(r"\bpro\b", r"\bxl\b", r"\bfold\b"),
    ),
    BucketRule(
        bucket="Pixel 10 Pro",
        family="Google",
        variant="premium",
        query_term="pixel 10 pro",
        include_patterns=(r"\bpixel 10 pro\b", r"\bpixel 10 pro xl\b", r"\bpixel 10 pro fold\b"),
    ),
    BucketRule(
        bucket="Xiaomi 15",
        family="Xiaomi",
        variant="base",
        query_term="xiaomi 15",
        include_patterns=(r"\bxiaomi 15\b",),
        exclude_patterns=(r"\bultra\b",),
    ),
    BucketRule(
        bucket="Xiaomi 15 Ultra",
        family="Xiaomi",
        variant="premium",
        query_term="xiaomi 15 ultra",
        include_patterns=(r"\bxiaomi 15 ultra\b",),
    ),
)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            "Missing .env file. Copy .env.example to .env and set YOUTUBE_API_KEY before running the pipeline."
        )

    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is missing from .env.")
    return api_key


def get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


def youtube_get(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    api_key: str,
    timeout: int = 30,
) -> dict[str, Any]:
    request_params = dict(params)
    request_params["key"] = api_key
    url = f"{YOUTUBE_API_BASE}/{endpoint}"
    response = session.get(url, params=request_params, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = response.text[:500].strip()
        raise requests.HTTPError(f"YouTube API request failed for {endpoint}: {message}", response=response) from exc
    return response.json()


def chunked(items: Iterable[str], size: int) -> Iterator[list[str]]:
    bucket: list[str] = []
    for item in items:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def parse_iso8601_duration(duration_text: str) -> int:
    pattern = re.compile(r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$")
    match = pattern.match(duration_text or "")
    if not match:
        return 0

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def clean_text(text: str) -> str:
    unescaped = html.unescape(text or "")
    return re.sub(r"\s+", " ", unescaped).strip()


def normalize_comment_text(text: str) -> str:
    return clean_text(text).casefold()
