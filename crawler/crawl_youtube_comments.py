from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from crawler.youtube_pipeline import (
    COMMENTS_CLEAN_PATH,
    COMMENTS_RAW_PATH,
    CRAWL_ERRORS_PATH,
    DEFAULT_MAX_THREAD_PAGES_PER_VIDEO,
    SEED_VIDEOS_PATH,
    build_session,
    clean_text,
    ensure_directories,
    get_env_int,
    load_api_key,
    normalize_comment_text,
    youtube_get,
)


CLEAN_COLUMNS = [
    "video_id",
    "video_url",
    "bucket",
    "family",
    "variant",
    "category",
    "search_query",
    "video_title",
    "channel_title",
    "video_published_at",
    "comment_id",
    "parent_id",
    "is_reply",
    "text",
    "like_count",
    "published_at",
    "updated_at",
]

RAW_COLUMNS = CLEAN_COLUMNS + ["text_normalized"]
ERROR_COLUMNS = ["video_id", "video_title", "stage", "parent_id", "status_code", "message"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl YouTube comments and replies for seed videos.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new crawl results to existing outputs instead of replacing them.",
    )
    return parser.parse_args()


def load_seed_videos() -> pd.DataFrame:
    if not SEED_VIDEOS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SEED_VIDEOS_PATH}. Run python build_seed_videos.py before crawling comments."
        )

    seed_df = pd.read_csv(SEED_VIDEOS_PATH)
    if seed_df.empty:
        raise RuntimeError(f"{SEED_VIDEOS_PATH} exists but contains no videos to crawl.")
    return seed_df


def load_existing_df(path, columns: list[str], append: bool) -> pd.DataFrame:
    if append and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def build_comment_row(
    seed_row: dict[str, Any],
    comment_id: str,
    parent_id: str,
    is_reply: bool,
    snippet: dict[str, Any],
) -> dict[str, Any]:
    text = clean_text(snippet.get("textOriginal") or snippet.get("textDisplay") or "")
    return {
        "video_id": seed_row["video_id"],
        "video_url": seed_row["video_url"],
        "bucket": seed_row["bucket"],
        "family": seed_row["family"],
        "variant": seed_row["variant"],
        "category": seed_row["category"],
        "search_query": seed_row["search_query"],
        "video_title": seed_row["video_title"],
        "channel_title": seed_row["channel_title"],
        "video_published_at": seed_row["published_at"],
        "comment_id": comment_id,
        "parent_id": parent_id,
        "is_reply": is_reply,
        "text": text,
        "like_count": int(snippet.get("likeCount", 0) or 0),
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
        "text_normalized": normalize_comment_text(text),
    }


def fetch_replies(
    session: requests.Session,
    api_key: str,
    seed_row: dict[str, Any],
    parent_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reply_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    next_page_token: str | None = None

    while True:
        try:
            params: dict[str, Any] = {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": 100,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            response = youtube_get(session=session, endpoint="comments", api_key=api_key, params=params)
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", "")
            errors.append(
                {
                    "video_id": seed_row["video_id"],
                    "video_title": seed_row["video_title"],
                    "stage": "comments.list",
                    "parent_id": parent_id,
                    "status_code": status_code,
                    "message": str(exc),
                }
            )
            break

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            reply_rows.append(
                build_comment_row(
                    seed_row=seed_row,
                    comment_id=item.get("id", ""),
                    parent_id=parent_id,
                    is_reply=True,
                    snippet=snippet,
                )
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return reply_rows, errors


def crawl_comments_for_video(
    session: requests.Session,
    api_key: str,
    seed_row: dict[str, Any],
    max_thread_pages: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    next_page_token: str | None = None
    page_count = 0

    while page_count < max_thread_pages:
        params: dict[str, Any] = {
            "part": "snippet",
            "videoId": seed_row["video_id"],
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "relevance",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = youtube_get(session=session, endpoint="commentThreads", api_key=api_key, params=params)
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", "")
            errors.append(
                {
                    "video_id": seed_row["video_id"],
                    "video_title": seed_row["video_title"],
                    "stage": "commentThreads.list",
                    "parent_id": "",
                    "status_code": status_code,
                    "message": str(exc),
                }
            )
            break

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            top_level = snippet.get("topLevelComment", {})
            top_level_snippet = top_level.get("snippet", {})
            parent_id = top_level.get("id", "")
            rows.append(
                build_comment_row(
                    seed_row=seed_row,
                    comment_id=parent_id,
                    parent_id="",
                    is_reply=False,
                    snippet=top_level_snippet,
                )
            )

            if int(snippet.get("totalReplyCount", 0) or 0) > 0 and parent_id:
                reply_rows, reply_errors = fetch_replies(
                    session=session,
                    api_key=api_key,
                    seed_row=seed_row,
                    parent_id=parent_id,
                )
                rows.extend(reply_rows)
                errors.extend(reply_errors)

        page_count += 1
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return rows, errors


def clean_comments(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=CLEAN_COLUMNS)

    cleaned = raw_df.copy()
    cleaned["text"] = cleaned["text"].fillna("").map(clean_text)
    cleaned["text_normalized"] = cleaned["text"].map(normalize_comment_text)
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned[~cleaned["text_normalized"].isin({"[deleted]", "[removed]"})]
    cleaned = cleaned.drop_duplicates(subset=["comment_id"], keep="first")
    cleaned = cleaned.drop_duplicates(subset=["video_id", "text_normalized"], keep="first")
    cleaned = cleaned[CLEAN_COLUMNS]
    cleaned.sort_values(by=["bucket", "video_id", "is_reply", "published_at", "comment_id"], inplace=True)
    return cleaned.reset_index(drop=True)


def save_outputs(
    existing_raw_df: pd.DataFrame,
    existing_error_df: pd.DataFrame,
    raw_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    new_raw_df = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)
    raw_df = pd.concat([existing_raw_df, new_raw_df], ignore_index=True)
    raw_df = raw_df.drop_duplicates(subset=["comment_id"], keep="first")
    raw_df.to_csv(COMMENTS_RAW_PATH, index=False)

    clean_df = clean_comments(raw_df=raw_df)
    clean_df.to_csv(COMMENTS_CLEAN_PATH, index=False)

    new_error_df = pd.DataFrame(error_rows, columns=ERROR_COLUMNS)
    error_df = pd.concat([existing_error_df, new_error_df], ignore_index=True)
    error_df.to_csv(CRAWL_ERRORS_PATH, index=False)

    print(f"Saved {len(raw_df)} raw comment records to {COMMENTS_RAW_PATH}.")
    print(f"Saved {len(clean_df)} cleaned comment records to {COMMENTS_CLEAN_PATH}.")
    print(f"Saved {len(error_df)} crawl errors to {CRAWL_ERRORS_PATH}.")


def main() -> None:
    ensure_directories()
    args = parse_args()
    api_key = load_api_key()
    seed_df = load_seed_videos()
    max_thread_pages = get_env_int("MAX_THREAD_PAGES_PER_VIDEO", DEFAULT_MAX_THREAD_PAGES_PER_VIDEO)
    session = build_session()
    existing_raw_df = load_existing_df(COMMENTS_RAW_PATH, RAW_COLUMNS, append=args.append)
    existing_error_df = load_existing_df(CRAWL_ERRORS_PATH, ERROR_COLUMNS, append=args.append)

    raw_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for seed_row in tqdm(seed_df.to_dict(orient="records"), desc="Videos"):
        rows, errors = crawl_comments_for_video(
            session=session,
            api_key=api_key,
            seed_row=seed_row,
            max_thread_pages=max_thread_pages,
        )
        raw_rows.extend(rows)
        error_rows.extend(errors)

    save_outputs(
        existing_raw_df=existing_raw_df,
        existing_error_df=existing_error_df,
        raw_rows=raw_rows,
        error_rows=error_rows,
    )


if __name__ == "__main__":
    main()
