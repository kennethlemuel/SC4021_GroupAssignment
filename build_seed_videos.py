from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from youtube_pipeline import (
    BUCKET_RULES,
    DEFAULT_KEEP_PER_QUERY,
    DEFAULT_MAX_PER_CHANNEL_PER_BUCKET,
    DEFAULT_SEARCH_RESULTS_PER_QUERY,
    MIN_DURATION_SECONDS,
    PUBLISHED_AFTER,
    QUERY_TEMPLATES,
    SEED_SUMMARY_PATH,
    SEED_VIDEOS_PATH,
    build_session,
    chunked,
    ensure_directories,
    get_env_int,
    load_api_key,
    parse_iso8601_duration,
    youtube_get,
)


SEED_COLUMNS = [
    "video_id",
    "video_url",
    "bucket",
    "family",
    "variant",
    "category",
    "search_query",
    "video_title",
    "channel_title",
    "published_at",
    "duration_seconds",
]


def search_video_ids(
    session: requests.Session,
    api_key: str,
    search_query: str,
    max_results: int,
) -> list[str]:
    video_ids: list[str] = []
    seen_ids: set[str] = set()
    next_page_token: str | None = None

    while len(video_ids) < max_results:
        page_size = min(50, max_results - len(video_ids))
        response = youtube_get(
            session=session,
            endpoint="search",
            api_key=api_key,
            params={
                "part": "snippet",
                "q": search_query,
                "type": "video",
                "maxResults": page_size,
                "order": "relevance",
                "relevanceLanguage": "en",
                "publishedAfter": PUBLISHED_AFTER,
                "pageToken": next_page_token,
            }
            if next_page_token
            else {
                "part": "snippet",
                "q": search_query,
                "type": "video",
                "maxResults": page_size,
                "order": "relevance",
                "relevanceLanguage": "en",
                "publishedAfter": PUBLISHED_AFTER,
            },
        )

        items = response.get("items", [])
        if not items:
            break

        for item in items:
            video_id = item.get("id", {}).get("videoId")
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                video_ids.append(video_id)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def fetch_video_details(
    session: requests.Session,
    api_key: str,
    video_ids: list[str],
) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for video_id_chunk in chunked(video_ids, 50):
        response = youtube_get(
            session=session,
            endpoint="videos",
            api_key=api_key,
            params={
                "part": "snippet,contentDetails",
                "id": ",".join(video_id_chunk),
                "maxResults": len(video_id_chunk),
            },
        )
        for item in response.get("items", []):
            details[item["id"]] = item
    return details


def build_seed_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api_key = load_api_key()
    session = build_session()
    search_results_per_query = get_env_int("SEARCH_RESULTS_PER_QUERY", DEFAULT_SEARCH_RESULTS_PER_QUERY)
    keep_per_query = get_env_int("KEEP_PER_QUERY", DEFAULT_KEEP_PER_QUERY)
    max_per_channel = get_env_int("MAX_PER_CHANNEL_PER_BUCKET", DEFAULT_MAX_PER_CHANNEL_PER_BUCKET)

    global_selected_video_ids: set[str] = set()
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for bucket_rule in tqdm(BUCKET_RULES, desc="Buckets"):
        channel_counts = Counter()

        for category, template in QUERY_TEMPLATES:
            search_query = template.format(term=bucket_rule.query_term)
            selected_for_query = 0

            try:
                candidate_ids = search_video_ids(
                    session=session,
                    api_key=api_key,
                    search_query=search_query,
                    max_results=search_results_per_query,
                )
                details_by_id = fetch_video_details(session=session, api_key=api_key, video_ids=candidate_ids)
            except requests.RequestException as exc:
                summary_rows.append(
                    {
                        "bucket": bucket_rule.bucket,
                        "family": bucket_rule.family,
                        "variant": bucket_rule.variant,
                        "category": category,
                        "search_query": search_query,
                        "search_results_requested": search_results_per_query,
                        "search_results_returned": 0,
                        "selected_videos": 0,
                        "unique_channels": 0,
                        "error": str(exc),
                    }
                )
                continue

            for video_id in candidate_ids:
                if selected_for_query >= keep_per_query:
                    break
                if video_id in global_selected_video_ids:
                    continue

                item = details_by_id.get(video_id)
                if not item:
                    continue

                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                title = snippet.get("title", "")
                channel_title = snippet.get("channelTitle", "")
                duration_seconds = parse_iso8601_duration(content_details.get("duration", ""))

                if duration_seconds < MIN_DURATION_SECONDS:
                    continue
                if not bucket_rule.title_in_scope(title):
                    continue
                if channel_counts[channel_title] >= max_per_channel:
                    continue

                row = {
                    "video_id": video_id,
                    "video_url": f"https://www.youtube.com/watch?v={video_id}",
                    "bucket": bucket_rule.bucket,
                    "family": bucket_rule.family,
                    "variant": bucket_rule.variant,
                    "category": category,
                    "search_query": search_query,
                    "video_title": title,
                    "channel_title": channel_title,
                    "published_at": snippet.get("publishedAt", ""),
                    "duration_seconds": duration_seconds,
                }
                seed_rows.append(row)
                global_selected_video_ids.add(video_id)
                channel_counts[channel_title] += 1
                selected_for_query += 1

            selected_rows = [row for row in seed_rows if row["bucket"] == bucket_rule.bucket and row["category"] == category]
            summary_rows.append(
                {
                    "bucket": bucket_rule.bucket,
                    "family": bucket_rule.family,
                    "variant": bucket_rule.variant,
                    "category": category,
                    "search_query": search_query,
                    "search_results_requested": search_results_per_query,
                    "search_results_returned": len(candidate_ids),
                    "selected_videos": selected_for_query,
                    "unique_channels": len({row["channel_title"] for row in selected_rows}),
                    "error": "",
                }
            )

    return seed_rows, summary_rows


def save_outputs(seed_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    seed_df = pd.DataFrame(seed_rows, columns=SEED_COLUMNS).drop_duplicates(subset=["video_id"])
    seed_df.sort_values(by=["bucket", "category", "published_at", "video_id"], inplace=True)
    seed_df.to_csv(SEED_VIDEOS_PATH, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by=["bucket", "category"], inplace=True)
    summary_df.to_csv(SEED_SUMMARY_PATH, index=False)


def main() -> None:
    ensure_directories()
    seed_rows, summary_rows = build_seed_rows()
    save_outputs(seed_rows=seed_rows, summary_rows=summary_rows)
    print(f"Saved {len(seed_rows)} selected videos to {SEED_VIDEOS_PATH}.")
    print(f"Saved discovery summary to {SEED_SUMMARY_PATH}.")


if __name__ == "__main__":
    main()
