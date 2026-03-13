from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from crawler.youtube_pipeline import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic YouTube seed videos for the smartphone opinion dataset."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new seed videos to the existing dataset instead of replacing it.",
    )
    return parser.parse_args()


def load_existing_seed_df(append: bool) -> pd.DataFrame:
    if append and SEED_VIDEOS_PATH.exists():
        return pd.read_csv(SEED_VIDEOS_PATH)
    return pd.DataFrame(columns=SEED_COLUMNS)


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
        params: dict[str, Any] = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": page_size,
            "order": "relevance",
            "relevanceLanguage": "en",
            "publishedAfter": PUBLISHED_AFTER,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = youtube_get(session=session, endpoint="search", api_key=api_key, params=params)
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


def build_seed_rows(existing_seed_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api_key = load_api_key()
    session = build_session()
    search_results_per_query = get_env_int("SEARCH_RESULTS_PER_QUERY", DEFAULT_SEARCH_RESULTS_PER_QUERY)
    keep_per_query = get_env_int("KEEP_PER_QUERY", DEFAULT_KEEP_PER_QUERY)
    max_per_channel = get_env_int("MAX_PER_CHANNEL_PER_BUCKET", DEFAULT_MAX_PER_CHANNEL_PER_BUCKET)

    global_selected_video_ids: set[str] = set(existing_seed_df["video_id"].astype(str)) if not existing_seed_df.empty else set()
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for bucket_rule in tqdm(BUCKET_RULES, desc="Buckets"):
        existing_bucket_df = existing_seed_df[existing_seed_df["bucket"] == bucket_rule.bucket]
        channel_counts = Counter(existing_bucket_df["channel_title"].dropna().tolist())

        for category, template in QUERY_TEMPLATES:
            search_query = template.format(term=bucket_rule.query_term)
            existing_for_query_df = existing_bucket_df[existing_bucket_df["category"] == category]
            existing_for_query = len(existing_for_query_df)
            selected_for_query = 0

            if existing_for_query >= keep_per_query:
                summary_rows.append(
                    {
                        "bucket": bucket_rule.bucket,
                        "family": bucket_rule.family,
                        "variant": bucket_rule.variant,
                        "category": category,
                        "search_query": search_query,
                        "search_results_requested": search_results_per_query,
                        "search_results_returned": 0,
                        "already_present": existing_for_query,
                        "selected_this_run": 0,
                        "selected_total_after_run": existing_for_query,
                        "unique_channels": len(set(existing_bucket_df["channel_title"].dropna().tolist())),
                        "error": "",
                    }
                )
                continue

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
                        "already_present": existing_for_query,
                        "selected_this_run": 0,
                        "selected_total_after_run": existing_for_query,
                        "unique_channels": len(set(existing_bucket_df["channel_title"].dropna().tolist())),
                        "error": str(exc),
                    }
                )
                continue

            for video_id in candidate_ids:
                if existing_for_query + selected_for_query >= keep_per_query:
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

            new_selected_rows = [
                row for row in seed_rows if row["bucket"] == bucket_rule.bucket and row["category"] == category
            ]
            summary_rows.append(
                {
                    "bucket": bucket_rule.bucket,
                    "family": bucket_rule.family,
                    "variant": bucket_rule.variant,
                    "category": category,
                    "search_query": search_query,
                    "search_results_requested": search_results_per_query,
                    "search_results_returned": len(candidate_ids),
                    "already_present": existing_for_query,
                    "selected_this_run": selected_for_query,
                    "selected_total_after_run": existing_for_query + selected_for_query,
                    "unique_channels": len(
                        set(existing_bucket_df["channel_title"].dropna().tolist())
                        | {row["channel_title"] for row in new_selected_rows}
                    ),
                    "error": "",
                }
            )

    return seed_rows, summary_rows


def save_outputs(existing_seed_df: pd.DataFrame, seed_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    new_seed_df = pd.DataFrame(seed_rows, columns=SEED_COLUMNS)
    seed_df = pd.concat([existing_seed_df, new_seed_df], ignore_index=True)
    seed_df = seed_df.drop_duplicates(subset=["video_id"], keep="first")
    seed_df.sort_values(by=["bucket", "category", "published_at", "video_id"], inplace=True)
    seed_df.to_csv(SEED_VIDEOS_PATH, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by=["bucket", "category"], inplace=True)
    summary_df.to_csv(SEED_SUMMARY_PATH, index=False)


def main() -> None:
    ensure_directories()
    args = parse_args()
    existing_seed_df = load_existing_seed_df(append=args.append)
    before_count = len(existing_seed_df)
    seed_rows, summary_rows = build_seed_rows(existing_seed_df=existing_seed_df)
    save_outputs(existing_seed_df=existing_seed_df, seed_rows=seed_rows, summary_rows=summary_rows)
    print(f"Saved {before_count + len(seed_rows)} total selected videos to {SEED_VIDEOS_PATH}.")
    print(f"Added {len(seed_rows)} new seed videos in this run.")
    print(f"Saved discovery summary to {SEED_SUMMARY_PATH}.")


if __name__ == "__main__":
    main()
