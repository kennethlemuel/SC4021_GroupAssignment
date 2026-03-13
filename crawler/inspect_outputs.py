from __future__ import annotations

from pathlib import Path

import pandas as pd

from crawler.youtube_pipeline import COMMENTS_CLEAN_PATH, SEED_VIDEOS_PATH


def read_csv_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Missing {path}.")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"{path} exists but is empty.")
    return df


def print_video_stats(seed_df: pd.DataFrame) -> None:
    print(f"Total number of videos: {len(seed_df)}")
    print("\nVideos by bucket and category:")
    print(
        seed_df.groupby(["bucket", "category"])
        .size()
        .rename("videos")
        .reset_index()
        .sort_values(by=["bucket", "category"])
        .to_string(index=False)
    )

    print("\nTop channels by selected video count:")
    print(
        seed_df["channel_title"]
        .value_counts()
        .head(10)
        .rename_axis("channel_title")
        .reset_index(name="selected_videos")
        .to_string(index=False)
    )


def print_comment_stats(comments_df: pd.DataFrame) -> None:
    print(f"\nTotal comments: {len(comments_df)}")
    print("\nComments by bucket:")
    print(
        comments_df.groupby("bucket")
        .size()
        .rename("comments")
        .reset_index()
        .sort_values(by="bucket")
        .to_string(index=False)
    )

    print("\nRandom sample comments per bucket:")
    for bucket in sorted(comments_df["bucket"].dropna().unique()):
        bucket_df = comments_df[comments_df["bucket"] == bucket]
        sample_df = bucket_df.sample(n=min(3, len(bucket_df)), random_state=42)
        print(f"\n[{bucket}]")
        for _, row in sample_df.iterrows():
            print(f"- {row['text']}")


def main() -> None:
    seed_df = read_csv_if_present(SEED_VIDEOS_PATH)
    comments_df = read_csv_if_present(COMMENTS_CLEAN_PATH)

    if seed_df is None and comments_df is None:
        print("No output files found yet. Run the seed builder first, then the comment crawler.")
        return

    if seed_df is not None and not seed_df.empty:
        print_video_stats(seed_df)

    if comments_df is not None and not comments_df.empty:
        print_comment_stats(comments_df)
    elif comments_df is None:
        print("\nComment output is not available yet. Run python crawl_youtube_comments.py after building seeds.")


if __name__ == "__main__":
    main()
