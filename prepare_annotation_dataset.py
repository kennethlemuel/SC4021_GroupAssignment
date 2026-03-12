from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import pandas as pd

from youtube_pipeline import (
    ANNOTATION_TEMPLATE_PATH,
    COMMENTS_CLEAN_PATH,
    COMMENTS_RELEVANT_PATH,
    COMMENTS_SCORED_PATH,
    RELEVANCE_SUMMARY_PATH,
)


@dataclass(frozen=True)
class BucketKeywords:
    include: tuple[str, ...]
    family_terms: tuple[str, ...]


BUCKET_KEYWORDS: dict[str, BucketKeywords] = {
    "iPhone 17": BucketKeywords(
        include=(r"\biphone 17\b",),
        family_terms=(r"\biphone\b", r"\bapple\b", r"\bios\b"),
    ),
    "iPhone 17 Pro": BucketKeywords(
        include=(r"\biphone 17 pro\b", r"\biphone 17 pro max\b"),
        family_terms=(r"\biphone\b", r"\bapple\b", r"\bpro max\b"),
    ),
    "Galaxy S26": BucketKeywords(
        include=(r"\bgalaxy s26\b", r"\bs26\b"),
        family_terms=(r"\bsamsung\b", r"\bgalaxy\b"),
    ),
    "Galaxy S26 Ultra": BucketKeywords(
        include=(r"\bgalaxy s26 ultra\b", r"\bs26 ultra\b"),
        family_terms=(r"\bsamsung\b", r"\bgalaxy\b", r"\bultra\b"),
    ),
    "Pixel 10": BucketKeywords(
        include=(r"\bpixel 10\b",),
        family_terms=(r"\bpixel\b", r"\bgoogle\b"),
    ),
    "Pixel 10 Pro": BucketKeywords(
        include=(r"\bpixel 10 pro\b", r"\bpixel 10 pro xl\b", r"\bpixel 10 pro fold\b"),
        family_terms=(r"\bpixel\b", r"\bgoogle\b"),
    ),
    "Xiaomi 15": BucketKeywords(
        include=(r"\bxiaomi 15\b",),
        family_terms=(r"\bxiaomi\b", r"\bmi\b"),
    ),
    "Xiaomi 15 Ultra": BucketKeywords(
        include=(r"\bxiaomi 15 ultra\b",),
        family_terms=(r"\bxiaomi\b", r"\bmi\b", r"\bultra\b"),
    ),
}

ASPECT_PATTERNS = (
    r"\bcamera\b",
    r"\bbattery\b",
    r"\bcharging\b",
    r"\boverheat",
    r"\bheat\b",
    r"\bbug\b",
    r"\bissue\b",
    r"\bproblem\b",
    r"\bperformance\b",
    r"\bdisplay\b",
    r"\bscreen\b",
    r"\bphoto\b",
    r"\bphotos\b",
    r"\bvideo\b",
    r"\bzoom\b",
    r"\bprocessor\b",
    r"\bchip\b",
    r"\bthermals\b",
    r"\bai\b",
)

OPINION_PATTERNS = (
    r"\blove\b",
    r"\bhate\b",
    r"\bgood\b",
    r"\bgreat\b",
    r"\bbad\b",
    r"\bbetter\b",
    r"\bworse\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bbuy\b",
    r"\bworth\b",
    r"\bdisappoint",
    r"\bamazing\b",
    r"\bterrible\b",
    r"\bimprov",
    r"\bregret\b",
    r"\bupgrade\b",
    r"\bdowngrade\b",
    r"\brecommend\b",
    r"\bprefer\b",
)

COMPARISON_PATTERNS = (
    r"\bvs\b",
    r"\bversus\b",
    r"\bthan\b",
    r"\bcompared to\b",
    r"\bswitch from\b",
    r"\bcoming from\b",
    r"\binstead of\b",
)

OTHER_PHONE_PATTERNS = (
    r"\biphone\b",
    r"\bapple\b",
    r"\bsamsung\b",
    r"\bgalaxy\b",
    r"\bpixel\b",
    r"\bgoogle\b",
    r"\bxiaomi\b",
    r"\bmi\b",
)

LOW_SIGNAL_PATTERNS = (
    r"^\s*first\s*!*\s*$",
    r"^\s*1st\s*!*\s*$",
    r"^\s*real first\s*!*\s*$",
    r"^\s*thank you[\s!.]*$",
    r"^\s*great video[\s!.]*$",
    r"^\s*great review[\s!.]*$",
    r"^\s*nice video[\s!.]*$",
    r"^\s*awesome video[\s!.]*$",
    r"^\s*go [a-z0-9_ ]+[\s!.]*$",
    r"^\s*\d{1,2}:\d{2}\b",
)


def has_match(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(bool(re.search(pattern, text)) for pattern in patterns)


def score_comment(text: str, bucket: str) -> dict[str, object]:
    lowered = (text or "").casefold()
    keywords = BUCKET_KEYWORDS[bucket]
    token_count = len(lowered.split())

    model_mention = has_match(lowered, keywords.include)
    family_mention = has_match(lowered, keywords.family_terms)
    aspect_mention = has_match(lowered, ASPECT_PATTERNS)
    opinion_mention = has_match(lowered, OPINION_PATTERNS)
    comparison_cue = has_match(lowered, COMPARISON_PATTERNS)
    multi_brand_mention = count_matches(lowered, OTHER_PHONE_PATTERNS) >= 2
    has_link = "http://" in lowered or "https://" in lowered or "www." in lowered
    low_signal = has_match(lowered, LOW_SIGNAL_PATTERNS)
    too_short = token_count <= 3
    short = token_count <= 5

    score = 0
    if model_mention:
        score += 4
    if family_mention:
        score += 2
    if aspect_mention:
        score += 2
    if opinion_mention:
        score += 2
    if comparison_cue:
        score += 1
    if multi_brand_mention:
        score += 1
    if token_count >= 8:
        score += 1
    if token_count >= 15:
        score += 1
    if has_link:
        score -= 2
    if short:
        score -= 1
    if too_short:
        score -= 2
    if low_signal:
        score -= 4

    if low_signal or too_short:
        tier = "low"
    elif score >= 6 and (model_mention or (family_mention and (aspect_mention or opinion_mention))):
        tier = "high"
    elif score >= 3 and (family_mention or aspect_mention or opinion_mention):
        tier = "medium"
    else:
        tier = "low"

    keep_for_annotation = tier in {"high", "medium"}
    keep_for_sentiment = tier == "high" or (tier == "medium" and (aspect_mention or opinion_mention))

    return {
        "text_length_words": token_count,
        "mentions_model": model_mention,
        "mentions_family": family_mention,
        "mentions_aspect": aspect_mention,
        "mentions_opinion": opinion_mention,
        "comparison_cue": comparison_cue or multi_brand_mention,
        "has_link": has_link,
        "low_signal": low_signal,
        "relevance_score": score,
        "relevance_tier": tier,
        "keep_for_annotation": keep_for_annotation,
        "keep_for_sentiment": keep_for_sentiment,
    }


def load_comments() -> pd.DataFrame:
    if not COMMENTS_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COMMENTS_CLEAN_PATH}. Run python crawl_youtube_comments.py before preparing annotations."
        )
    comments_df = pd.read_csv(COMMENTS_CLEAN_PATH)
    if comments_df.empty:
        raise RuntimeError(f"{COMMENTS_CLEAN_PATH} exists but contains no comments.")
    return comments_df


def build_outputs(comments_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_records = []
    for row in comments_df.to_dict(orient="records"):
        scored_row = dict(row)
        scored_row.update(score_comment(text=row["text"], bucket=row["bucket"]))
        scored_records.append(scored_row)

    scored_df = pd.DataFrame(scored_records)
    scored_df.sort_values(
        by=["bucket", "relevance_tier", "relevance_score", "like_count", "published_at"],
        ascending=[True, True, False, False, True],
        inplace=True,
    )

    relevant_df = scored_df[scored_df["keep_for_sentiment"]].copy()
    relevant_df.sort_values(
        by=["bucket", "relevance_score", "like_count", "text_length_words"],
        ascending=[True, False, False, False],
        inplace=True,
    )

    annotation_df = relevant_df.copy()
    annotation_df["sentiment_label"] = ""
    annotation_df["aspect_label"] = ""
    annotation_df["comparison_target"] = ""
    annotation_df["annotator_1"] = ""
    annotation_df["annotator_2"] = ""
    annotation_df["annotator_3"] = ""
    annotation_df["agreement_notes"] = ""

    summary_df = (
        scored_df.groupby("bucket")
        .agg(
            total_comments=("comment_id", "size"),
            high_relevance=("relevance_tier", lambda s: int((s == "high").sum())),
            medium_relevance=("relevance_tier", lambda s: int((s == "medium").sum())),
            low_relevance=("relevance_tier", lambda s: int((s == "low").sum())),
            sentiment_ready=("keep_for_sentiment", lambda s: int(s.sum())),
            avg_score=("relevance_score", "mean"),
        )
        .reset_index()
    )
    summary_df["sentiment_ready_pct"] = (summary_df["sentiment_ready"] / summary_df["total_comments"] * 100).round(1)
    summary_df["avg_score"] = summary_df["avg_score"].round(2)

    return scored_df, relevant_df, annotation_df, summary_df


def save_outputs(
    scored_df: pd.DataFrame,
    relevant_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    limit_per_bucket: int | None,
) -> None:
    scored_df.to_csv(COMMENTS_SCORED_PATH, index=False)
    relevant_df.to_csv(COMMENTS_RELEVANT_PATH, index=False)
    summary_df.to_csv(RELEVANCE_SUMMARY_PATH, index=False)

    if limit_per_bucket is not None:
        annotation_export = (
            annotation_df.groupby("bucket", group_keys=False)
            .head(limit_per_bucket)
            .reset_index(drop=True)
        )
    else:
        annotation_export = annotation_df.reset_index(drop=True)

    annotation_export.to_csv(ANNOTATION_TEMPLATE_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score YouTube comments for phone relevance and prepare an annotation-ready dataset."
    )
    parser.add_argument(
        "--limit-per-bucket",
        type=int,
        default=None,
        help="Optional cap on the number of annotation rows exported per bucket.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comments_df = load_comments()
    scored_df, relevant_df, annotation_df, summary_df = build_outputs(comments_df)
    save_outputs(
        scored_df=scored_df,
        relevant_df=relevant_df,
        annotation_df=annotation_df,
        summary_df=summary_df,
        limit_per_bucket=args.limit_per_bucket,
    )

    print(summary_df.to_string(index=False))
    print(f"\nSaved scored comments to {COMMENTS_SCORED_PATH}.")
    print(f"Saved sentiment-ready comments to {COMMENTS_RELEVANT_PATH}.")
    print(f"Saved relevance summary to {RELEVANCE_SUMMARY_PATH}.")
    if args.limit_per_bucket is None:
        print(f"Saved full annotation template to {ANNOTATION_TEMPLATE_PATH}.")
    else:
        print(
            f"Saved annotation template capped at {args.limit_per_bucket} rows per bucket to {ANNOTATION_TEMPLATE_PATH}."
        )


if __name__ == "__main__":
    main()
