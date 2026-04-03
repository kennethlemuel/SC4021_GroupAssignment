from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import pandas as pd

from crawler.youtube_pipeline import (
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

POSITIVE_PATTERNS = (
    r"\blove\b",
    r"\blike\b",
    r"\bgood\b",
    r"\bgreat\b",
    r"\bamazing\b",
    r"\bexcellent\b",
    r"\bsolid\b",
    r"\bbetter\b",
    r"\bbest\b",
    r"\bworth\b",
    r"\brecommend\b",
    r"\bimpress",
    r"\benjoy",
    r"\bfast\b",
    r"\bsmooth\b",
)

NEGATIVE_PATTERNS = (
    r"\bhate\b",
    r"\bbad\b",
    r"\bterrible\b",
    r"\bawful\b",
    r"\bworse\b",
    r"\bworst\b",
    r"\bdisappoint",
    r"\bunderwhelm",
    r"\bpoor\b",
    r"\boverheat",
    r"\bheat\b",
    r"\bbug\b",
    r"\bissue\b",
    r"\bproblem\b",
    r"\blag\b",
    r"\btrash\b",
    r"\bregret\b",
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

GENERIC_PHONE_PATTERNS = (
    r"\bphone\b",
    r"\bphones\b",
    r"\bdevice\b",
    r"\bflagship\b",
    r"\bhandset\b",
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

CREATOR_ONLY_PATTERNS = (
    r"\bgreat video\b",
    r"\bnice video\b",
    r"\bawesome video\b",
    r"\bthanks for (?:the )?video\b",
    r"\byou shake your hands\b",
    r"\bholy bot\b",
    r"\bappreciate you\b",
    r"\bwatching from\b",
    r"\bborrowed a book off\b",
)

REQUEST_PATTERNS = (
    r"^\s*please\b",
    r"\bcan you\b",
    r"\bmake\b.*\b(?:video|comparison|review|test)\b",
    r"\bspeed test\b",
    r"\bcamera test\b",
    r"\bcomparison\b",
)

COMMON_ENGLISH_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "do",
    "for",
    "from",
    "good",
    "great",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "like",
    "my",
    "not",
    "of",
    "on",
    "or",
    "really",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "very",
    "was",
    "what",
    "when",
    "which",
    "with",
    "would",
    "you",
    "your",
}

COMMENT_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "camera",
        (
            r"\bcamera\b",
            r"\bphoto\b",
            r"\bphotos\b",
            r"\bvideo\b",
            r"\bvideos\b",
            r"\bzoom\b",
            r"\bselfie\b",
            r"\blens\b",
            r"\bsensor\b",
            r"\bdynamic range\b",
            r"\bimage quality\b",
        ),
    ),
    (
        "battery",
        (
            r"\bbattery\b",
            r"\bbattery life\b",
            r"\bdrain\b",
            r"\bdrains\b",
            r"\bscreen on time\b",
            r"\bendurance\b",
            r"\boverheat",
            r"\bthermals\b",
            r"\bheat\b",
        ),
    ),
    (
        "charging",
        (
            r"\bcharge\b",
            r"\bcharging\b",
            r"\bcharger\b",
            r"\bwireless charging\b",
            r"\bmagsafe\b",
            r"\bmagnetic charging\b",
            r"\b\d{2,3}w\b",
            r"\bwatt\b",
            r"\bwatts\b",
        ),
    ),
    (
        "display",
        (
            r"\bdisplay\b",
            r"\bscreen\b",
            r"\bpanel\b",
            r"\bbrightness\b",
            r"\brefresh rate\b",
            r"\b120hz\b",
            r"\b60hz\b",
            r"\bltpo\b",
            r"\bresolution\b",
            r"\bbezel",
        ),
    ),
    (
        "design",
        (
            r"\bdesign\b",
            r"\bbuild\b",
            r"\blook\b",
            r"\blooks\b",
            r"\bsize\b",
            r"\bsmall\b",
            r"\bcompact\b",
            r"\bweight\b",
            r"\btitanium\b",
            r"\baluminum\b",
            r"\baluminium\b",
            r"\bcorner",
        ),
    ),
    (
        "performance",
        (
            r"\bperformance\b",
            r"\bchip\b",
            r"\bchipset\b",
            r"\bprocessor\b",
            r"\bsnapdragon\b",
            r"\btensor\b",
            r"\bexynos\b",
            r"\blag\b",
            r"\blaggy\b",
            r"\bsmooth\b",
            r"\bgaming\b",
            r"\bbenchmark\b",
            r"\bthrottle",
        ),
    ),
    (
        "software",
        (
            r"\bsoftware\b",
            r"\bone ui\b",
            r"\bios\b",
            r"\bandroid\b",
            r"\bupdate\b",
            r"\bupdates\b",
            r"\bai\b",
            r"\bfeature\b",
            r"\bfeatures\b",
            r"\bui\b",
            r"\bbug\b",
            r"\bbugs\b",
            r"\binterface\b",
        ),
    ),
    (
        "price",
        (
            r"\bprice\b",
            r"\bcost\b",
            r"\bexpensive\b",
            r"\bcheap\b",
            r"\boverpriced\b",
            r"\bworth\b",
            r"\bvalue\b",
            r"\bdiscount\b",
            r"\bdeal\b",
            r"\bjustify\b",
            r"\b\d{3,4}\b",
        ),
    ),
    (
        "storage",
        (
            r"\bstorage\b",
            r"\bram\b",
            r"\bmemory\b",
            r"\b128gb\b",
            r"\b256gb\b",
            r"\b512gb\b",
            r"\b1tb\b",
            r"\bgb\b",
        ),
    ),
)

DEFAULT_MIN_COMMENT_WORDS = 4
DEFAULT_MAX_COMMENT_WORDS = 120


def has_match(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(bool(re.search(pattern, text)) for pattern in patterns)


def is_probably_english(text: str) -> bool:
    lowered = (text or "").casefold()
    tokens = re.findall(r"[a-z']+", lowered)
    if not tokens:
        return False

    non_space_chars = [char for char in lowered if not char.isspace()]
    ascii_ratio = (
        sum(ord(char) < 128 for char in non_space_chars) / len(non_space_chars)
        if non_space_chars
        else 0.0
    )
    english_hits = sum(token in COMMON_ENGLISH_WORDS for token in tokens)
    english_ratio = english_hits / max(len(tokens), 1)

    if ascii_ratio < 0.85:
        return False
    if english_hits >= 2:
        return True
    if len(tokens) <= 6 and english_hits >= 1:
        return True
    if (
        ascii_ratio >= 0.97
        and len(tokens) >= 3
        and (
            has_match(lowered, ASPECT_PATTERNS)
            or has_match(lowered, OPINION_PATTERNS)
            or has_match(lowered, COMPARISON_PATTERNS)
            or has_match(lowered, OTHER_PHONE_PATTERNS)
            or has_match(lowered, GENERIC_PHONE_PATTERNS)
            or any(has_match(lowered, patterns) for _, patterns in COMMENT_CATEGORY_PATTERNS)
        )
    ):
        return True
    return english_ratio >= 0.12


def first_match_position(text: str, patterns: tuple[str, ...]) -> int | None:
    positions = [match.start() for pattern in patterns for match in re.finditer(pattern, text)]
    if not positions:
        return None
    return min(positions)


def infer_comment_categories(text: str) -> dict[str, object]:
    lowered = (text or "").casefold()
    category_scores: dict[str, int] = {}
    category_positions: dict[str, int | None] = {}

    for category_name, patterns in COMMENT_CATEGORY_PATTERNS:
        score = count_matches(lowered, patterns)
        category_scores[category_name] = score
        category_positions[category_name] = first_match_position(lowered, patterns)

    positive_scores = {category: score for category, score in category_scores.items() if score > 0}
    if not positive_scores:
        return {
            "comment_category": "overall",
            "comment_category_tied": False,
            "comment_category_tie_candidates": "",
            "comment_category_top_score": 0,
        }

    top_score = max(positive_scores.values())
    tied_categories = [category for category, score in positive_scores.items() if score == top_score]
    tied_categories.sort(
        key=lambda category: (
            category_positions[category] is None,
            category_positions[category] if category_positions[category] is not None else 10**9,
        )
    )
    final_choice = "overall" if len(tied_categories) > 1 else tied_categories[0]

    return {
        "comment_category": final_choice,
        "comment_category_tied": len(tied_categories) > 1,
        "comment_category_tie_candidates": ", ".join(tied_categories),
        "comment_category_top_score": top_score,
    }


def score_comment(
    text: str,
    bucket: str,
    seed_category: str,
    min_words: int,
    max_words: int,
) -> dict[str, object]:
    lowered = (text or "").casefold()
    keywords = BUCKET_KEYWORDS[bucket]
    token_count = len(lowered.split())
    english = is_probably_english(lowered)

    model_mention = has_match(lowered, keywords.include)
    family_mention = has_match(lowered, keywords.family_terms)
    generic_phone_mention = has_match(lowered, GENERIC_PHONE_PATTERNS)
    aspect_mention = has_match(lowered, ASPECT_PATTERNS)
    opinion_mention = has_match(lowered, OPINION_PATTERNS)
    comparison_cue = has_match(lowered, COMPARISON_PATTERNS)
    multi_brand_mention = count_matches(lowered, OTHER_PHONE_PATTERNS) >= 2
    has_link = "http://" in lowered or "https://" in lowered or "www." in lowered
    low_signal = has_match(lowered, LOW_SIGNAL_PATTERNS)
    creator_only = has_match(lowered, CREATOR_ONLY_PATTERNS)
    request_only = has_match(lowered, REQUEST_PATTERNS) and not opinion_mention
    too_short = token_count < min_words
    too_long = token_count > max_words
    short = token_count <= 8
    phone_relevant = model_mention or family_mention or generic_phone_mention or aspect_mention
    category_info = infer_comment_categories(text=lowered)

    score = 0
    if model_mention:
        score += 4
    if family_mention:
        score += 2
    if generic_phone_mention:
        score += 1
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
    if too_long:
        score -= 2
    if low_signal:
        score -= 4
    if creator_only:
        score -= 4
    if request_only:
        score -= 3
    if not english:
        score -= 6
    if not phone_relevant:
        score -= 5

    if (
        not english
        or low_signal
        or creator_only
        or request_only
        or too_short
        or too_long
        or not phone_relevant
    ):
        tier = "low"
    elif score >= 7 and (model_mention or (family_mention and (aspect_mention or opinion_mention))):
        tier = "high"
    elif score >= 4 and (phone_relevant and (aspect_mention or opinion_mention or comparison_cue)):
        tier = "medium"
    else:
        tier = "low"

    keep_for_annotation = tier in {"high", "medium"}
    keep_for_sentiment = (
        tier == "high"
        or (
            tier == "medium"
            and english
            and phone_relevant
            and not too_long
            and (aspect_mention or opinion_mention or comparison_cue or model_mention or family_mention)
        )
    )

    return {
        "text_length_words": token_count,
        "is_english": english,
        "mentions_model": model_mention,
        "mentions_family": family_mention,
        "mentions_generic_phone": generic_phone_mention,
        "mentions_aspect": aspect_mention,
        "mentions_opinion": opinion_mention,
        "comparison_cue": comparison_cue or multi_brand_mention,
        "has_link": has_link,
        "low_signal": low_signal,
        "creator_only": creator_only,
        "request_only": request_only,
        "too_long": too_long,
        **category_info,
        "relevance_score": score,
        "relevance_tier": tier,
        "keep_for_annotation": keep_for_annotation,
        "keep_for_sentiment": keep_for_sentiment,
    }


def suggest_sentiment_label(text: str) -> dict[str, object]:
    lowered = (text or "").casefold()
    positive_score = count_matches(lowered, POSITIVE_PATTERNS)
    negative_score = count_matches(lowered, NEGATIVE_PATTERNS)

    if negative_score >= positive_score + 2 and negative_score >= 2:
        suggestion = "negative"
    elif positive_score >= negative_score + 2 and positive_score >= 2:
        suggestion = "positive"
    else:
        suggestion = "neutral"

    if max(positive_score, negative_score) >= 3:
        confidence = "high"
    elif max(positive_score, negative_score) >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "suggested_sentiment_label": suggestion,
        "sentiment_positive_score": positive_score,
        "sentiment_negative_score": negative_score,
        "suggestion_confidence": confidence,
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


def build_outputs(
    comments_df: pd.DataFrame,
    min_words: int,
    max_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_records = []
    for row in comments_df.to_dict(orient="records"):
        scored_row = dict(row)
        scored_row["seed_category"] = row["category"]
        scored_row.update(
            score_comment(
                text=row["text"],
                bucket=row["bucket"],
                seed_category=row["category"],
                min_words=min_words,
                max_words=max_words,
            )
        )
        scored_row.update(suggest_sentiment_label(text=row["text"]))
        scored_records.append(scored_row)

    scored_df = pd.DataFrame(scored_records)
    scored_df.sort_values(
        by=["bucket", "relevance_tier", "relevance_score", "like_count", "text_length_words", "published_at"],
        ascending=[True, True, False, False, True, True],
        inplace=True,
    )

    relevant_df = scored_df[scored_df["keep_for_sentiment"]].copy()
    relevant_df.sort_values(
        by=["bucket", "comment_category", "relevance_score", "like_count", "text_length_words"],
        ascending=[True, True, False, False, True],
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
            english_comments=("is_english", lambda s: int(s.sum())),
            high_relevance=("relevance_tier", lambda s: int((s == "high").sum())),
            medium_relevance=("relevance_tier", lambda s: int((s == "medium").sum())),
            low_relevance=("relevance_tier", lambda s: int((s == "low").sum())),
            sentiment_ready=("keep_for_sentiment", lambda s: int(s.sum())),
            avg_score=("relevance_score", "mean"),
            avg_words=("text_length_words", "mean"),
        )
        .reset_index()
    )
    summary_df["sentiment_ready_pct"] = (summary_df["sentiment_ready"] / summary_df["total_comments"] * 100).round(1)
    summary_df["avg_score"] = summary_df["avg_score"].round(2)
    summary_df["avg_words"] = summary_df["avg_words"].round(1)

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
        annotation_export = annotation_df.groupby("bucket", group_keys=False).head(limit_per_bucket).reset_index(drop=True)
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
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_COMMENT_WORDS,
        help="Minimum number of words required for a comment to be considered classification-ready.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_COMMENT_WORDS,
        help="Maximum number of words allowed for a comment to be considered classification-ready.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comments_df = load_comments()
    scored_df, relevant_df, annotation_df, summary_df = build_outputs(
        comments_df=comments_df,
        min_words=args.min_words,
        max_words=args.max_words,
    )
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
