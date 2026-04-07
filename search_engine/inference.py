from __future__ import annotations

import re


def bucket_pattern(bucket: str, all_buckets: list[str]) -> re.Pattern[str]:
    lowered = (bucket or "").casefold()
    parts = re.findall(r"[a-z0-9]+", lowered)
    if not parts:
        return re.compile(r"a^", flags=re.IGNORECASE)

    token_expr = r"\W*".join(re.escape(p) for p in parts)
    premium_suffixes = r"(pro|max|plus|ultra|mini|fold|flip|edge|fe)"

    is_base_prefix = any(
        other.casefold().startswith(lowered + " ") for other in all_buckets if other != bucket
    )
    if is_base_prefix:
        return re.compile(
            rf"\b{token_expr}\b(?!\W*{premium_suffixes}\b)",
            flags=re.IGNORECASE,
        )

    return re.compile(rf"\b{token_expr}\b", flags=re.IGNORECASE)


def infer_bucket_from_query(query_text: str, all_buckets: list[str]) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    ordered = sorted(all_buckets, key=lambda x: len(x), reverse=True)
    matches: list[str] = []
    for bucket in ordered:
        pattern = bucket_pattern(bucket, all_buckets)
        if pattern.search(q):
            matches.append(bucket)

    if matches:
        return matches[0]
    return None


def infer_category_from_query(query_text: str) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    category_patterns: dict[str, tuple[str, ...]] = {
        "camera": (r"\bcamera\b", r"\bphoto\b", r"\bvideo\b", r"\bzoom\b", r"\bsensor\b"),
        "battery": (r"\bbattery\b", r"\bcharging\b", r"\bmah\b", r"\bmagsafe\b"),
        "issue": (
            r"\bissue\b",
            r"\bproblem\b",
            r"\boverheat(?:ing)?\b",
            r"\bbug\b",
            r"\bfault\b",
            r"\bdefect\b",
        ),
        "review": (r"\breview\b", r"\bimpression\b", r"\bthoughts\b", r"\bopinion\b"),
    }

    matches: list[str] = []
    for category, patterns in category_patterns.items():
        if any(re.search(p, q) for p in patterns):
            matches.append(category)

    if len(matches) == 1:
        return matches[0]
    return None


def infer_family_from_query(query_text: str, all_families: list[str]) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    matches: list[str] = []
    for family in all_families:
        aliases = [family.casefold()]
        if family.casefold() == "apple":
            aliases.extend(["iphone", "ios"])
        elif family.casefold() == "samsung":
            aliases.extend(["galaxy"])
        elif family.casefold() == "google":
            aliases.extend(["pixel"])
        elif family.casefold() == "xiaomi":
            aliases.extend(["mi"])

        if any(re.search(rf"\b{re.escape(alias)}\b", q) for alias in aliases):
            matches.append(family)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def infer_unknown_model_from_query(
    query_text: str,
    all_buckets: list[str],
    all_families: list[str],
) -> str | None:
    q = (query_text or "").strip()
    if not q:
        return None

    if infer_bucket_from_query(q, all_buckets) is not None:
        return None

    inferred_family = infer_family_from_query(q, all_families)
    if inferred_family is None:
        return None

    lowered = q.casefold()
    has_model_number = bool(re.search(r"\b(?:\d{1,3}|[a-z]\d{1,3})\b", lowered))
    if not has_model_number:
        return None

    alias_map = {
        "apple": ["iphone", "ios", "apple"],
        "samsung": ["galaxy", "samsung"],
        "google": ["pixel", "google"],
        "xiaomi": ["xiaomi", "mi", "redmi"],
    }
    aliases = alias_map.get(inferred_family.casefold(), [inferred_family.casefold()])
    alias_expr = "|".join(re.escape(a) for a in aliases)

    candidate_patterns = [
        rf"\b(?:{alias_expr})\W*[a-z]?\W*\d{{1,3}}(?:\W*(?:pro|max|plus|ultra|mini|fold|flip|edge|fe))?\b",
        r"\b[a-z]\d{1,3}(?:\W*(?:pro|max|plus|ultra|mini|fold|flip|edge|fe))?\b",
    ]
    for pattern in candidate_patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return q[match.start():match.end()].strip()

    return q
