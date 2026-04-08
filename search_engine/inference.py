from __future__ import annotations

import re


PREMIUM_SUFFIXES = ("pro", "max", "plus", "ultra", "mini", "fold", "flip", "edge", "fe")
GENERIC_MODEL_PREFIXES = {"galaxy", "iphone", "pixel", "xiaomi", "redmi", "mi"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _family_aliases(family: str) -> list[str]:
    f = (family or "").casefold()
    if f == "apple":
        return ["apple", "iphone", "ios"]
    if f == "samsung":
        return ["samsung", "galaxy"]
    if f == "google":
        return ["google", "pixel"]
    if f == "xiaomi":
        return ["xiaomi", "mi", "redmi"]
    return [f] if f else []


def _bucket_alias_token_sequences(bucket: str) -> list[list[str]]:
    tokens = _tokenize(bucket)
    if not tokens:
        return []

    aliases: list[list[str]] = [tokens]
    if tokens[0] in GENERIC_MODEL_PREFIXES and len(tokens) > 1:
        core = tokens[1:]
        if any(any(ch.isdigit() for ch in t) for t in core):
            aliases.append(core)

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for alias in aliases:
        key = tuple(alias)
        if key not in seen:
            seen.add(key)
            unique.append(alias)
    return unique


def _is_base_prefix(alias_tokens: list[str], bucket: str, all_buckets: list[str]) -> bool:
    for other in all_buckets:
        if other == bucket:
            continue
        for other_alias in _bucket_alias_token_sequences(other):
            if len(other_alias) > len(alias_tokens) and other_alias[: len(alias_tokens)] == alias_tokens:
                return True
    return False


def _find_bucket_match_span(query_text: str, bucket: str, all_buckets: list[str]) -> tuple[int, int] | None:
    q = query_text or ""
    if not q.strip():
        return None

    best_match: tuple[int, int] | None = None
    for alias_tokens in _bucket_alias_token_sequences(bucket):
        token_expr = r"\W*".join(re.escape(token) for token in alias_tokens)
        is_base_alias = _is_base_prefix(alias_tokens, bucket, all_buckets)
        if is_base_alias:
            pattern = re.compile(
                rf"\b{token_expr}\b(?!\W*(?:{'|'.join(PREMIUM_SUFFIXES)})\b)",
                flags=re.IGNORECASE,
            )
        else:
            pattern = re.compile(rf"\b{token_expr}\b", flags=re.IGNORECASE)

        for match in pattern.finditer(q):
            span = (match.start(), match.end())
            if best_match is None:
                best_match = span
                continue
            best_len = best_match[1] - best_match[0]
            curr_len = span[1] - span[0]
            if curr_len > best_len or (curr_len == best_len and span[0] < best_match[0]):
                best_match = span

    return best_match


def bucket_pattern(bucket: str, all_buckets: list[str]) -> re.Pattern[str]:
    aliases = _bucket_alias_token_sequences(bucket)
    if not aliases:
        return re.compile(r"a^", flags=re.IGNORECASE)

    # Keep compatibility with existing callers by using the first alias as representative.
    parts = aliases[0]
    token_expr = r"\W*".join(re.escape(p) for p in parts)

    is_base_prefix = _is_base_prefix(parts, bucket, all_buckets)
    if is_base_prefix:
        return re.compile(
            rf"\b{token_expr}\b(?!\W*(?:{'|'.join(PREMIUM_SUFFIXES)})\b)",
            flags=re.IGNORECASE,
        )

    return re.compile(rf"\b{token_expr}\b", flags=re.IGNORECASE)


def infer_bucket_from_query(query_text: str, all_buckets: list[str]) -> str | None:
    q = (query_text or "").strip().casefold()
    if not q:
        return None

    ordered = sorted(all_buckets, key=lambda x: len(x), reverse=True)
    for bucket in ordered:
        if _find_bucket_match_span(q, bucket, all_buckets) is not None:
            return bucket
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
        aliases = _family_aliases(family)

        if any(re.search(rf"\b{re.escape(alias)}\b", q) for alias in aliases):
            matches.append(family)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def normalize_query_to_canonical_model(
    query_text: str,
    bucket: str,
    all_buckets: list[str],
    family: str | None = None,
) -> str:
    query = query_text or ""
    if not query.strip() or not bucket:
        return query_text

    span = _find_bucket_match_span(query, bucket, all_buckets)
    if span is None:
        return query_text

    include_family = False
    if family:
        aliases = _family_aliases(family)
        lowered = query.casefold()
        include_family = not any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases)

    replacement = f"{family} {bucket}" if include_family and family else bucket
    normalized = f"{query[:span[0]]}{replacement}{query[span[1]:]}"
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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
