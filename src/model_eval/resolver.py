from __future__ import annotations

import difflib
import re


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def suggest_similar(name: str, known_names: list[str], n: int = 3) -> list[str]:
    query = name.lower()
    lower_known = [k.lower() for k in known_names]

    matches = difflib.get_close_matches(query, lower_known, n=n)
    if matches:
        return matches

    norm_query = _normalize(name)
    if not norm_query:
        return []
    scored: list[tuple[float, str]] = []
    for known in lower_known:
        norm_known = _normalize(known)
        prefix = norm_known[: len(norm_query) + 2]
        ratio = difflib.SequenceMatcher(None, norm_query, prefix).ratio()
        if ratio >= 0.6:
            scored.append((ratio, known))

    scored.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    results: list[str] = []
    for _, known in scored:
        if known not in seen:
            seen.add(known)
            results.append(known)
            if len(results) >= n:
                break
    return results
