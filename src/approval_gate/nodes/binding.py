"""Which identifiers are reachable from what the user named.

Trust spreads from the user's request through the records that earlier tool
calls returned, any number of steps, but only through identifier-like values
(free text such as file contents and email bodies is where injections live)
that are distinctive (a value like "me", shared by every transaction, would
link everything).
"""

from __future__ import annotations

import re
from collections import Counter

from approval_gate.nodes.arguments import appears_in, normalize, normalize_text

_DATE_OR_TIME = re.compile(r"\d{4}-\d{2}-\d{2}.*|\d{1,2}:\d{2}.*")
MAX_SHARED = 3   # a value in more records than this does not link them


def records(result) -> list[dict]:
    """Every dict-like object in a tool result, however deeply nested."""
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if isinstance(result, dict):
        found = [result]
        for value in result.values():
            found += records(value)
        return found
    if isinstance(result, (list, tuple)):
        return [r for item in result for r in records(item)]
    return []


def _is_identifier(value) -> bool:
    if isinstance(value, (bool, float)):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    return (0 < len(text) <= 64 and "\n" not in text and len(text.split()) <= 4
            and not _DATE_OR_TIME.fullmatch(text))


def canon(value) -> str:
    """Same form for an id whether it arrives as 7 or "7"."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return normalize(value)


def _identifiers(record: dict) -> set[str]:
    values = []
    for value in record.values():
        values += value if isinstance(value, (list, tuple)) else [value]
    return {canon(v) for v in values if _is_identifier(v)}


def reachable(user_prompt: str, all_records: list[dict]) -> set[str]:
    prompt = normalize_text(user_prompt)
    # The same lookup repeated must not make its values look shared
    ids = [set(s) for s in {frozenset(_identifiers(r)) for r in all_records} if s]
    shared = Counter(v for record_ids in ids for v in record_ids)
    linkers = [{v for v in record_ids if shared[v] <= MAX_SHARED} for record_ids in ids]
    trusted: set[str] = set()
    linked = [False] * len(ids)
    changed = True
    while changed:
        changed = False
        for i in range(len(ids)):
            if linked[i]:
                continue
            if linkers[i] & trusted or any(appears_in(v, prompt) for v in linkers[i]):
                linked[i] = True
                trusted |= ids[i]
                changed = True
    return trusted
