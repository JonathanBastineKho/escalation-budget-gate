"""Argument helpers shared by the scope cache and taint tracking."""

from __future__ import annotations

import re

DESTINATION_NAMES = {
    "recipient", "recipients", "to", "email", "emails", "iban", "url", "channel",
    "user", "participants", "hotel", "restaurant", "company",
}
TARGET_NAMES = {"id", "file_id", "file_path", "filename"}
CRITICAL_NAMES = DESTINATION_NAMES | TARGET_NAMES | {"amount", "password"}

_IBAN = re.compile(r"[a-z]{2}\d{2}[a-z0-9]{10,30}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def flatten(value) -> list:
    if isinstance(value, (list, tuple, set)):
        return [x for v in value for x in flatten(v)]
    return [value]


def normalize(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = re.sub(r"^\.?/+", "", text)
    compact = text.replace(" ", "")
    return compact if _IBAN.fullmatch(compact) else text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower())


def critical_args(args: dict) -> dict:
    return {k: v for k, v in args.items() if k in CRITICAL_NAMES}


def destination_args(args: dict) -> dict:
    dest = {k: v for k, v in args.items() if k in DESTINATION_NAMES}
    return dest or {k: v for k, v in args.items() if k in TARGET_NAMES}


def appears_in(value, text: str) -> bool:
    """Whether value occurs in text, which must already be normalize_text()ed."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return any(abs(float(tok) - float(value)) < 0.005 for tok in _NUMBER.findall(text))
    needle = normalize(value)
    if len(needle) < 4:
        return False
    return needle in text or needle.replace(" ", "") in text.replace(" ", "")
