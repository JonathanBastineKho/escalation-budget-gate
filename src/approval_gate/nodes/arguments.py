"""Argument roles and value matching shared by Nodes 2 and 3."""

from __future__ import annotations

import re

from approval_gate.nodes.allowlist import TOOL_ROLES, classify

DESTINATION = "destination"
TARGET = "target"
VALUE = "value"

_IBAN = re.compile(r"[a-z]{2}\d{2}[a-z0-9]{10,30}")
_NUMBER = re.compile(r"(?<![\d.\-:/])\d+(?:\.\d+)?(?![\d\-:/]|\.\d)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_KINDS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "iban": re.compile(r"\b[a-z]{2}\d{2}[a-z0-9]{10,30}\b"),
    "url": re.compile(r"(?:https?://|www\.)\S+"),
}


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


def arg_roles(tool: str, args: dict) -> dict[str, str]:
    """Unlabelled write tools treat every argument as a destination (fail closed)."""
    if tool in TOOL_ROLES:
        return {k: r for k, r in TOOL_ROLES[tool].items() if k in args}
    if classify(tool).level == 0:
        return {}
    return {k: DESTINATION for k in args}


def critical_args(tool: str, args: dict) -> dict:
    return {k: args[k] for k in arg_roles(tool, args)}


def args_with_role(tool: str, args: dict, role: str) -> dict:
    return {k: args[k] for k, r in arg_roles(tool, args).items() if r == role}


def destination_args(tool: str, args: dict) -> dict:
    """Targets are left out: ids are normally looked up from tool outputs, so
    where they first appeared says nothing about an injection."""
    return args_with_role(tool, args, DESTINATION)


def kind(value) -> str | None:
    """email, iban or url, for values that are wholly one of those."""
    text = normalize(value)
    for name, pattern in _KINDS.items():
        if pattern.fullmatch(text):
            return name
    return None


def kinds_in(text: str) -> set[str]:
    return {name for name, pattern in _KINDS.items() if pattern.search(normalize_text(text))}


def _whole(needle: str, text: str) -> bool:
    pattern = rf"(?<![\w@.\-]){re.escape(needle)}(?![\w@\-]|\.\w)"
    return re.search(pattern, text) is not None


def appears_in(value, text: str) -> bool:
    """Whether value occurs in text as a whole value, not inside a longer one.
    text must already be normalize_text()ed."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        numbers = _NUMBER.findall(_THOUSANDS.sub("", text))
        return any(abs(float(tok) - float(value)) < 0.005 for tok in numbers)
    needle = normalize(value)
    if len(needle) < 4:
        return False
    compact = needle.replace(" ", "")
    return _whole(needle, text) or (compact != needle and _whole(compact, text.replace(" ", "")))
