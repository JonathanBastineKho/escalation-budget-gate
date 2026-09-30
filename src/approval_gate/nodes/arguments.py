"""Argument roles and value matching shared by Nodes 2 and 3."""

from __future__ import annotations

import re

from approval_gate.nodes.allowlist import TOOL_ROLES, classify

DESTINATION = "destination"
TARGET = "target"

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


def arg_roles(tool: str, args: dict) -> dict[str, str]:
    """Unlabelled write tools treat every argument as a destination (fail closed)."""
    if tool in TOOL_ROLES:
        return {k: r for k, r in TOOL_ROLES[tool].items() if k in args}
    if classify(tool).level == 0:
        return {}
    return {k: DESTINATION for k in args}


def critical_args(tool: str, args: dict) -> dict:
    return {k: args[k] for k in arg_roles(tool, args)}


def destination_args(tool: str, args: dict) -> dict:
    """A tool's destinations, or its targets when it has none."""
    roles = arg_roles(tool, args)
    dest = {k: args[k] for k, r in roles.items() if r == DESTINATION}
    return dest or {k: args[k] for k, r in roles.items() if r == TARGET}


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
