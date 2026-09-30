"""Node 1: tool tiers and harm, read from config/tools.yaml.

Tier: level 0 is read_only, 1 and 2 are write, 3 and 4 are
irreversible_external. Unknown tools get level 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from approval_gate.settings import CONFIG_DIR

TOOLS_FILE = CONFIG_DIR / "tools.yaml"


class Tier(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    IRREVERSIBLE_EXTERNAL = "irreversible_external"


HARM_BY_LEVEL = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.8, 4: 1.0}
UNKNOWN_LEVEL = 4
ROLES = {"destination", "target", "value"}


def _load(path: Path) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    levels, roles = {}, {}
    for name, entry in yaml.safe_load(path.read_text()).items():
        if entry.get("level") not in HARM_BY_LEVEL:
            raise ValueError(f"{path.name}: {name} needs a level from 0 to 4")
        bad = set(entry.get("roles", {}).values()) - ROLES
        if bad:
            raise ValueError(f"{path.name}: {name} has unknown roles {sorted(bad)}")
        levels[name] = entry["level"]
        if "roles" in entry:
            roles[name] = dict(entry["roles"])
    return levels, roles


TOOL_LEVELS, TOOL_ROLES = _load(TOOLS_FILE)


@dataclass(frozen=True)
class ToolInfo:
    level: int
    harm: float
    tier: Tier
    known: bool


def _tier(level: int) -> Tier:
    if level == 0:
        return Tier.READ_ONLY
    if level <= 2:
        return Tier.WRITE
    return Tier.IRREVERSIBLE_EXTERNAL


def classify(tool_name: str) -> ToolInfo:
    level = TOOL_LEVELS.get(tool_name, UNKNOWN_LEVEL)
    return ToolInfo(level=level, harm=HARM_BY_LEVEL[level], tier=_tier(level),
                    known=tool_name in TOOL_LEVELS)


def check_coverage(tool_names) -> None:
    """Fail if a suite has tools missing from the table."""
    missing = sorted(set(tool_names) - set(TOOL_LEVELS))
    if missing:
        raise SystemExit(f"config/tools.yaml has no entry for: {missing}")
