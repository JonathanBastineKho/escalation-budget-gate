"""Loads the YAML files in config/."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


POLICY = load("policy.yaml")
CONFIG_HASH = hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted(CONFIG_DIR.glob("*.yaml")))
).hexdigest()[:12]
