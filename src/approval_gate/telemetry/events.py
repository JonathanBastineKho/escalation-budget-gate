"""Append-only episode telemetry. Prompts never receive this metadata."""

from __future__ import annotations
import hashlib
import json
import warnings
from pathlib import Path


def serializable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=serializable)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=serializable) + "\n")
    temp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            warnings.warn(
                f"Incomplete final JSONL record in {path}; accounting may be a lower bound"
            )
    return rows


class EventLog:
    def __init__(self, directory, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}

    def emit(self, stream, **event):
        row = {**self.metadata, **event}
        with (self.directory / f"{stream}.jsonl").open("a") as f:
            f.write(canonical(row) + "\n")
        return row

    def log(self, record):
        from dataclasses import asdict

        self.emit("decisions", **asdict(record))
