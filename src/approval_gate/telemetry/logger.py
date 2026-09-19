"""JSONL decision log, one record per tool call."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from approval_gate.decisions import DecisionRecord


class DecisionLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self.decisions: Counter[str] = Counter()
        self.escalations_used = 0
        self.denied = 0
        self.tokens_in: Counter[str] = Counter()
        self.tokens_out: Counter[str] = Counter()

    def log(self, record: DecisionRecord) -> None:
        self._fh.write(json.dumps(asdict(record)) + "\n")
        self._fh.flush()   # keep records if the run crashes
        self.decisions[record.decision] += 1
        if record.decided_by_node == "reviewer":
            self.escalations_used += 1
        if not record.executed:
            self.denied += 1
        self.tokens_in.update(record.tokens_in)
        self.tokens_out.update(record.tokens_out)

    def summary(self) -> str:
        total = sum(self.decisions.values())
        parts = [f"{d}={n}" for d, n in sorted(self.decisions.items())]
        tokens = "  ".join(
            f"{c}_tokens={self.tokens_in[c]}in/{self.tokens_out[c]}out" for c in sorted(self.tokens_in)
        )
        return (
            f"{total} calls gated  |  " + "  ".join(parts)
            + f"  |  escalations={self.escalations_used}  denied={self.denied}"
            + (f"  |  {tokens}" if tokens else "")
        )

    def close(self) -> None:
        self._fh.close()
