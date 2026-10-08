"""JSONL logs: one record per tool call, and one per case."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from approval_gate.decisions import CaseRecord, DecisionRecord


class DecisionLogger:
    def __init__(self, path: Path, cases_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self._cases_fh = cases_path.open("a", encoding="utf-8")
        self.decisions: Counter[str] = Counter()
        self.escalations_used = 0
        self.denied = 0
        self.judge_unavailable = 0
        self.judge_usage_missing = 0
        # From case records: they include the agent's final turn, which
        # per-call records miss.
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

    def log_case(self, record: CaseRecord) -> None:
        self._cases_fh.write(json.dumps(asdict(record)) + "\n")
        self._cases_fh.flush()
        self.tokens_in.update(record.tokens_in)
        self.tokens_out.update(record.tokens_out)
        self.judge_unavailable += record.judge_unavailable
        self.judge_usage_missing += record.judge_usage_missing

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
        self._cases_fh.close()
