"""Decision types and the per-call log record."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum


class Decision(StrEnum):
    AUTO_APPROVE = "AUTO_APPROVE"
    AUTO_DENY = "AUTO_DENY"
    ESCALATE = "ESCALATE"


def hash_args(args: dict) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class RunMeta:
    """Constant for a whole run, copied into every record."""

    run_id: str
    suite: str
    policy: str
    k_budget: int | None          # None means unbounded
    seed: int
    model_agent: str
    model_gate: str
    attack_name: str | None = None
    model_reviewer: str | None = None


@dataclass
class DecisionRecord:
    """One record per tool call, including auto-approved reads."""

    run_id: str
    task_id: str
    suite: str
    policy: str
    k_budget: int | None
    seed: int
    model_agent: str
    model_gate: str
    model_reviewer: str | None
    attack_name: str | None
    step_index: int

    tool_name: str
    tool_tier: str = "unknown"
    args_hash: str = ""
    critical_args: dict = field(default_factory=dict)

    taint_flags: dict = field(default_factory=dict)
    envelope_deviation: float = 0.0
    risk_R: float = 0.0
    harm_H: float = 0.0
    voi: float = 0.0

    tau_low: float = 0.0
    tau_deny: float = 1.0
    tau_esc: float = 0.0
    k_remaining_before: int | None = None
    decision: str = Decision.AUTO_APPROVE
    decided_by_node: str = "unset"
    reviewer_verdict: str | None = None
    reviewer_reason: str | None = None
    speculative_used: bool = False
    rolled_back: bool = False
    executed: bool = True           # final outcome after an escalation resolves

    tokens_in: dict = field(default_factory=dict)     # component -> tokens
    tokens_out: dict = field(default_factory=dict)
    latency_ms: float = 0.0
