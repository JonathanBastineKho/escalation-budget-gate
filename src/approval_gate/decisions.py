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
    config_hash: str | None = None
    comparison_id: str | None = None


@dataclass
class DecisionRecord:
    """One record per tool call, including auto-approved reads."""

    run_id: str
    task_id: str
    injection_task_id: str | None   # None on clean runs
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
    tool_call_id: str | None = None
    tool_tier: str = "unknown"
    args_hash: str = ""
    critical_args: dict = field(default_factory=dict)

    taint_flags: dict = field(default_factory=dict)
    risk_features: dict = field(default_factory=dict)
    envelope_deviation: float | None = None
    risk_R: float | None = None
    harm_H: float = 0.0
    voi: float | None = None
    tau_low: float | None = None
    tau_esc: float | None = None
    k_remaining_before: int | None = None
    decision: str = Decision.AUTO_APPROVE
    decided_by_node: str = "unset"
    reviewer_verdict: str | None = None
    reviewer_reason: str | None = None
    reviewer_error: str | None = None
    reviewer_usage_missing: int = 0
    speculative_used: bool = False
    rolled_back: bool = False
    executed: bool = True
    tool_error: str | None = None
    config_hash: str | None = None
    comparison_id: str | None = None
    judge_status: str | None = None
    judge_reason: str | None = None
    judge_diagnostics: dict = field(default_factory=dict)
    judge_evidence: dict = field(default_factory=dict)
    judge_allowed_labels: list[str] | None = None
    judge_thresholds: dict = field(default_factory=dict)
    judge_probabilities: dict = field(default_factory=dict)
    judge_input_hash: str | None = None
    judge_calls: int = 0
    judge_usage_missing: int = 0
    judge_latency_ms: float = 0.0
    preview_status: str | None = None
    preview_observation: str | None = None
    preview_error: str | None = None
    preview_calls: int = 0
    preview_latency_ms: float = 0.0
    preview_effects: dict | None = None

    # component -> tokens. "agent" is the model turn that proposed this call,
    # counted on the first call of that turn only, so sums stay correct.
    tokens_in: dict = field(default_factory=dict)
    tokens_out: dict = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class CaseRecord:
    """One record per (user task, injection task) case, written when the case
    ends. Holds totals the per-call records can't: the agent's final answer
    turn has no tool call, so its tokens only appear here."""

    run_id: str
    task_id: str
    injection_task_id: str | None
    suite: str
    policy: str
    k_budget: int | None
    seed: int
    attack_name: str | None
    attempts: int                  # pipeline.query() calls; >1 means AgentDojo retried
    gated_calls: int
    escalations: int               # calls the reviewer actually saw
    utility: bool | None           # None if the case raised
    security: bool | None          # None on clean runs, where AgentDojo's value is a placeholder
    error: str | None
    tokens_in: dict
    tokens_out: dict
    config_hash: str | None = None
    comparison_id: str | None = None
    model_agent: str | None = None
    model_gate: str | None = None
    model_reviewer: str | None = None
    judge_calls: int = 0
    judge_unavailable: int = 0
    judge_usage_missing: int = 0
    judge_latency_ms: float = 0.0
    preview_calls: int = 0
    preview_failures: int = 0
    preview_latency_ms: float = 0.0
    gate_latency_ms: float = 0.0
    reviewer_failures: int = 0
    reviewer_usage_missing: int = 0
    non_read_only_calls: int = 0
    non_read_only_executed: int = 0
