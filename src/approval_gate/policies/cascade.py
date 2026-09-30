"""Our policy: allowlist, task memory, risk features and scorer, budget-aware VOI gate."""

from __future__ import annotations

from dataclasses import asdict

from approval_gate.decisions import Decision
from approval_gate.nodes.allowlist import Tier, classify
from approval_gate.nodes.envelope import EnvelopeExtractor
from approval_gate.nodes.memory import TaskMemory
from approval_gate.nodes.risk import extract
from approval_gate.nodes.scorers import FormulaScorer
from approval_gate.nodes.taint import DELEGATED, USER
from approval_gate.nodes.voi import TAU_LOW, assess_voi, escalation_bar
from approval_gate.policies.base import CallContext, Policy


class CascadePolicy(Policy):
    name = "cascade"

    def __init__(self, extractor: EnvelopeExtractor, scorer=None, scope_deny: bool = True) -> None:
        self.extractor = extractor
        self.scorer = scorer or FormulaScorer()
        self.scope_deny = scope_deny
        self.memory = TaskMemory()

    def start_task(self, task_id: str, user_prompt: str) -> None:
        self.memory.reset()
        self.envelope = self.extractor.extract(user_prompt)
        self._envelope_tokens_pending = True

    def bar(self, ctx: CallContext) -> float:
        # Denied attempts are not done work, so only executed writes count here.
        n_remaining = max(1, self.envelope.expected_writes - self.memory.executed)
        return escalation_bar(ctx.k_remaining, n_remaining)

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        if self._envelope_tokens_pending:
            ctx.features["tokens_in"] = {"envelope": self.envelope.tokens_in}
            ctx.features["tokens_out"] = {"envelope": self.envelope.tokens_out}
            self._envelope_tokens_pending = False

        info = classify(ctx.tool_call.function)
        if info.tier is Tier.READ_ONLY:
            return Decision.AUTO_APPROVE, "allowlist"
        if self.memory.denied_repeat(ctx.tool_call):
            return Decision.AUTO_DENY, "memory_repeat"

        f = extract(ctx, self.envelope, self.memory)
        R = self.scorer.score(f)
        v = assess_voi(R, info.harm, f.deviation, info.tier)
        bar = self.bar(ctx)
        ctx.features.update(
            taint_flags=f.taint, envelope_deviation=f.deviation, risk_R=R, voi=v.voi,
            tau_low=TAU_LOW, tau_esc=None if bar == float("inf") else bar,
            risk_features={**asdict(f), "scorer": self.scorer.name},
        )

        if (self.scope_deny and info.level == 4 and f.tool_unexpected == 1.0
                and f.origin not in (USER, DELEGATED)):
            return Decision.AUTO_DENY, "scope_deny"
        if v.voi >= bar:
            return Decision.ESCALATE, "voi"
        if v.auto_approve:
            return Decision.AUTO_APPROVE, "voi_approve"
        return Decision.AUTO_DENY, "voi_deny"

    def observe(self, ctx: CallContext, approved: bool, decided_by: str) -> None:
        if decided_by != "allowlist":
            self.memory.record(ctx.tool_call, approved, decided_by)
