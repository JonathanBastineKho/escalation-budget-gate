"""Our policy: allowlist, scope cache, risk features, budget-aware VOI gate."""

from __future__ import annotations

from approval_gate.decisions import Decision
from approval_gate.nodes.allowlist import Tier, classify
from approval_gate.nodes.envelope import EnvelopeExtractor
from approval_gate.nodes.risk import assess
from approval_gate.nodes.scope_cache import ScopeCache
from approval_gate.nodes.voi import TAU_DENY, TAU_LOW, assess_voi, escalation_bar
from approval_gate.policies.base import CallContext, Policy

_NOT_JUDGMENTS = {"allowlist", "cache_pos", "cache_neg", "budget_exhausted"}


class CascadePolicy(Policy):
    name = "cascade"

    def __init__(self, extractor: EnvelopeExtractor) -> None:
        self.extractor = extractor
        self.cache = ScopeCache()

    def start_task(self, task_id: str, user_prompt: str) -> None:
        self.cache.reset()
        self.writes_done = 0
        self.envelope = self.extractor.extract(user_prompt)
        self._envelope_tokens_pending = True

    def bar(self, ctx: CallContext) -> float:
        n_remaining = max(1, self.envelope.expected_writes - self.writes_done)
        return escalation_bar(ctx.k_remaining, n_remaining)

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        if self._envelope_tokens_pending:
            ctx.features["tokens_in"] = {"envelope": self.envelope.tokens_in}
            ctx.features["tokens_out"] = {"envelope": self.envelope.tokens_out}
            self._envelope_tokens_pending = False

        info = classify(ctx.tool_call.function)
        if info.tier is Tier.READ_ONLY:
            return Decision.AUTO_APPROVE, "allowlist"

        cached = self.cache.lookup(ctx.tool_call)
        if cached is not None:
            return cached

        risk = assess(ctx, self.envelope, self.writes_done)
        v = assess_voi(risk.R, info.harm, risk.deviation, info.tier)
        bar = self.bar(ctx)
        self.writes_done += 1
        ctx.features.update(
            taint_flags=risk.taint, envelope_deviation=risk.deviation, risk_R=risk.R,
            voi=v.voi, tau_low=TAU_LOW, tau_deny=TAU_DENY,
            tau_esc=None if bar == float("inf") else bar,
        )

        if risk.taint_critical and v.loss_approve >= TAU_DENY:
            return Decision.AUTO_DENY, "voi_obvious"
        if v.voi >= bar:
            return Decision.ESCALATE, "voi"
        if v.auto_approve:
            return Decision.AUTO_APPROVE, "voi_approve"
        return Decision.AUTO_DENY, "voi_deny"

    def observe(self, ctx: CallContext, approved: bool, decided_by: str) -> None:
        if decided_by not in _NOT_JUDGMENTS:
            self.cache.record(ctx.tool_call, approved)
