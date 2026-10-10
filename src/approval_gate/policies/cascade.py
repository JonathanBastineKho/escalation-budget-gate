"""Our policy: allowlist, task memory, risk features and scorer, budget-aware VOI gate."""

from __future__ import annotations

from dataclasses import asdict

from approval_gate.decisions import Decision
from approval_gate.nodes.allowlist import Tier, classify
from approval_gate.nodes.binding import reachable, records
from approval_gate.nodes.envelope import EnvelopeExtractor
from approval_gate.nodes.memory import TaskMemory
from approval_gate.nodes.risk import extract
from approval_gate.nodes.scorers import FormulaScorer, decision_state
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
        self.records: list[dict] = []
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

        f = extract(ctx, self.envelope, self.memory, reachable(ctx.user_prompt, self.records))
        state = None
        if self.scorer.needs_state:
            state = decision_state(ctx, self.envelope, self.memory, f, self.scorer.include_checks)
        R = self.scorer.score(f, state)
        scored = self.scorer.last
        if scored.get("tokens_in") or scored.get("tokens_out"):
            ctx.features.setdefault("tokens_in", {})["risk_scorer"] = scored["tokens_in"]
            ctx.features.setdefault("tokens_out", {})["risk_scorer"] = scored["tokens_out"]
        v = assess_voi(R, info.harm, f.deviation, info.tier)
        bar = self.bar(ctx)
        ctx.features.update(
            taint_flags=f.taint, envelope_deviation=f.deviation, risk_R=R, voi=v.voi,
            tau_low=TAU_LOW, tau_esc=None if bar == float("inf") else bar,
            risk_features={**asdict(f), "scorer": self.scorer.name,
                           "scorer_status": scored.get("status", "ok"),
                           "scorer_latency_ms": scored.get("latency_ms")},
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

    def observe_execution(self, tool_call, result, error) -> None:
        if error:
            self.memory.mark_failed(tool_call)
        else:
            self.records += records(result)
