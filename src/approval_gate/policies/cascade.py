"""Our policy: allowlist, task memory, risk features and scorer, budget-aware VOI gate."""

from __future__ import annotations

from dataclasses import asdict

from approval_gate.decisions import Decision
from approval_gate.nodes.allowlist import Tier, ToolInfo, classify
from approval_gate.nodes.binding import reachable, records
from approval_gate.nodes.envelope import EnvelopeExtractor
from approval_gate.nodes.memory import TaskMemory
from approval_gate.nodes.risk import RiskFeatures, extract
from approval_gate.nodes.scorers import FormulaScorer, decision_state
from approval_gate.nodes.taint import DELEGATED, USER
from approval_gate.nodes.voi import TAU_LOW, VoiResult, assess_voi, escalation_bar
from approval_gate.policies.base import CallContext, Policy


class CascadePolicy(Policy):
    name = "cascade"

    def __init__(
        self,
        extractor: EnvelopeExtractor,
        scorer=None,
        scope_deny: bool = True,
        feature_extractor=None,
    ) -> None:
        self.extractor = extractor
        self.scorer = scorer or FormulaScorer()
        self.scope_deny = scope_deny
        self.feature_extractor = feature_extractor  # None means the code features
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

        features, value, bar = self._assess_call(ctx, info)

        if (
            self.scope_deny
            and info.level == 4
            and features.tool_unexpected == 1.0
            and features.origin not in (USER, DELEGATED)
        ):
            return Decision.AUTO_DENY, "scope_deny"
        if value.voi >= bar:
            return Decision.ESCALATE, "voi"
        if value.auto_approve:
            return Decision.AUTO_APPROVE, "voi_approve"
        return Decision.AUTO_DENY, "voi_deny"

    def _extract_features(self, ctx: CallContext) -> tuple[RiskFeatures, dict]:
        """Prepare risk features and record any feature-model usage."""
        trusted = reachable(ctx.user_prompt, self.records)
        feature_info = {"feature_source": "code"}
        if self.feature_extractor is None:
            features = extract(ctx, self.envelope, self.memory, trusted)
        else:
            features = self.feature_extractor.extract(ctx, self.envelope, self.memory, trusted)
            fx = self.feature_extractor.last
            feature_info = {
                "feature_source": self.feature_extractor.name,
                "feature_status": fx.get("status"),
                "feature_reason": fx.get("reason"),
                "feature_latency_ms": fx.get("latency_ms"),
            }
            if fx.get("tokens_in") or fx.get("tokens_out"):
                ctx.features.setdefault("tokens_in", {})["features"] = fx["tokens_in"]
                ctx.features.setdefault("tokens_out", {})["features"] = fx["tokens_out"]
        return features, feature_info

    def _assess_call(
        self, ctx: CallContext, info: ToolInfo
    ) -> tuple[RiskFeatures, VoiResult, float]:
        """Score a non-read call and record costs and diagnostics before deciding."""
        features, feature_info = self._extract_features(ctx)
        state = None
        if self.scorer.needs_state:
            state = decision_state(
                ctx, self.envelope, self.memory, features, self.scorer.include_checks
            )
        risk = self.scorer.score(features, state)
        scored = self.scorer.last
        if scored.get("tokens_in") or scored.get("tokens_out"):
            ctx.features.setdefault("tokens_in", {})["risk_scorer"] = scored["tokens_in"]
            ctx.features.setdefault("tokens_out", {})["risk_scorer"] = scored["tokens_out"]
        value = assess_voi(risk, info.harm, features.deviation, info.tier)
        bar = self.bar(ctx)
        ctx.features.update(
            taint_flags=features.taint,
            envelope_deviation=features.deviation,
            risk_R=risk,
            voi=value.voi,
            tau_low=TAU_LOW,
            tau_esc=None if bar == float("inf") else bar,
            risk_features={
                **asdict(features),
                **feature_info,
                "scorer": self.scorer.name,
                "scorer_status": scored.get("status", "ok"),
                "scorer_latency_ms": scored.get("latency_ms"),
            },
        )

        return features, value, bar

    def observe(self, ctx: CallContext, approved: bool, decided_by: str) -> None:
        if decided_by != "allowlist":
            self.memory.record(ctx.tool_call, approved, decided_by)

    def observe_execution(self, tool_call, result, error) -> None:
        if error:
            self.memory.mark_failed(tool_call)
        else:
            self.records += records(result)
