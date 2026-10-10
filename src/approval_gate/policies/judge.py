"""Matched raw-context judge policies, differing only in one-call preview evidence."""

from __future__ import annotations

from approval_gate.decisions import Decision
from approval_gate.judging import Judgment, canonical, fingerprint, judge_state
from approval_gate.policies.base import CallContext, Policy


class DirectJudgePolicy(Policy):
    name = "direct_judge"
    requires_effects = False

    def __init__(
        self, judge, schemas: dict, threshold: float = 0.8, max_input_chars: int = 48000,
        *, approve_threshold: float | None = None, deny_threshold: float | None = None,
        verdict_format: str = "scores",
    ) -> None:
        self.approve_threshold = threshold if approve_threshold is None else approve_threshold
        self.deny_threshold = threshold if deny_threshold is None else deny_threshold
        if (not all(0.5 < t <= 1 for t in (threshold, self.approve_threshold, self.deny_threshold))
                or max_input_chars < 1 or verdict_format not in {"scores", "evidence"}):
            raise ValueError("Require 0.5 < threshold <= 1 and a positive input limit")
        self.judge, self.schemas = judge, schemas
        self.threshold, self.max_input_chars = threshold, max_input_chars
        self.verdict_format = verdict_format

    def state(self, ctx):
        return judge_state(ctx, self.schemas[ctx.tool_call.function],
                           include_evidence=self.verdict_format == "evidence",
                           include_effects=self.requires_effects)

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        if self.requires_preview and (ctx.preview is None or ctx.preview.status != "ok"):
            result = Judgment(
                status="preview_unavailable",
                reason="Complete preview unavailable",
                calls=0,
                usage_known=True,
            )
        elif self.requires_effects and (not ctx.preview.effects
                                       or ctx.preview.effects.get("status") != "complete"):
            result = Judgment(status="effects_unavailable", reason="Audited effects unavailable",
                              calls=0, usage_known=True)
        elif ctx.tool_call.function not in self.schemas:
            result = Judgment(
                status="missing_schema", reason="Tool schema unavailable", calls=0, usage_known=True
            )
        else:
            try:
                state = self.state(ctx)
                ctx.features["judge_input_hash"] = fingerprint(state)
                oversized = len(canonical(state)) > self.max_input_chars
            except (TypeError, ValueError):
                result = Judgment(
                    status="invalid_input",
                    reason="Cannot serialize complete evidence",
                    calls=0,
                    usage_known=True,
                )
            else:
                if oversized:
                    result = Judgment(
                        status="input_too_large",
                        reason="Input exceeds limit; not truncated",
                        calls=0,
                        usage_known=True,
                    )
                else:
                    result = self.judge.judge(state)
        if (result.status == "ok" and self.verdict_format == "evidence"
                and (not result.evidence or result.allowed_labels is None)):
            result.status = "invalid_evidence"
            result.reason = "Required authorization evidence is missing"
        ctx.features.update(
            judge_status=result.status,
            judge_reason=result.reason,
            judge_diagnostics=result.diagnostics,
            judge_evidence=result.evidence,
            judge_allowed_labels=result.allowed_labels,
            judge_thresholds={"approve": self.approve_threshold, "deny": self.deny_threshold},
            judge_probabilities=result.probabilities,
            judge_calls=result.calls,
            judge_latency_ms=result.latency_ms,
            judge_usage_missing=int(result.calls > 0 and not result.usage_known),
            tokens_in={"judge": result.tokens_in},
            tokens_out={"judge": result.tokens_out},
        )
        if result.status == "ok":
            for label, action, cutoff in (
                ("APPROVE", Decision.AUTO_APPROVE, self.approve_threshold),
                ("DENY", Decision.AUTO_DENY, self.deny_threshold),
            ):
                eligible = result.allowed_labels is None or label in result.allowed_labels
                if eligible and result.probabilities.get(label, 0.0) >= cutoff:
                    return action, f"{self.name}_{label.lower()}"
        # Gate owns the budget. At zero budget it resolves this request to denial.
        return Decision.ESCALATE, f"{self.name}_uncertain"


class SandboxJudgePolicy(DirectJudgePolicy):
    name = "sandbox_judge"
    requires_preview = True


class SandboxEffectsJudgePolicy(SandboxJudgePolicy):
    name = "sandbox_effects_judge"
    requires_effects = True
