"""Ablation: Node 1 only. Read-only calls are approved, everything else escalated."""

from __future__ import annotations

from approval_gate.decisions import Decision
from approval_gate.nodes.allowlist import Tier, classify
from approval_gate.policies.base import CallContext, Policy


class CascadeN1Policy(Policy):
    name = "cascade_n1"

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        if classify(ctx.tool_call.function).tier is Tier.READ_ONLY:
            return Decision.AUTO_APPROVE, "allowlist"
        return Decision.ESCALATE, "escalate_default"
