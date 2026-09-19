"""Baseline: escalate every call. Once k runs out the gate denies instead."""

from __future__ import annotations

from approval_gate.decisions import Decision
from approval_gate.policies.base import CallContext, Policy


class FullEscalatePolicy(Policy):
    name = "full_escalate"

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        return Decision.ESCALATE, "baseline"
