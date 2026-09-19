"""Baseline: approve every call. Equivalent to running without a gate."""

from __future__ import annotations

from approval_gate.decisions import Decision
from approval_gate.policies.base import CallContext, Policy


class FullApprovePolicy(Policy):
    name = "full_approve"

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        return Decision.AUTO_APPROVE, "baseline"
