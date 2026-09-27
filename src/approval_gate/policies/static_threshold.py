"""Baseline: the cascade with a fixed escalation bar that ignores the budget."""

from __future__ import annotations

from approval_gate.nodes.voi import STATIC_BAR
from approval_gate.policies.base import CallContext
from approval_gate.policies.cascade import CascadePolicy


class StaticThresholdPolicy(CascadePolicy):
    name = "static_threshold"

    def bar(self, ctx: CallContext) -> float:
        return STATIC_BAR
