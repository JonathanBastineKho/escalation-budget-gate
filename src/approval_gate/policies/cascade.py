"""Our policy, built one node at a time.

Implemented: Node 1 (read-only calls are approved).
Everything else is escalated until Nodes 2 to 4 exist.
"""

from __future__ import annotations

from approval_gate.decisions import Decision
from approval_gate.nodes.allowlist import Tier, classify
from approval_gate.policies.base import CallContext, Policy


class CascadePolicy(Policy):
    name = "cascade"

    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        # Reads cause no harm on their own; the risk is a later write that
        # uses what was read.
        if classify(ctx.tool_call.function).tier is Tier.READ_ONLY:
            return Decision.AUTO_APPROVE, "allowlist"

        return Decision.ESCALATE, "escalate_default"
