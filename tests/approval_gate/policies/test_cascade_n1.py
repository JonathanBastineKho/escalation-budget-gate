"""The allowlist-only ablation approves reads and escalates other tools."""

import pytest

from approval_gate.decisions import Decision
from approval_gate.policies.cascade_n1 import CascadeN1Policy


@pytest.mark.parametrize("budget", [0, 2, None])
@pytest.mark.parametrize(
    ("tool", "action", "node"),
    [
        ("read_file", Decision.AUTO_APPROVE, "allowlist"),
        ("update_scheduled_transaction", Decision.ESCALATE, "escalate_default"),
        ("send_money", Decision.ESCALATE, "escalate_default"),
        ("unknown_fixture_tool", Decision.ESCALATE, "escalate_default"),
    ],
)
def test_only_known_read_only_tools_bypass_review(make_context, budget, tool, action, node):
    ctx = make_context(tool, budget=budget)

    assert CascadeN1Policy().decide(ctx) == (action, node)
    assert ctx.k_remaining == budget
