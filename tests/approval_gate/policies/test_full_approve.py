"""Full approval is independent of tool classification and review budget."""

import pytest

from approval_gate.decisions import Decision
from approval_gate.policies.full_approve import FullApprovePolicy


@pytest.mark.parametrize("tool", ["read_file", "send_money", "unknown_fixture_tool"])
@pytest.mark.parametrize("budget", [0, 1, None])
def test_approves_without_spending_budget_or_adding_features(make_context, tool, budget):
    ctx = make_context(tool, budget=budget)

    assert FullApprovePolicy().decide(ctx) == (Decision.AUTO_APPROVE, "baseline")
    assert ctx.k_remaining == budget
    assert ctx.features == {}
