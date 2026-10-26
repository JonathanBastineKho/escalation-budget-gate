"""The policy requests review; the gate decides whether budget permits it."""

import pytest

from approval_gate.decisions import Decision
from approval_gate.policies.full_escalate import FullEscalatePolicy


@pytest.mark.parametrize("tool", ["read_file", "send_money", "unknown_fixture_tool"])
@pytest.mark.parametrize("budget", [0, 1, None])
def test_requests_review_even_at_zero_budget(make_context, tool, budget):
    ctx = make_context(tool, budget=budget)

    assert FullEscalatePolicy().decide(ctx) == (Decision.ESCALATE, "baseline")
    assert ctx.k_remaining == budget
    assert ctx.features == {}
