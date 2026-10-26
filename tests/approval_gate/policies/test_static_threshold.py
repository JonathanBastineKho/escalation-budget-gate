"""A fixed review threshold differs from the cascade's budget-aware threshold."""

from unittest.mock import Mock

import pytest

from approval_gate.decisions import Decision
from approval_gate.nodes.voi import STATIC_BAR
from approval_gate.policies.static_threshold import StaticThresholdPolicy


@pytest.mark.parametrize("budget", [0, 1, None])
def test_bar_is_independent_of_budget_and_completed_writes(make_context, make_cascade, budget):
    policy = make_cascade(policy_class=StaticThresholdPolicy)
    ctx = make_context(budget=budget)

    assert policy.bar(ctx) == STATIC_BAR
    policy.observe(ctx, True, "reviewer")
    policy.observe_execution(ctx.tool_call, {}, None)
    assert policy.bar(ctx) == STATIC_BAR


def test_static_policy_can_request_review_at_zero_budget(make_context, make_cascade):
    scorer = Mock(needs_state=False, last={})
    scorer.name = "fixture"
    scorer.score.return_value = 0.3
    static = make_cascade(policy_class=StaticThresholdPolicy, scorer=scorer)
    adaptive = make_cascade(scorer=scorer)

    assert static.decide(make_context(budget=0)) == (Decision.ESCALATE, "voi")
    assert adaptive.decide(make_context(budget=0)) == (Decision.AUTO_APPROVE, "voi_approve")
