"""Cascade orchestration, task-local memory, and component accounting."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from approval_gate.decisions import Decision
from approval_gate.nodes.risk import RiskFeatures
from approval_gate.nodes.taint import DELEGATED, TOOL_OUTPUT, USER


def fixed_scorer(risk):
    scorer = Mock(needs_state=False, last={})
    scorer.name = "fixture"
    scorer.score.return_value = risk
    return scorer


def fixed_features(**overrides):
    features = RiskFeatures(
        taint={"recipient": USER},
        origin=USER,
        tool_unexpected=0.0,
        destination_unnamed=0.0,
        over_expected_writes=0.0,
        value_from_tool_output=0.0,
        target_unbound=0.0,
        dest_denied_before=0.0,
        dest_approved_before=0.0,
        denials_so_far=0,
    )
    extractor = Mock(last={})
    extractor.name = "fixture-features"
    extractor.extract.return_value = replace(features, **overrides)
    return extractor


def test_reads_bypass_feature_extraction_and_scoring(make_context, make_cascade):
    scorer, features = fixed_scorer(0.95), fixed_features()
    policy = make_cascade(scorer=scorer, feature_extractor=features)

    assert policy.decide(make_context("read_file", args={"file_path": "bill.txt"})) == (
        Decision.AUTO_APPROVE,
        "allowlist",
    )
    features.extract.assert_not_called()
    scorer.score.assert_not_called()


def test_envelope_cost_is_charged_once_and_resets_on_next_task(make_context, make_cascade):
    policy = make_cascade()
    first, second = make_context("read_file"), make_context("read_file")
    policy.decide(first)
    policy.decide(second)

    assert first.features["tokens_in"] == {"envelope": 17}
    assert first.features["tokens_out"] == {"envelope": 5}
    assert "tokens_in" not in second.features
    assert "tokens_out" not in second.features
    policy.extractor.extract.assert_called_once()

    policy.start_task("next-task", "Read another invoice.")
    third = make_context("read_file")
    policy.decide(third)
    assert third.features["tokens_in"] == {"envelope": 17}
    assert policy.extractor.extract.call_count == 2


def test_reviewer_denied_repeat_short_circuits_but_changed_arguments_do_not(
    make_context, make_cascade
):
    scorer = fixed_scorer(0.2)
    policy = make_cascade(scorer=scorer)
    original = make_context()
    policy.observe(original, False, "reviewer")

    assert policy.decide(make_context()) == (Decision.AUTO_DENY, "memory_repeat")
    scorer.score.assert_not_called()

    policy.decide(make_context(args={"recipient": "Alice", "amount": 26}))
    scorer.score.assert_called_once()


@pytest.mark.parametrize(
    ("approved", "node"),
    [(False, "voi_deny"), (False, "budget_exhausted"), (True, "reviewer")],
)
def test_only_reviewer_denials_short_circuit_future_decisions(
    make_context, make_cascade, approved, node
):
    scorer = fixed_scorer(0.2)
    policy = make_cascade(scorer=scorer)
    policy.observe(make_context(), approved, node)

    _, deciding_node = policy.decide(make_context())

    assert deciding_node != "memory_repeat"
    scorer.score.assert_called_once()


def test_start_task_clears_previous_denials_and_binding_records(make_context, make_cascade):
    policy = make_cascade()
    ctx = make_context()
    policy.observe(ctx, False, "reviewer")
    policy.observe_execution(ctx.tool_call, {"id": "old-record"}, None)

    policy.start_task("new-task", ctx.user_prompt)

    assert policy.memory.entries == []
    assert policy.records == []
    assert policy.decide(make_context())[1] != "memory_repeat"


def test_allowlisted_reads_do_not_use_write_memory(make_context, make_cascade):
    policy = make_cascade()
    policy.observe(make_context("read_file"), True, "allowlist")

    assert policy.memory.proposed == policy.memory.executed == 0


def test_successful_execution_keeps_nested_records(make_context, make_cascade):
    policy = make_cascade()
    ctx = make_context()
    result = {"transactions": [{"id": "payment-1", "recipient": "Alice"}]}
    policy.observe(ctx, True, "reviewer")
    policy.observe_execution(ctx.tool_call, result, None)

    assert policy.memory.executed == 1
    assert policy.records == [result, result["transactions"][0]]


def test_failed_execution_does_not_count_as_completed_write_or_trusted_output(
    make_context, make_cascade
):
    policy = make_cascade()
    ctx = make_context(budget=1)
    initial_bar = policy.bar(ctx)
    policy.observe(ctx, True, "reviewer")
    assert policy.bar(ctx) < initial_bar

    policy.observe_execution(ctx.tool_call, {"id": "untrusted-partial-result"}, "tool failed")

    assert policy.memory.proposed == 1
    assert policy.memory.executed == 0
    assert policy.records == []
    assert policy.bar(ctx) == initial_bar


@pytest.mark.parametrize(
    ("budget", "expected"), [(0, float("inf")), (None, 0.05), (1, 0.375), (4, 0.05)]
)
def test_review_bar_tracks_budget_share(make_context, make_cascade, budget, expected):
    policy = make_cascade(expected_writes=4)

    assert policy.bar(make_context(budget=budget)) == expected


def test_denied_proposals_do_not_reduce_remaining_work(make_context, make_cascade):
    policy = make_cascade(expected_writes=4)
    ctx = make_context(budget=1)
    before = policy.bar(ctx)
    for _ in range(5):
        policy.observe(ctx, False, "budget_exhausted")

    assert policy.bar(ctx) == before


@pytest.mark.parametrize(
    ("risk", "budget", "action", "node"),
    [
        (0.05, 1, Decision.AUTO_APPROVE, "voi_approve"),
        (0.95, 1, Decision.AUTO_DENY, "voi_deny"),
        (0.375, 1, Decision.ESCALATE, "voi"),  # Exact review threshold is inclusive.
        (0.3749, 1, Decision.AUTO_APPROVE, "voi_approve"),
        (0.2, None, Decision.ESCALATE, "voi"),
        (0.2, 0, Decision.AUTO_APPROVE, "voi_approve"),
        (0.95, 0, Decision.AUTO_DENY, "voi_deny"),
    ],
)
def test_voi_decisions_and_budget_boundaries(
    make_context, make_cascade, risk, budget, action, node
):
    policy = make_cascade(scorer=fixed_scorer(risk))
    ctx = make_context(budget=budget)

    assert policy.decide(ctx) == (action, node)
    assert ctx.k_remaining == budget
    assert ctx.features["risk_features"]["feature_source"] == "code"
    if budget == 0:
        assert ctx.features["tau_esc"] is None  # JSON-safe representation of infinity.


@pytest.mark.parametrize("origin", [USER, DELEGATED, TOOL_OUTPUT])
@pytest.mark.parametrize("scope_deny", [True, False])
def test_scope_veto_requires_untrusted_unexpected_high_harm_call(
    make_context, make_cascade, origin, scope_deny
):
    policy = make_cascade(
        scorer=fixed_scorer(0.2),
        feature_extractor=fixed_features(origin=origin, tool_unexpected=1.0),
        scope_deny=scope_deny,
    )

    outcome = policy.decide(make_context(budget=None))

    if scope_deny and origin == TOOL_OUTPUT:
        assert outcome == (Decision.AUTO_DENY, "scope_deny")
    else:
        assert outcome == (Decision.ESCALATE, "voi")


def test_scope_veto_does_not_apply_to_lower_harm_tools(make_context, make_cascade):
    policy = make_cascade(
        scorer=fixed_scorer(0.2),
        feature_extractor=fixed_features(origin=TOOL_OUTPUT, tool_unexpected=1.0),
    )

    assert policy.decide(make_context("delete_file", args={"file_id": "f1"}, budget=None)) == (
        Decision.ESCALATE,
        "voi",
    )


def test_optional_extractors_preserve_all_component_costs(make_context, make_cascade):
    scorer, features = fixed_scorer(0.2), fixed_features()
    scorer.last = {"tokens_in": 31, "tokens_out": 7, "status": "ok", "latency_ms": 3.0}
    features.last = {
        "tokens_in": 23,
        "tokens_out": 11,
        "status": "ok",
        "reason": "fixture",
        "latency_ms": 2.0,
    }
    policy = make_cascade(scorer=scorer, feature_extractor=features)
    ctx = make_context()

    policy.decide(ctx)

    assert ctx.features["tokens_in"] == {"envelope": 17, "features": 23, "risk_scorer": 31}
    assert ctx.features["tokens_out"] == {"envelope": 5, "features": 11, "risk_scorer": 7}
    assert ctx.features["risk_features"]["feature_source"] == "fixture-features"
    assert ctx.features["risk_features"]["scorer_status"] == "ok"


@pytest.mark.parametrize("include_checks", [True, False])
def test_stateful_scorer_receives_structured_context_without_review_budget(
    make_context, make_cascade, include_checks
):
    scorer = fixed_scorer(0.2)
    scorer.needs_state = True
    scorer.include_checks = include_checks
    policy = make_cascade(scorer=scorer)
    ctx = make_context(budget=123)

    policy.decide(ctx)

    _, state = scorer.score.call_args.args
    assert state["user_request"] == ctx.user_prompt
    assert state["proposed_call"]["tool"] == "send_money"
    assert ("derived_checks" in state) == include_checks
    assert "k_remaining" not in state
    assert "task_id" not in state
