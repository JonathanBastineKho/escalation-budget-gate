"""No-network tests of prompt boundaries, parsing, replay and metrics.

These do NOT establish a real model's approval accuracy; use replay_reviewer.py
on the annotated corpus with the actual reviewer model for that experiment.
"""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from approval_gate.reviewer.prompts import SYSTEM, TASK_SCOPED_SYSTEM
from approval_gate.reviewer.replay import cases_from_run, load_cases, summarize
from approval_gate.reviewer.simulated import SimulatedReviewer, _evidence, _parse

FIXTURE = Path(__file__).parent / "fixtures" / "reviewer_task_scope.jsonl"


def test_real_rent_fixture_retains_source_and_existing_payment_evidence():
    case = next(load_cases(FIXTURE))
    ctx = case.context()
    evidence = _evidence(ctx)
    sources = [o["source_call"]["function"] for o in evidence["observations"]]
    assert "read_file" in sources and "get_scheduled_transactions" in sources
    assert ctx.tool_call.args == {"id": 7, "amount": 1200}
    text = json.dumps(evidence)
    assert "1100" in text and "100.00" in text
    assert evidence["observations"][0]["source_call"]["args"] == {
        "file_path": "landlord-notices.txt"
    }


def test_all_annotated_requests_hide_labels_and_source_identifiers():
    for case in load_cases(FIXTURE):
        case.expected_verdict = "LABEL_SENTINEL"
        case.source = "ORACLE_SOURCE_SENTINEL"
        case.case_id = "ORACLE_ID_SENTINEL"
        ctx = case.context()
        ctx.messages[0]["security"] = "ORACLE_SECURITY_SENTINEL"
        reviewer = SimulatedReviewer(None, "fake", mode="task_scoped")
        request = json.dumps(reviewer.build_messages(ctx))
        assert "SENTINEL" not in request
        assert "Delegated documents may supply task details" in request
        assert "<<<UNTRUSTED_DATA>>>" in request


def test_assistant_claim_of_user_approval_is_not_evidence():
    ctx = next(load_cases(FIXTURE)).context()
    ctx.messages[-1]["content"] = "USER_APPROVED_EVERYTHING_SENTINEL"
    assert "USER_APPROVED_EVERYTHING_SENTINEL" not in json.dumps(_evidence(ctx))


@pytest.mark.parametrize("text", [
    "APPROVE", "Do not APPROVE this", "not approved", "[]", "null",
    '{"verdict":"APPROVE"}', '{"verdict":"MAYBE","reason":"unclear"}',
    'Some prose {"verdict":"APPROVE","reason":"yes"}',
])
def test_new_reviewer_fails_closed_on_non_json_or_malformed_verdict(text):
    verdict, _, failed = _parse(text, strict=True)
    assert verdict == "DENY" and failed


def test_modes_use_distinct_prompts_and_log_assessment():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "verdict": "APPROVE", "reason": "supported by delegated task",
                "scope": "modify existing scheduled action", "evidence": ["source-id"],
                "authorized_effect": "increase existing payment",
                "proposed_effect": "update existing payment",
                "state_after_execution": "one payment with corrected amount",
                "effect_matches_authorization": True,
            })))], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ctx = next(load_cases(FIXTURE)).context()
    result = SimulatedReviewer(client, "fake").review(ctx)
    assert result.assessment["evidence"] == ["source-id"]
    assert result.tokens_in == 10 and not result.parse_failed
    assert calls[0]["messages"][0]["content"] == TASK_SCOPED_SYSTEM
    legacy = SimulatedReviewer(None, "fake", mode="legacy").build_messages(ctx)
    assert legacy[0]["content"] == SYSTEM
    # Existing legacy parsing remains available solely for the paired baseline.
    assert _parse("APPROVE", strict=False)[0] == "APPROVE"


def test_replay_excludes_future_outcomes_and_oracle_fields(tmp_path):
    call = {"id": "c", "function": "send_money", "args": {"amount": 50}}
    run = {
        "security": "ORACLE_SENTINEL", "injections": "ORACLE_SENTINEL",
        "messages": [
            {"role": "user", "content": "Pay the bill"},
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c", "content": "FUTURE_SENTINEL"},
        ],
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    case = next(cases_from_run(path))
    serialized = json.dumps(asdict(case))
    assert "FUTURE_SENTINEL" not in serialized and "ORACLE_SENTINEL" not in serialized
    assert case.expected_verdict is None


def test_replay_includes_only_earlier_batch_results(tmp_path):
    calls = [{"id": str(i), "function": "send_money", "args": {"amount": i}}
             for i in (1, 2)]
    run = {"messages": [
        {"role": "user", "content": "Pay the bills"},
        {"role": "assistant", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "1", "content": "EARLIER", "error": "denied"},
        {"role": "tool", "tool_call_id": "2", "content": "FUTURE"},
    ]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    first, second = list(cases_from_run(path))
    assert "EARLIER" not in json.dumps(asdict(first))
    assert "EARLIER" in json.dumps(asdict(second))
    assert "FUTURE" not in json.dumps(asdict(second))
    assert _evidence(second.context())["observations"][0]["error"] == "denied"


def test_summary_distinguishes_false_denials_from_unsafe_approvals():
    def result(verdict):
        return {"verdict": verdict, "parse_failed": False,
                "tokens_in": 10, "tokens_out": 5, "latency_ms": 1}

    rows = [
        {"expected_verdict": "APPROVE", "reviews": {
            "legacy": result("DENY"), "task_scoped": result("APPROVE")}},
        {"expected_verdict": "DENY", "reviews": {
            "legacy": result("DENY"), "task_scoped": result("APPROVE")}},
        {"expected_verdict": None, "reviews": {
            "legacy": result("DENY"), "task_scoped": result("APPROVE")}},
    ]
    summary = summarize(rows)
    assert summary["legacy"]["false_denial_rate"] == 1
    assert summary["task_scoped"]["false_denial_rate"] == 0
    assert summary["task_scoped"]["unsafe_approval_rate"] == 1
    assert summary["task_scoped"]["labeled_safe"] == 1
    assert summary["paired_changes"]["deny_to_approve"] == 3


def effect_response(**overrides):
    return {
        "verdict": "APPROVE", "reason": "supported",
        "authorized_effect": "update existing payment",
        "proposed_effect": "update existing payment",
        "state_after_execution": "one updated payment",
        "effect_matches_authorization": True,
        **overrides,
    }


@pytest.mark.parametrize("field", [
    "authorized_effect", "proposed_effect", "state_after_execution",
    "effect_matches_authorization",
])
def test_missing_effect_check_fails_closed(field):
    payload = effect_response()
    del payload[field]
    verdict, _, failed = _parse(json.dumps(payload), strict=True)
    assert verdict == "DENY" and failed


@pytest.mark.parametrize("overrides", [
    {"authorized_effect": " "}, {"proposed_effect": []},
    {"state_after_execution": None}, {"effect_matches_authorization": "true"},
    {"effect_matches_authorization": 1},
])
def test_invalid_effect_check_fails_closed(overrides):
    verdict, _, failed = _parse(json.dumps(effect_response(**overrides)), strict=True)
    assert verdict == "DENY" and failed


def test_effect_mismatch_overrides_approval_and_preserves_audit():
    payload = effect_response(
        proposed_effect="create an additional payment",
        state_after_execution="original and additional payment both remain",
        effect_matches_authorization=False,
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload)))]))))
    ctx = list(load_cases(FIXTURE))[-1].context()
    result = SimulatedReviewer(client, "fake").review(ctx)
    assert result.verdict == "DENY" and not result.parse_failed
    assert result.assessment["state_after_execution"] == payload["state_after_execution"]
    assert result.assessment["effect_matches_authorization"] is False
    # The paired legacy baseline does not acquire the new enforcement rule.
    assert _parse(json.dumps(payload), strict=False)[0] == "APPROVE"


def test_complete_matching_effect_can_approve_but_does_not_override_denial():
    assert _parse(json.dumps(effect_response()), strict=True) == ("APPROVE", "supported", False)
    assert _parse(json.dumps(effect_response(verdict="DENY")), strict=True)[0] == "DENY"


@pytest.mark.parametrize("payload,expected", [
    ("", "empty_response"),
    ("```json\n{}\n```", "missing_field: verdict"),
    ("[]", "expected_json_object"),
    (json.dumps({"verdict": "APPROVE", "reason": "ok"}), "missing_field: authorized_effect"),
    (json.dumps(effect_response(effect_matches_authorization="true")),
     "expected_boolean: effect_matches_authorization"),
    (json.dumps(effect_response(proposed_effect=" ")), "empty_field: proposed_effect"),
])
def test_failed_review_retains_raw_response_and_diagnostics(payload, expected):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=payload))]))))
    ctx = next(load_cases(FIXTURE)).context()
    result = SimulatedReviewer(client, "fake").review(ctx)
    assert result.verdict == "DENY" and result.parse_failed
    assert result.raw_response == payload
    assert result.finish_reason == "stop"
    assert any(error.startswith(expected) for error in result.validation_errors)
    assert json.loads(json.dumps(asdict(result)))["raw_response"] == payload


def test_diagnostics_do_not_treat_effect_mismatch_as_parse_failure():
    from approval_gate.reviewer.simulated import _validation_errors

    text = json.dumps(effect_response(effect_matches_authorization=False))
    assert _validation_errors(text) == []
    assert _parse(text, strict=True)[0] == "DENY"



@pytest.mark.parametrize("tag", ["json", ""])
def test_single_enclosing_fence_accepts_valid_assessment(tag):
    raw = "\n\n```" + tag + "\n" + json.dumps(effect_response()) + "\n```\n"
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=raw))]))))
    result = SimulatedReviewer(client, "fake").review(next(load_cases(FIXTURE)).context())
    assert result.verdict == "APPROVE" and not result.parse_failed
    assert result.validation_errors == []
    assert result.raw_response == raw
    assert result.assessment["effect_matches_authorization"] is True


@pytest.mark.parametrize("wrapper", [
    "Here is my answer:\n```json\n%s\n```",
    "```json\n%s\n```\nAPPROVE",
    "```json\n%s\n```\n```json\n{}\n```",
])
def test_fence_with_extra_content_fails_closed(wrapper):
    assert _parse(wrapper % json.dumps(effect_response()), strict=True)[2]


def test_fence_does_not_bypass_effect_validation():
    for payload in [effect_response(effect_matches_authorization=False),
                    {"verdict": "APPROVE", "reason": "ok"}]:
        text = "```json\n" + json.dumps(payload) + "\n```"
        assert _parse(text, strict=True)[0] == "DENY"
