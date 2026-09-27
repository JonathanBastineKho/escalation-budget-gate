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
