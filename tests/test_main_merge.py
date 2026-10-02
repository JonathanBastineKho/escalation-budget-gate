"""Regressions at the boundary between upstream and task-scoped behavior."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from approval_gate.decisions import RunMeta
from approval_gate.gate import ApprovalGate
from approval_gate.llm import UsageMeter
from approval_gate.policies.full_escalate import FullEscalatePolicy
from test_task_scoped import PROMPT, call, make_policy


def test_retry_shares_budget_and_case_usage_includes_final_agent_turn():
    cases = []
    meter = UsageMeter()
    reviewer = SimpleNamespace(review=lambda ctx: SimpleNamespace(
        verdict="APPROVE", reason="ok", tokens_in=3, tokens_out=2))
    gate = ApprovalGate(
        FullEscalatePolicy(), RunMeta("r", "banking", "full_escalate", 1, 0, "a", "a"),
        agent_meter=meter, reviewer=reviewer,
        logger=SimpleNamespace(log=lambda record: None, log_case=cases.append),
    )
    gate.start_case("task", "injection")
    gate.start_task(PROMPT)
    meter.tokens_in = 10
    assert gate.decide(call(), []).approved
    gate.start_task(PROMPT)
    denied = gate.decide(call(), [])
    assert not denied.approved and denied.record.decided_by_node == "budget_exhausted"
    meter.tokens_in += 5
    gate.end_case(True, False, None)
    assert cases[0].attempts == 2 and cases[0].escalations == 1
    assert cases[0].tokens_in == {"reviewer": 3, "agent": 15}
    gate.start_case("next", None)
    gate.start_task(PROMPT)
    assert gate.decide(call(), []).approved


def test_task_scope_success_memory_survives_retry_but_resets_per_case():
    policy = make_policy()
    gate = ApprovalGate(policy, RunMeta("r", "banking", policy.name, 0, 0, "a", "a"))
    gate.start_case("task", None)
    gate.start_task(PROMPT)
    assert gate.decide(call(), []).approved
    policy.observe_execution(call(), None)
    gate.start_task(PROMPT)
    assert not gate.decide(call(), []).approved
    gate.start_case("next", None)
    gate.start_task(PROMPT)
    assert gate.decide(call(), []).approved


@pytest.mark.parametrize("policy_name", ["cascade", "cascade_n1", "static_threshold"])
def test_upstream_policies_build_with_separate_reviewer(policy_name, monkeypatch, tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("merge_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    endpoints = []

    def client(seed, **kwargs):
        endpoints.append(kwargs.get("base_url"))
        return object()

    def benchmark(pipeline, suite, **kwargs):
        gate = pipeline.gate
        assert gate.policy.name == policy_name
        assert gate.reviewer.model == "reviewer"
        assert gate.reviewer.mode == "task_scoped"
        if policy_name != "cascade_n1":
            assert gate.policy.extractor.model == "agent"
        return {"utility_results": {("user_task_0", None): True}}

    monkeypatch.setattr(runner, "build_client", client)
    monkeypatch.setattr(runner, "build_pipeline", lambda model, gate, name, **kw:
                        SimpleNamespace(gate=gate, name=name))
    monkeypatch.setattr(runner, "benchmark_suite_without_injections", benchmark)
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setenv("REVIEWER_BASE_URL", "http://reviewer.test/v1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [str(path), "--model", "agent", "--reviewer-model",
                                   "reviewer", "--policy", policy_name])
    runner.main()
    assert endpoints[-1] == "http://reviewer.test/v1"
