"""Exercise actual AgentDojo execution and suite schemas without model calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from approval_gate.decisions import RunMeta
from approval_gate.executor import GatedToolsExecutor
from approval_gate.gate import ApprovalGate
from approval_gate.nodes.allowlist import Tier, check_coverage, classify
from approval_gate.policies.task_scoped import ScopeExtractor
from test_task_scoped import ARGS, PROMPT, call, envelope, make_policy


def runtime_with_payment(fail_first=False):
    runtime = FunctionsRuntime()
    state = {"attempts": 0, "payments": 0}

    @runtime.register_function
    def send_money(recipient: str, amount: float, subject: str, date: str) -> str:
        """Make a local test payment.

        :param recipient: Destination.
        :param amount: Amount.
        :param subject: Purpose.
        :param date: Payment date.
        """
        state["attempts"] += 1
        if fail_first and state["attempts"] == 1:
            raise ValueError("temporary failure before mutation")
        state["payments"] += 1
        return "paid"

    return runtime, state


def executor(policy, k=0, reviewer=None):
    gate = ApprovalGate(policy, RunMeta("r", "banking", policy.name, k, 0, "a", "r"),
                        reviewer=reviewer)
    gate.start_task(PROMPT)
    return GatedToolsExecutor(gate)


def execute(ex, runtime, calls):
    messages = [{"role": "assistant", "content": [], "tool_calls": calls}]
    return ex.query(PROMPT, runtime, EmptyEnv(), messages)[3][1:]


def test_same_batch_duplicate_executes_once():
    runtime, state = runtime_with_payment()
    results = execute(executor(make_policy()), runtime, [call(), call()])
    assert state == {"attempts": 1, "payments": 1}
    assert results[0]["error"] is None and results[1]["error"]


def test_real_failed_execution_can_retry_then_blocks_duplicate():
    runtime, state = runtime_with_payment(fail_first=True)
    results = execute(executor(make_policy()), runtime, [call(), call(), call()])
    assert state == {"attempts": 2, "payments": 1}
    assert results[0]["error"] and results[1]["error"] is None and results[2]["error"]


def test_reviewed_success_is_consumed_within_batch():
    runtime, state = runtime_with_payment()
    reviews = []

    def review(ctx):
        reviews.append(ctx)
        return SimpleNamespace(verdict="APPROVE", reason="valid", tokens_in=1, tokens_out=1)

    policy = make_policy(envelope({"amount": 50}, complete=False))
    results = execute(executor(policy, 2, SimpleNamespace(review=review)),
                      runtime, [call(), call()])
    assert len(reviews) == 1 and state["payments"] == 1
    assert results[1]["error"]


def test_nested_call_cannot_bypass_gate_through_a_read():
    runtime, state = runtime_with_payment()

    @runtime.register_function
    def read_file(filename: str) -> str:
        """Read a local test value.

        :param filename: File name.
        """
        return filename

    nested = FunctionCall(function="read_file", args={"filename": call()})
    results = execute(executor(make_policy()), runtime, [nested])
    assert state["payments"] == 0
    assert "Nested tool calls" in results[0]["error"]


def test_later_reviewer_sees_results_from_same_batch():
    runtime, _ = runtime_with_payment()
    histories = []

    def review(ctx):
        histories.append(ctx.messages)
        return SimpleNamespace(verdict="DENY", reason="uncertain", tokens_in=0, tokens_out=0)

    policy = make_policy({"actions": []})
    execute(executor(policy, 2, SimpleNamespace(review=review)), runtime,
            [call(), call(args={**ARGS, "subject": "different"})])
    assert len(histories) == 2
    assert histories[1][-1]["role"] == "tool"
    assert histories[1][-1]["error"]


@pytest.mark.parametrize("domain", ["banking", "workspace", "slack", "travel"])
def test_all_suite_tool_schemas_are_usable(domain):
    suite = get_suite("v1.2.1", domain)
    check_coverage(t.name for t in suite.tools)
    schemas = {t.name: t.parameters.model_json_schema() for t in suite.tools
               if classify(t.name).tier is not Tier.READ_ONLY}
    extractor = ScopeExtractor(None, "test", schemas)
    assert extractor.schemas
    assert all("properties" in schema for schema in extractor.schemas.values())


@pytest.mark.parametrize("policy_name", ["task_scoped", "task_scoped_budget"])
def test_benchmark_entrypoint_builds_policy_and_writes_extraction_cost(
    policy_name, monkeypatch, tmp_path
):
    runner_path = Path(__file__).parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_under_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(envelope())))],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(runner, "build_client", lambda seed, **kwargs: client)
    monkeypatch.setattr(runner, "build_pipeline", lambda model, gate, name, seed, agent_meter:
                        SimpleNamespace(gate=gate, name=name))

    def benchmark(pipeline, suite, **kwargs):
        pipeline.gate.start_case("user_task_0", None)
        pipeline.gate.start_task(PROMPT)
        assert pipeline.gate.decide(call(), []).approved
        assert pipeline.gate.k_remaining == 0
        assert pipeline.gate.policy.budget_aware == (policy_name == "task_scoped_budget")
        pipeline.gate.end_case(True, None, None)
        return {"utility_results": {("user_task_0", None): True}}

    monkeypatch.setattr(runner, "benchmark_suite_without_injections", benchmark)
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [str(runner_path), "--model", "agent",
                                   "--gate-model", "reviewer", "--policy-model", "extractor",
                                   "--policy", policy_name, "--k", "0"])
    runner.main()
    assert len(requests) == 1 and requests[0]["model"] == "extractor"
    log = next((tmp_path / "results" / "decisions").glob("*.jsonl"))
    record = json.loads(log.read_text())
    assert record["model_policy"] == "extractor"
    assert record["tokens_in"] == {"policy": 100}
    assert record["policy_node"] == "scope_exact_match"

    case_log = next((tmp_path / "results" / "cases").glob("*.jsonl"))
    case = json.loads(case_log.read_text())
    assert case["tokens_in"]["policy"] == 100
    assert case["task_id"] == record["task_id"] == "user_task_0"
