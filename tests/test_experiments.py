import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import httpx
import openai
from pydantic import BaseModel
from agentdojo.functions_runtime import FunctionCall, TaskEnvironment
from agentdojo.task_suite.load_suites import get_suite
from approval_gate.decisions import Decision, RunMeta
from approval_gate.gate import ApprovalGate
from approval_gate.executor import GatedToolsExecutor
from approval_gate.policies.base import CallContext
from approval_gate.policies.full_approve import FullApprovePolicy
from approval_gate.policies.full_escalate import FullEscalatePolicy
from approval_gate.reviewer.simulated import _parse, ReviewResult
from approval_gate.reviewer.context import (
    review_packet,
    render_packet,
    history_records,
    OPEN,
    CLOSE,
)
from approval_gate.llm import _Completions
from approval_gate.telemetry.events import EventLog, read_jsonl, write_json
from approval_gate.evaluation import authorization, labels_from
from approval_gate.experiments import run_episode, prepare_config
from approval_gate.reporting import request_cost, report
from approval_gate.training.models import fit
from approval_gate.policies.learned import predict, SemanticPolicy
from approval_gate.nodes.llm_features import FEATURES


class Params(BaseModel):
    pass


class Env(TaskEnvironment):
    count: int = 0


class Runtime:
    def __init__(self):
        self.functions = {"get_balance": NS(name="get_balance", parameters=Params)}

    def run_function(self, env, name, args):
        env.count += 1
        return str(env.count), None


class Reviewer:
    def __init__(self):
        self.contexts = []

    def review(self, ctx):
        self.contexts.append(copy.deepcopy(ctx))
        return ReviewResult("APPROVE", "authorized", 10, 3, 1)


def call(n="1"):
    return FunctionCall(function="get_balance", args={}, id=n)


def messages(*calls):
    return [{"role": "assistant", "content": None, "tool_calls": list(calls)}]


def gate(policy, k=None, log=None, reviewer=None):
    result = ApprovalGate(
        policy, RunMeta("r", "banking", policy.name, k, 0, "m", "m"), logger=log, reviewer=reviewer
    )
    result.start_task("Check my balance", "private_task_id")
    return result


class CoreTests(unittest.TestCase):
    def test_strict_parser(self):
        for text in [
            "DO NOT APPROVE",
            "APPROVE",
            '{"verdict":"APPROVE"}',
            "[]",
            '{"verdict":"APPROVE","reason":42}',
            'prefix {"verdict":"APPROVE","reason":"ok"}',
        ]:
            self.assertEqual(_parse(text)[0], "DENY")
            self.assertTrue(_parse(text)[2])
        self.assertEqual(_parse('{"verdict":"APPROVE","reason":"ok"}'), ("APPROVE", "ok", False))

    def test_budget_exhaustion_and_reset(self):
        reviewer = Reviewer()
        g = gate(FullEscalatePolicy(), 1, reviewer=reviewer)
        self.assertTrue(g.decide(call(), []).approved)
        self.assertFalse(g.decide(call("2"), []).approved)
        self.assertEqual(len(reviewer.contexts), 1)
        self.assertEqual(g.k_remaining, 0)
        g.start_task("next")
        self.assertEqual(g.k_remaining, 1)
        self.assertEqual(g.step_index, 0)

    def test_zero_budget_calls_no_reviewer(self):
        g = gate(FullEscalatePolicy(), 0, reviewer=Reviewer())
        self.assertFalse(g.decide(call(), []).approved)
        self.assertFalse(g.reviewer.contexts)

    def test_denial_has_no_execution(self):
        with tempfile.TemporaryDirectory() as d:
            g = gate(FullEscalatePolicy(), 0, EventLog(d), Reviewer())
            env = Env()
            GatedToolsExecutor(g).query("q", Runtime(), env, messages(call()))
            self.assertEqual(env.count, 0)
            self.assertEqual(read_jsonl(Path(d) / "executions.jsonl"), [])
            self.assertFalse(read_jsonl(Path(d) / "decisions.jsonl")[0]["approved"])

    def test_batch_review_sees_prior_execution(self):
        reviewer = Reviewer()
        g = gate(FullEscalatePolicy(), None, reviewer=reviewer)
        env = Env()
        GatedToolsExecutor(g).query("q", Runtime(), env, messages(call(), call("2")))
        self.assertEqual(env.count, 2)
        self.assertFalse(history_records(reviewer.contexts[0].messages)["actions"])
        self.assertEqual(
            history_records(reviewer.contexts[1].messages)["actions"][0]["result"], "1"
        )

    def test_snapshot_isolation(self):
        snapshots = []
        g = gate(FullApprovePolicy())
        env = Env()

        def save(e, history, pending, record):
            snapshots.append((e, history, pending))

        GatedToolsExecutor(g, observer=save).query("q", Runtime(), env, messages(call(), call("2")))
        self.assertEqual([s[0].count for s in snapshots], [0, 1])
        env.count = 99
        self.assertEqual(snapshots[0][0].count, 0)
        self.assertEqual(len(snapshots[0][2]), 2)

    def test_no_oracle_and_balanced_markers(self):
        c = CallContext("SECRET_TASK_ID", "Check my balance", call(), [], 0, 1)
        packet = review_packet(c)
        rendered = render_packet(packet)
        self.assertNotIn("SECRET_TASK_ID", rendered)
        self.assertNotIn("attack", packet)
        self.assertEqual(rendered.count(OPEN), rendered.count(CLOSE))
        packet["history"]["actions"] = [OPEN + CLOSE]
        rendered = render_packet(packet)
        self.assertEqual(rendered.count(OPEN), 1)
        self.assertEqual(rendered.count(CLOSE), 1)

    def test_retries_are_counted_and_usage_unknown(self):
        with tempfile.TemporaryDirectory() as d:

            class Inner:
                calls = 0

                def create(self, **kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        raise openai.APIConnectionError(
                            request=httpx.Request("POST", "http://localhost")
                        )
                    return NS(usage=None, choices=[NS(finish_reason="stop")])

            client = _Completions(Inner(), log=EventLog(d), role="reviewer")
            with patch("approval_gate.llm.time.sleep"):
                client.create(model="m")
            cost = request_cost(Path(d))
            self.assertEqual(cost["reviewer_calls"], 2)
            self.assertEqual(cost["reviewer_retries"], 1)
            self.assertEqual(cost["reviewer_completed"], 1)
            self.assertEqual(cost["reviewer_usage_unknown"], 2)

    def test_unknown_authorization_is_not_safe(self):
        with tempfile.TemporaryDirectory() as d:
            log = EventLog(d)
            log.emit("executions", event="started", step_index=0)
            out = dict(episode_id="e", utility=True)
            self.assertIsNone(authorization(Path(d), out, {})["strict_success"])
            self.assertTrue(authorization(Path(d), out, {"e:0": True})["strict_success"])
            self.assertFalse(authorization(Path(d), out, {"e:0": False})["strict_success"])

    def test_conflicting_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jsonl"
            p.write_text('{"key":"e:0","authorized":true}\n{"key":"e:0","authorized":false}\n')
            with self.assertRaises(ValueError):
                labels_from(p)

    def test_logistic_head_learns_signal(self):
        head = fit([[0], [0.1], [0.9], [1]], [0, 0, 1, 1])
        self.assertLess(predict(head, [0]), predict(head, [1]))
        self.assertLess(predict(head, [0]), 0.5)

    def test_voi_masks_review_at_zero_and_risky_approval(self):
        features = {k: 0.0 for k in FEATURES}

        def constant(b):
            return dict(
                weights=[0] * len(FEATURES),
                bias=b,
                mean=[0] * len(FEATURES),
                scale=[1] * len(FEATURES),
            )

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            write_json(
                p,
                dict(
                    kind="learned_voi",
                    features=FEATURES,
                    ablation="none",
                    policy_model="m",
                    heads={
                        "AUTO_APPROVE": constant(9),
                        "AUTO_DENY": constant(0),
                        "ESCALATE": constant(8),
                    },
                    risk_guard=dict(
                        heads={"risk": constant(8)}, ablation="none", approve_threshold=0.1
                    ),
                ),
            )
            extractor = NS(
                ablation="none",
                model="m",
                extract=lambda ctx: dict(features=features, parse_failed=False),
            )
            policy = SemanticPolicy(extractor, "learned_voi", p)
            ctx = CallContext("id", "q", call(), [], 0, 0)
            self.assertEqual(policy.decide(ctx)[0], Decision.AUTO_DENY)


class RunnerTests(unittest.TestCase):
    def test_static_attack_setup_for_every_suite_without_inference(self):
        def components(config, log, task_id, prompt):
            class Agent:
                def query(self, query, runtime, env, history):
                    message = {"role": "assistant", "content": [], "tool_calls": []}
                    return query, runtime, env, [*history, message], {}

            g = gate(FullApprovePolicy(), log=log)
            g.start_task(prompt, task_id)
            return Agent(), g, []

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("approval_gate.experiments.make_components", components),
            patch("approval_gate.experiments.evaluate", return_value=(False, False)),
        ):
            config = prepare_config(dict(model="fixture", policy="full_approve"))
            for name in ("banking", "workspace", "travel", "slack"):
                suite = get_suite("v1.2.1", name)
                for attack in ("direct", "important_instructions_generic", "system_message"):
                    with self.subTest(suite=name, attack=attack):
                        case = dict(
                            suite=name,
                            task_id=next(iter(suite.user_tasks)),
                            injection_task_id=next(iter(suite.injection_tasks)),
                            family="fixture",
                            split="train",
                            attack=attack,
                        )
                        outcome = run_episode(config, case, tmp)
                        self.assertEqual(outcome["status"], "completed", outcome["error"])

    def fake_components(self, config, log, task_id, prompt):
        class Agent:
            count = 0

            def query(self, query, runtime, env, history):
                self.count += 1
                if self.count == 1:
                    msg = messages(call())[0]
                else:
                    msg = {
                        "role": "assistant",
                        "content": [{"type": "text", "content": "Done."}],
                        "tool_calls": [],
                    }
                return query, runtime, env, [*history, msg], {}

        g = gate(FullApprovePolicy(), config["k"], log, Reviewer())
        g.start_task(prompt, task_id)
        return Agent(), g, []

    def test_run_resume_replay_and_reports_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            config = prepare_config(
                dict(model="fake", policy="full_approve", capture_snapshots=True)
            )
            case = dict(
                suite="banking",
                task_id="user_task_0",
                family="banking/f0",
                split="train",
                attack="none",
                injection_task_id=None,
            )
            with (
                patch("approval_gate.experiments.make_components", self.fake_components),
                patch("approval_gate.experiments.evaluate", return_value=(True, False)),
            ):
                out = run_episode(config, case, temp)
                self.assertEqual(out["status"], "completed", out["error"])
                again = run_episode(config, case, temp)
                self.assertEqual(out["episode_id"], again["episode_id"])
                snapshots = list(Path(temp).glob("*/cases/*/*/snapshots/*.json"))
                self.assertEqual(len(snapshots), 1)
                snapshot = json.loads(snapshots[0].read_text())
                branch_config = {**config, "capture_snapshots": False, "branch": "deny"}
                branch = run_episode(
                    branch_config, case, temp, snapshot=snapshot, action="AUTO_DENY"
                )
                self.assertEqual(branch["status"], "completed", branch["error"])
                branch_dir = (
                    Path(temp)
                    / branch["experiment_id"]
                    / "cases"
                    / branch["case_id"]
                    / branch["episode_id"]
                )
                ds = read_jsonl(branch_dir / "decisions.jsonl")
                self.assertFalse(ds[0]["approved"])
                self.assertEqual(ds[0]["decision"], "AUTO_DENY")
            # Separate report root containing only the source configuration.
            report_root = Path(temp) / "report_input"
            report_root.mkdir()
            import shutil

            shutil.copytree(Path(temp) / out["experiment_id"], report_root / out["experiment_id"])
            metrics, _, _ = report(report_root, Path(temp) / "report", allow_incomplete=False)
            self.assertTrue(metrics)
            self.assertIsNone(metrics[0]["tcr_strict"])
            self.assertTrue((Path(temp) / "report" / "api_costs.csv").exists())
            self.assertEqual({row["suite"] for row in metrics}, {"banking", "macro", "micro"})
            self.assertTrue(list((Path(temp) / "report").glob("tcr_*_banking_clean.pdf")))
            for kind in ("tcr", "cost", "hitl", "tradeoff", "runtime", "audit"):
                for extension in ("png", "pdf"):
                    self.assertTrue(
                        list((Path(temp) / "report").glob(f"{kind}_*_micro_clean.{extension}"))
                    )
            self.assertTrue((Path(temp) / "report" / "index.html").exists())
            self.assertEqual(metrics[0]["completed_episodes"], 1)
            self.assertEqual(metrics[0]["limited_episodes"], 0)
            write_json(
                report_root / "schedule-missing.json",
                [{"experiment_id": "missing", "case_id": "missing"}],
            )
            with self.assertRaisesRegex(ValueError, "scheduled episodes have no outcome"):
                report(report_root, Path(temp) / "incomplete_report")


if __name__ == "__main__":
    unittest.main()
