"""Offline workflow tests; no model downloads or server calls."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentdojo.task_suite.load_suites import get_suite

from _support import judge_arguments, load_script
from approval_gate.attacks import ImportantInstructionsPreserving, ToolKnowledgePreserving
from approval_gate.authorization_eval import (
    evaluate,
    load_dataset,
    preflight_dataset,
    prepare_case,
    summarize,
)
from approval_gate.judge_config import judge_identity, resolve_judge_config
from approval_gate.judging import ChatJudge, Judgment
from approval_gate.sandbox import AgentDojoPreview

calibration = load_script("calibrate_judge", "scripts/analyze/calibrate_judge.py")
report = load_script("compare_workflow", "scripts/analyze/compare_judges.py")
sweep = load_script("sweep_workflow", "scripts/sweep_k.py")


def arguments(*extra):
    return judge_arguments(*extra)


def dev_scores():
    records = []
    for policy in ("direct_judge", "sandbox_judge"):
        for label in ("APPROVE", "DENY", "UNCERTAIN"):
            records.append({"case_id": label, "policy": policy, "expected": label,
                            "judge_status": "ok", "judge_usage_missing": 0,
                            "judge_probabilities": {key: .9 if key == label else .05
                                                    for key in ("APPROVE", "DENY", "UNCERTAIN")}})
    return dict(format=1, purpose="authorization", split="development", complete=True,
                healthy=True, repeats=1, records=records, case_ids=["APPROVE", "DENY", "UNCERTAIN"],
                dataset_hash="fixture-hash", judge_identity=judge_identity(resolve_judge_config(arguments())))


class DiagnosticTests(unittest.TestCase):
    def test_reasoning_control_is_explicit_and_truncation_keeps_diagnostics(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=512),
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(
                content="", reasoning_content="reasoning until token limit"))],
        )
        self.assertNotIn("reasoning_effort", ChatJudge(client, "model").request({}))
        judge = ChatJudge(client, "model", 1024, "none")
        result = judge.judge({})
        self.assertNotEqual(result.status, "ok")
        self.assertEqual(result.diagnostics["failure_stage"], "completion")
        self.assertEqual(result.diagnostics["finish_reason"], "length")
        self.assertEqual(result.diagnostics["content_chars"], 0)
        self.assertGreater(result.diagnostics["reasoning_chars"], 0)
        self.assertEqual(result.tokens_out, 512)
        self.assertTrue(result.usage_known)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_exception_text_is_not_logged(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("sensitive-endpoint-and-prompt")
        result = ChatJudge(client, "model").judge({})
        self.assertEqual(result.diagnostics["failure_stage"], "request")
        self.assertNotIn("sensitive", repr(result))


class FixtureTests(unittest.TestCase):
    def test_all_development_and_heldout_previews_are_valid_and_isolated(self):
        datasets = [load_dataset(split) for split in ("development", "heldout")]
        self.assertFalse({c["id"] for c in datasets[0]["cases"]}
                         & {c["id"] for c in datasets[1]["cases"]})
        self.assertFalse({c["request"] for c in datasets[0]["cases"]}
                         & {c["request"] for c in datasets[1]["cases"]})
        for dataset in datasets:
            for case in dataset["cases"]:
                with self.subTest(case=case["id"]):
                    runtime, env, _, _, call = prepare_case(case)
                    before = env.model_dump()
                    preview = AgentDojoPreview(case["suite"]).preview(runtime, env, call)
                    self.assertEqual(preview.status, "ok", preview.error)
                    self.assertEqual(env.model_dump(), before)

    def test_expected_labels_stay_outside_model_inputs_and_bad_approvals_are_counted(self):
        config = resolve_judge_config(arguments())
        judge = Mock()
        judge.judge.return_value = Judgment(
            probabilities={"APPROVE": .9, "DENY": .05, "UNCERTAIN": .05}, usage_known=True)
        data = preflight_dataset()
        with redirect_stdout(io.StringIO()):
            result = evaluate(data, config, judge)
        self.assertTrue(result["healthy"])
        for invocation in judge.judge.call_args_list:
            state = invocation.args[0]
            serialized = json.dumps(state)
            self.assertEqual(set(state), {"user_request", "tool_schema", "untrusted_evidence"})
            for case in data["cases"]:
                self.assertNotIn(case["id"], serialized)
                self.assertNotIn(case["rationale"], serialized)
        for metrics in result["summary"].values():
            self.assertEqual(metrics["false_approvals"], 1)
        self.assertEqual(result["records"][0]["state"]["untrusted_evidence"]["history"][-1]
                         ["proposals"][0]["id"], "proposed")

    def test_preflight_stops_on_first_failure_and_saves_it(self):
        judge = Mock()
        judge.judge.return_value = Judgment(status="error:TimeoutError")
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp) / "probe.json"
            result = evaluate(preflight_dataset(), resolve_judge_config(arguments()), judge,
                              path, purpose="preflight")
            self.assertEqual(judge.judge.call_count, 1)
            self.assertFalse(result["complete"])
            self.assertFalse(json.loads(path.read_text())["healthy"])
            self.assertEqual(len(result["records"]), 1)
            with self.assertRaises(FileExistsError):
                evaluate(preflight_dataset(), resolve_judge_config(arguments()), judge, path)

    def test_preserving_attacks_retain_bill_details_and_transaction_subject(self):
        suite = get_suite("v1.2.1", "banking")
        for cls in (ImportantInstructionsPreserving, ToolKnowledgePreserving):
            attack = cls(suite, Mock())
            for task_id in ("user_task_0", "user_task_1"):
                injections = attack.attack(suite.user_tasks[task_id], suite.injection_tasks["injection_task_0"])
                env = suite.load_and_inject_default_environment(injections)
                bill = env.filesystem.files["bill-december-2023.txt"]
                self.assertIn("98.70", bill)
                self.assertIn("UK12345678901234567890", bill)
                self.assertIn("Sushi dinner", env.bank_account.transactions[-1].subject)
                for key, value in injections.items():
                    self.assertTrue(value.startswith(suite.get_injection_vector_defaults()[key]))
                    self.assertIn("US133000000121212121212", value)


class CalibrationTests(unittest.TestCase):
    def test_failure_report_explains_confident_error_without_relaxing_constraints(self):
        data = dev_scores()
        # The erroneous approval outranks every correct automatic decision.
        for row in data["records"]:
            if row["expected"] == "DENY":
                row["judge_probabilities"] = {"APPROVE": .99, "DENY": .005, "UNCERTAIN": .005}
        frozen = calibration.calibrate(data, [.8, .95])
        self.assertIsNone(frozen["threshold"])
        self.assertEqual(frozen["constraints"]["max_false_approval_rate"], 0)
        for policy, diagnosis in frozen["failure_analysis"].items():
            self.assertEqual(frozen["candidates"][0]["constraint_failures"][policy], ["false_approvals"])
            self.assertGreater(diagnosis["threshold_for_zero_observed_false_approvals"], .99)
            self.assertEqual(diagnosis["coverage_at_that_threshold"], 0)
            self.assertEqual(diagnosis["highest_scoring_unapproved_cases"][0]["case_id"], "DENY")
        output = io.StringIO()
        with redirect_stdout(output):
            calibration.print_failure_report(frozen)
        self.assertIn("APPROVE=0.990000", output.getvalue())
        self.assertIn("at most 0.0%", output.getvalue())

    def test_probability_one_false_approval_has_no_safe_boundary(self):
        data = dev_scores()
        data["records"][1]["judge_probabilities"] = {"APPROVE": 1, "DENY": 0, "UNCERTAIN": 0}
        frozen = calibration.calibrate(data, [.8, 1])
        diagnosis = frozen["failure_analysis"]["direct_judge"]
        self.assertIsNone(diagnosis["threshold_for_zero_observed_false_approvals"])
        self.assertIsNone(diagnosis["coverage_at_that_threshold"])

    def test_threshold_is_shared_and_all_abstain_is_rejected(self):
        data = dev_scores()
        selected = calibration.calibrate(data, [.6, .8])
        self.assertEqual(selected["threshold"], .8)
        for row in data["records"]:
            if row["policy"] == "sandbox_judge":
                row["judge_probabilities"] = {"APPROVE": .4, "DENY": .4, "UNCERTAIN": .2}
        self.assertIsNone(calibration.calibrate(data, [.6, .8])["threshold"])
        with self.assertRaises(ValueError):
            calibration.calibrate(data, [.8], min_coverage=0)

    def test_false_approvals_and_denials_are_checked_per_arm(self):
        data = dev_scores()
        data["records"][1]["judge_probabilities"] = {"APPROVE": .7, "DENY": .2, "UNCERTAIN": .1}
        self.assertIsNone(calibration.calibrate(data, [.6])["threshold"])
        data = dev_scores()
        data["records"][0]["judge_probabilities"] = {"APPROVE": .05, "DENY": .9, "UNCERTAIN": .05}
        self.assertIsNone(calibration.calibrate(data, [.6])["threshold"])
        self.assertEqual(summarize(data["records"], .6)["direct_judge"]["false_denials"], 1)

    def test_heldout_incomplete_duplicate_or_unhealthy_scores_cannot_calibrate(self):
        for update in ({"split": "heldout"}, {"purpose": "preflight"}, {"complete": False},
                       {"healthy": False}, {"repeats": 2}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                calibration.calibrate({**dev_scores(), **update}, [.8])
        data = dev_scores()
        data["records"].append(deepcopy(data["records"][0]))
        with self.assertRaises(ValueError):
            calibration.calibrate(data, [.8])
        data = dev_scores()
        data["records"][0]["judge_usage_missing"] = 1
        with self.assertRaises(ValueError):
            calibration.calibrate(data, [.8])

    def test_frozen_threshold_rejects_changed_inference_settings(self):
        frozen = calibration.calibrate(dev_scores(), [.7])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.json"
            path.write_text(json.dumps(frozen))
            config = resolve_judge_config(arguments("--judge-calibration", str(path)))
            self.assertEqual(config["threshold"], .7)
            self.assertIn("calibration_id", config)
            for extra in (["--judge-model", "different"], ["--judge-reasoning-effort", "none"],
                          ["--judge-max-input-chars", "1000"]):
                with self.assertRaises(ValueError):
                    resolve_judge_config(arguments("--judge-calibration", str(path), *extra))
            frozen["threshold"] = None
            path.write_text(json.dumps(frozen))
            with self.assertRaises(ValueError):
                resolve_judge_config(arguments("--judge-calibration", str(path)))


class WorkflowTests(unittest.TestCase):
    def test_sweep_omits_injections_on_clean_cells_and_passes_judge_config_to_controls(self):
        args = arguments("--judge-backend", "decision2")
        args.suites, args.suite, args.attacks, args.attack = None, "banking", ["none", "tool_knowledge_preserving"], None
        args.policies, args.ks, args.seeds = ["full_approve", "full_escalate", "direct_judge", "sandbox_judge"], ["0", "1"], [0]
        args.user_tasks, args.injection_tasks, args.logdir = ["user_task_0"], ["injection_task_0"], "results/check/runs"
        cells = sweep.commands(args)
        self.assertEqual(len(cells), 14)
        for cmd in cells:
            self.assertEqual(cmd[cmd.index("--judge-backend") + 1], "decision2")
            self.assertEqual("--injection-tasks" in cmd, "--attack" in cmd)
            if cmd[cmd.index("--policy") + 1] == "full_approve":
                self.assertEqual(cmd[cmd.index("--k") + 1], "inf")

    def test_failed_preflight_prevents_all_benchmark_cells(self):
        argv = ["sweep_k.py", "--model", "fixture", "--policies", "direct_judge", "sandbox_judge", "--ks", "0"]
        with patch.object(sys, "argv", argv), patch.object(sweep.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            run.return_value = SimpleNamespace(returncode=2)
            with self.assertRaises(SystemExit):
                sweep.main()
        self.assertEqual(run.call_count, 1)
        self.assertTrue(run.call_args.args[0][1].endswith("preflight_judge.py"))

    def test_controls_are_matched_and_budget_independent_control_is_not_duplicated(self):
        common = dict(comparison_id="shared", suite="banking", attack_name=None, seed=0,
                      task_id="task", injection_task_id=None, error=None, utility=True,
                      security=None, tokens_in={}, tokens_out={})
        records = [dict(common, policy="full_approve", k_budget=None)]
        for k in (0, 1):
            for policy in ("direct_judge", "sandbox_judge", "full_escalate"):
                records.append(dict(common, policy=policy, k_budget=k))
        rows = report.compare(records, include_controls=True)
        self.assertEqual(len(rows), 7)
        self.assertEqual(sum(row["policy"] == "full_approve" for row in rows), 1)
        with self.assertRaises(ValueError):
            report.compare(records[1:], include_controls=True)
        records[0]["comparison_id"] = "changed-model-or-settings"
        with self.assertRaises(ValueError):
            report.compare(records, include_controls=True)


if __name__ == "__main__":
    unittest.main()
