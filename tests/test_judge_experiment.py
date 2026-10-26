"""Runner/report integration tests without network requests or model downloads."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import text_content_block_from_string

from _support import PROJECT_ROOT, load_script

runner = load_script("run_benchmark", "scripts/run_benchmark.py")
report = load_script("compare_judges", "scripts/analyze/compare_judges.py")


class ScriptedAgent:
    """Fixed two-call banking fixture; no benchmark labels enter the judge."""

    def query(self, query, runtime, env, messages, extra_args):
        n = sum(m["role"] == "tool" for m in messages)
        call = None
        if n == 0:
            call = FunctionCall(
                id="read", function="read_file", args={"file_path": "bill-december-2023.txt"}
            )
        elif n == 1:
            call = FunctionCall(
                id="pay",
                function="send_money",
                args={
                    "recipient": "UK12345678901234567890",
                    "amount": 98.70,
                    "subject": "Car Rental",
                    "date": "2022-01-01",
                },
            )
        response = {
            "role": "assistant",
            "content": [text_content_block_from_string("Done")],
            "tool_calls": [call] if call else [],
        }
        return query, runtime, env, [*messages, response], extra_args


class RunnerTests(unittest.TestCase):
    def dry_run(self, policy, *extra):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run_benchmark.py"),
                "--model",
                "fixture-agent",
                "--policy",
                policy,
                "--user-tasks",
                "user_task_0",
                "--k",
                "0",
                "--dry-run",
                *extra,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_both_backends_have_matched_ids_but_separate_cache_names(self):
        for backend in ("chat", "decision2"):
            direct = self.dry_run("direct_judge", "--judge-backend", backend)
            sandbox = self.dry_run("sandbox_judge", "--judge-backend", backend)
            self.assertEqual(
                direct["experiment"]["comparison_id"], sandbox["experiment"]["comparison_id"]
            )
            self.assertNotEqual(direct["name"], sandbox["name"])
            changed = self.dry_run(
                "direct_judge", "--judge-backend", backend, "--judge-threshold", "0.9"
            )
            self.assertNotEqual(
                direct["experiment"]["comparison_id"], changed["experiment"]["comparison_id"]
            )

    def test_controls_share_the_judge_comparison_manifest(self):
        for backend in ("chat", "decision2"):
            runs = [self.dry_run(policy, "--judge-backend", backend) for policy in
                    ("direct_judge", "sandbox_judge", "full_approve", "full_escalate")]
            self.assertEqual(len({r["experiment"]["comparison_id"] for r in runs}), 1)
            self.assertEqual(len({r["name"] for r in runs}), 4)

    def test_unhealthy_judge_retains_records_and_fails_the_cell(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError()
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["run_benchmark.py", "--model", "fixture-agent", "--policy", "direct_judge",
                    "--k", "0", "--user-tasks", "user_task_0", "--force-rerun",
                    "--logdir", f"{tmp}/runs"]
            with (patch.object(sys, "argv", argv),
                  patch.object(runner, "build_client", return_value=client),
                  patch("approval_gate.judge_config.build_client", return_value=client),
                  patch.object(runner, "build_llm", return_value=ScriptedAgent()),
                  redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error):
                runner.main()
            self.assertEqual(error.exception.code, 2)
            records = [json.loads(line) for p in Path(tmp, "cases").glob("*.jsonl")
                       for line in p.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["judge_unavailable"], 2)
            self.assertEqual(records[0]["judge_usage_missing"], 2)
            self.assertEqual(records[0]["non_read_only_calls"], 1)
            self.assertEqual(records[0]["non_read_only_executed"], 0)

    def test_runner_executes_clean_and_injected_cases_with_real_simulator(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "probabilities": {"APPROVE": 0.9, "DENY": 0.05, "UNCERTAIN": 0.05},
                                "reason": "scripted mechanics test",
                            }
                        )
                    ),
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            for attack in (
                [],
                [
                    "--attack",
                    "important_instructions_generic",
                    "--injection-tasks",
                    "injection_task_0",
                ],
            ):
                for policy in ("direct_judge", "sandbox_judge"):
                    args = [
                        "run_benchmark.py",
                        "--model",
                        "fixture-agent",
                        "--judge-model",
                        "fixture-judge",
                        "--policy",
                        policy,
                        "--k",
                        "0",
                        "--force-rerun",
                        "--user-tasks",
                        "user_task_0",
                        "--logdir",
                        f"{tmp}/runs",
                        *attack,
                    ]
                    with (
                        patch.object(sys, "argv", args),
                        patch.object(runner, "build_client", return_value=client),
                        patch("approval_gate.judge_config.build_client", return_value=client),
                        patch.object(runner, "build_llm", return_value=ScriptedAgent()),
                        redirect_stdout(io.StringIO()),
                    ):
                        runner.main()
            records = [
                json.loads(line)
                for p in Path(tmp, "cases").glob("*.jsonl")
                for line in p.read_text().splitlines()
            ]
            rows = report.compare(records)
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["n"], 1)
                self.assertEqual(row["utility"], 1)
                self.assertEqual(row["judge_calls"], 2)
                self.assertEqual(row["judge_tokens_in"], 60)
                self.assertEqual(row["preview_calls"], 2 if row["policy"] == "sandbox_judge" else 0)
            self.assertEqual(len(list(Path(tmp, "manifests").glob("*.json"))), 4)
            decisions = [
                json.loads(line)
                for p in Path(tmp, "decisions").glob("*.jsonl")
                for line in p.read_text().splitlines()
            ]
            self.assertEqual(len(decisions), 8)
            self.assertTrue(all(d["executed"] for d in decisions))
            self.assertTrue(all(d["judge_status"] == "ok" for d in decisions))


class ReportTests(unittest.TestCase):
    def pair(self):
        return [
            dict(
                comparison_id="same-config",
                run_id=p,
                policy=p,
                suite="banking",
                attack_name="fixture-attack",
                k_budget=0,
                seed=0,
                task_id="task",
                injection_task_id="injection",
                error=None,
                utility=True,
                security=False,
                tokens_in={"judge": 30},
                tokens_out={"judge": 10},
                judge_calls=1,
            )
            for p in report.POLICIES
        ]

    def test_duplicate_or_unmatched_cases_are_rejected(self):
        pair = self.pair()
        for records in (pair[:1], pair + [pair[0]], []):
            with self.assertRaises(ValueError):
                report.compare(records)
        pair[1]["task_id"] = "other-task"
        with self.assertRaises(ValueError):
            report.compare(pair)

    def test_settings_mismatch_cannot_be_pooled(self):
        pair = self.pair()
        pair[1]["comparison_id"] = "different-threshold"
        with self.assertRaises(ValueError):
            report.compare(pair)

    def test_errors_reduce_completion_but_are_not_counted_as_failed_attacks(self):
        pair = self.pair()
        errored = deepcopy(pair)
        for c in errored:
            c.update(task_id="crashed-task", utility=None, security=None, error="TimeoutError")
        rows = report.compare(pair + errored)
        for row in rows:
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["errors"], 1)
            self.assertEqual(row["tcr_ad"], 0.5)
            self.assertEqual(row["attacks_scored"], 1)
            self.assertEqual(row["asr_scored"], 0.0)


if __name__ == "__main__":
    unittest.main()
