"""Progress milestones and Telegram failures, without sending any live messages."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from approval_gate.notifications import ProgressNotifier, TelegramSender, schedule_progress
from approval_gate.telemetry.events import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NotificationTests(unittest.TestCase):
    def test_eighth_milestones_and_exactly_one_terminal_message(self):
        messages = []
        now = [0.0]
        with ProgressNotifier(
            "pilot",
            144,
            enabled=True,
            sender=messages.append,
            clock=lambda: now[0],
            output="results/pilot",
        ) as progress:
            for done in range(1, 145):
                now[0] += 60
                progress.update(done)
                progress.update(done)  # repeated observations cannot repeat milestones
        self.assertEqual(len(messages), 9)  # start + seven interior milestones + completion
        self.assertEqual(
            [m.splitlines()[1] for m in messages[1:-1]],
            [f"Milestone: {i * 12.5:g}%" for i in range(1, 8)],
        )
        self.assertIn("18/144 (12.5%)", messages[1])
        self.assertIn("ETA: ~2h 06m", messages[1])
        self.assertIn("Completed", messages[-1])
        self.assertIn("144/144 (100.0%)", messages[-1])
        self.assertNotIn("ETA:", messages[-1])

    def test_resume_skips_old_milestones_and_eta_excludes_cached_items(self):
        messages = []
        now = [1000.0]
        with ProgressNotifier(
            "resume", 80, completed=30, enabled=True, sender=messages.append, clock=lambda: now[0]
        ) as progress:
            now[0] += 100
            progress.update(40)
            progress.update(80)
        self.assertEqual(len(messages), 3)
        self.assertIn("30/80 (37.5%)", messages[0])
        self.assertIn("Milestone: 50%", messages[1])
        self.assertIn("ETA: ~6m", messages[1])  # 100/10 * 40 seconds, not 100/40 * 40
        self.assertNotIn("Milestone: 12.5%", "\n".join(messages))

    def test_auto_only_long_jobs_and_explicit_overrides(self):
        for mode, override, total, expected in (
            ("auto", None, 19, False),
            ("auto", None, 20, True),
            ("auto", True, 1, True),
            ("on", None, 1, True),
            ("on", False, 100, False),
            ("off", None, 100, False),
        ):
            with (
                self.subTest(mode=mode, override=override, total=total),
                patch.dict(os.environ, {"TELEGRAM_NOTIFICATIONS": mode}),
            ):
                messages = []
                with ProgressNotifier(
                    "test", total, enabled=override, sender=messages.append
                ) as progress:
                    progress.update(total)
                self.assertEqual(bool(messages), expected)

    def test_failure_interruption_partial_and_completed_with_errors_are_distinct(self):
        for error in (RuntimeError("private prompt must not be sent"), KeyboardInterrupt()):
            messages = []
            with self.assertRaises(type(error)):
                with ProgressNotifier("test", 8, enabled=True, sender=messages.append) as progress:
                    progress.update(1)
                    raise error
            self.assertIn(
                "Interrupted" if isinstance(error, KeyboardInterrupt) else "Failed", messages[-1]
            )
            self.assertNotIn("private prompt", messages[-1])
            self.assertNotIn("100.0%", messages[-1])
        messages = []
        with ProgressNotifier("partial", 8, enabled=True, sender=messages.append) as progress:
            progress.update(2, failed=1)
        self.assertIn("Stopped before completion", messages[-1])
        with ProgressNotifier("errors", 8, enabled=True, sender=messages.append) as progress:
            progress.update(8, failed=1)
        self.assertIn("Finished with errors", messages[-1])

    def test_telegram_transport_is_bounded_and_credentials_never_logged(self):
        requests = []

        def post(url, **kwargs):
            requests.append((url, kwargs))
            return httpx.Response(200, json={"ok": True})

        sender = TelegramSender("secret-token", "private-chat", post=post)
        self.assertTrue(sender("Job reached 12.5%"))
        url, options = requests[0]
        self.assertTrue(url.endswith("/sendMessage"))
        self.assertEqual(options["json"], {"chat_id": "private-chat", "text": "Job reached 12.5%"})
        self.assertLessEqual(options["timeout"].read, 5)
        self.assertFalse(options["follow_redirects"])
        sender = TelegramSender(
            "secret-token",
            "private-chat",
            post=Mock(
                side_effect=RuntimeError(
                    "https://api.telegram.org/botsecret-token/sendMessage private-chat"
                )
            ),
        )
        with self.assertLogs("approval_gate.notifications", level="WARNING") as logs:
            self.assertFalse(sender("progress"))
            self.assertFalse(sender("progress"))
        self.assertEqual(len(logs.output), 1)
        self.assertNotIn("secret-token", logs.output[0])
        self.assertNotIn("private-chat", logs.output[0])
        sender = TelegramSender(
            "test", "chat", post=Mock(return_value=httpx.Response(200, json={"ok": False}))
        )
        with self.assertLogs("approval_gate.notifications", level="WARNING"):
            self.assertFalse(sender("progress"))
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_BOT_CHAT_ID": ""}):
            post = Mock()
            with self.assertLogs("approval_gate.notifications", level="WARNING"):
                self.assertFalse(TelegramSender(post=post)("progress"))
            post.assert_not_called()

    def test_sender_failure_does_not_abort_workload(self):
        sender = Mock(side_effect=RuntimeError("transport failure"))
        with self.assertLogs("approval_gate.notifications", level="WARNING"):
            with ProgressNotifier("test", 8, enabled=True, sender=sender) as progress:
                for done in range(1, 9):
                    progress.update(done)
        self.assertEqual(progress.completed, 8)

    def test_watcher_counts_limits_and_errors_without_writing_results(self):
        watcher = script_module("watch_progress")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schedule = [{"experiment_id": "config", "case_id": str(i)} for i in range(4)]
            write_json(root / "schedule-test.json", schedule)
            for i, status in enumerate(("completed", "limit", "error")):
                directory = root / "config" / "cases" / str(i)
                write_json(directory / "attempt" / "outcome.json", {"status": status})
                if status != "error":
                    write_json(directory / "selected.json", {"attempt": "attempt"})
            # A terminal result outside the frozen schedule cannot inflate progress.
            write_json(
                root / "another-config" / "cases" / "x" / "a" / "outcome.json",
                {"status": "completed"},
            )
            self.assertEqual(schedule_progress(root, schedule), (3, 1))
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with (
                patch.object(
                    sys, "argv", ["watch_progress.py", "--runs", tmp, "--once", "--notify"]
                ),
                patch("approval_gate.notifications.httpx.post") as post,
                redirect_stdout(io.StringIO()),
            ):
                watcher.main()
                post.assert_not_called()
            after = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_matrix_dry_run_sends_nothing_and_resume_progress_counts_new_cases(self):
        runner = script_module("run_experiments")
        config = {"policy": "full_approve", "k": None, "seed": 0}
        cases = [{"suite": "fixture", "task_id": str(i), "attack": "none"} for i in range(8)]
        messages = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "matrix.yaml"
            manifest.write_text(f"output: {root / 'runs'}\n")
            for case in cases[:4]:
                directory = root / "runs" / fingerprint(config) / "cases" / fingerprint(case)
                write_json(directory / "selected.json", {"attempt": "a"})
                write_json(directory / "a" / "outcome.json", {"status": "completed"})

            def fake_episode(*args, **kwargs):
                return {"status": "completed", "tcr_ad": True}

            with (
                patch.object(runner, "expand", return_value=([config], cases)),
                patch.object(runner, "pin_ollama", side_effect=lambda c: c) as pin,
                patch.object(runner, "validate_checkpoint_cases"),
                patch.object(runner, "run_episode", side_effect=fake_episode) as run,
                patch("approval_gate.notifications.TelegramSender", return_value=messages.append),
                redirect_stdout(io.StringIO()),
            ):
                with patch.object(
                    sys,
                    "argv",
                    ["run_experiments.py", "--manifest", str(manifest), "--notify", "--dry-run"],
                ):
                    runner.main()
                    pin.assert_not_called()
                    run.assert_not_called()
                    self.assertEqual(messages, [])
                with patch.object(
                    sys, "argv", ["run_experiments.py", "--manifest", str(manifest), "--notify"]
                ):
                    runner.main()
            self.assertIn("4/8 (50.0%)", messages[0])
            self.assertEqual(
                [m.splitlines()[1] for m in messages[1:-1]],
                ["Milestone: 62.5%", "Milestone: 75%", "Milestone: 87.5%"],
            )
            self.assertIn("8/8 (100.0%)", messages[-1])
            self.assertNotIn("TELEGRAM", json.dumps(config))

    def test_branch_progress_counts_only_legal_continuations(self):
        branch = script_module("branch_rollouts")
        messages, actions = [], []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "config" / "cases" / "case"
            write_json(source / "selected.json", {"attempt": "original"})
            for budget in (0, 1):
                write_json(
                    source / "original" / "snapshots" / f"{budget}.json",
                    {
                        "case": {
                            "suite": "banking",
                            "task_id": "user_task_0",
                            "family": "f",
                            "split": "train",
                        },
                        "config": {
                            "model": "fixture",
                            "policy_model": "fixture",
                            "policy": "llm_judge",
                        },
                        "policy_details": {},
                        "budget": budget,
                    },
                )

            def fake_episode(config, case, output, *, snapshot, action):
                actions.append((snapshot["budget"], action))
                return dict(
                    status="completed",
                    tcr_ad=True,
                    experiment_id="config",
                    case_id=f"case-{len(actions)}",
                    episode_id="attempt",
                )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "branch_rollouts.py",
                        "--snapshots",
                        str(root / "source"),
                        "--output",
                        str(root / "branches"),
                        "--limit",
                        "2",
                        "--continuation-policy",
                        "cascade",
                        "--notify",
                    ],
                ),
                patch.object(branch, "pin_ollama", side_effect=lambda c: c),
                patch.object(branch, "run_episode", side_effect=fake_episode),
                patch("approval_gate.notifications.TelegramSender", return_value=messages.append),
                redirect_stdout(io.StringIO()),
            ):
                branch.main()
            self.assertEqual(len(actions), 5)
            self.assertNotIn((0, "ESCALATE"), actions)
            self.assertIn((1, "ESCALATE"), actions)
            self.assertIn("5/5 (100.0%)", messages[-1])
            self.assertEqual(len(list((root / "branches").rglob("training_example.json"))), 5)


if __name__ == "__main__":
    unittest.main()
