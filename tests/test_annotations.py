"""Offline annotation workflow, using a mock HTTP transport instead of real inference."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
import openai

from approval_gate.annotations import (
    blind_rows,
    packet_hash,
    read_annotations,
    resolve_reviews,
    write_annotations,
)
from approval_gate.annotation_labeler import label_packets, parse_label
from approval_gate.evaluation import labels_from
from approval_gate.experiments import prepare_config
from approval_gate.nodes.llm_features import FEATURES
from approval_gate.reporting import request_cost

ROOT = Path(__file__).resolve().parents[1]


def packets(count=2):
    return [
        dict(
            key=f"opaque:{index}",
            split="validation",
            family=f"validation/{index}",
            packet=dict(
                user_request=f"Read my balance {index}",
                proposed_call={"name": "get_balance", "arguments": {}},
                tool_schema={},
                history={"earlier_records_omitted": 0, "actions": []},
            ),
            authorized=None,
            annotator=None,
            rationale="",
            features={name: 0.5 for name in FEATURES},
            policy_model="feature-model",
            feature_signature="fixture",
            trajectory="secret-future-outcome",
            executed=True,
        )
        for index in range(count)
    ]


def machine_rows():
    rows = packets()
    for index, row in enumerate(rows):
        row.update(
            authorized=not bool(index),
            label_source="llm",
            human_verified=False,
            reviewed=False,
            annotator="qwen3.8:27b",
            rationale="hidden machine reason",
            machine_annotation={
                "model": "qwen3.8:27b",
                "status": "labeled",
                "authorized": not bool(index),
                "reason": "hidden reason",
            },
        )
    return rows


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(name, *args, answers=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{name}.py"), *map(str, args)],
        cwd=ROOT,
        input=answers,
        capture_output=True,
        text=True,
        timeout=60,
    )


class AnnotationTests(unittest.TestCase):
    def test_blind_projection_hides_machine_answers_and_outcomes(self):
        rows = machine_rows()
        blind = blind_rows(rows)
        text = json.dumps(blind)
        for secret in (
            "hidden",
            "qwen",
            "features",
            "trajectory",
            "executed",
            "family",
            "validation",
            "secret-future-outcome",
            "machine_annotation",
        ):
            self.assertNotIn(secret, text)
        self.assertIsNone(blind[0]["authorized"])
        self.assertFalse(blind[0]["reviewed"])
        self.assertEqual(blind[0]["packet"], rows[0]["packet"])
        blind[0]["packet"]["user_request"] = "changed"
        self.assertNotEqual(blind[0]["packet"], rows[0]["packet"])

    def test_human_override_and_abstention_preserve_original_machine_labels(self):
        machine = machine_rows()
        human = blind_rows(machine)
        for row in human:
            row.update(reviewed=True, rationale="my independent judgment")
        human[0]["authorized"] = False
        human[0]["features"] = {"do_not_use": 99}
        resolved, report = resolve_reviews(machine, human, "aloysius")
        self.assertFalse(resolved[0]["authorized"])
        self.assertTrue(resolved[0]["machine_annotation"]["authorized"])
        self.assertEqual(resolved[0]["features"], machine[0]["features"])
        self.assertTrue(resolved[0]["human_verified"])
        self.assertIsNone(resolved[1]["authorized"])
        self.assertFalse(resolved[1]["human_verified"])
        self.assertEqual(
            (report["disagreements"], report["comparable_n"], report["unresolved"]), (1, 1, 1)
        )
        self.assertEqual(report["false_approval_rate_on_comparable_unsafe"], 1)
        self.assertIsNone(report["false_denial_rate_on_comparable_safe"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved.jsonl"
            write_annotations(path, resolved)
            self.assertEqual(labels_from(path), {"opaque:0": False, "opaque:1": None})

    def test_resolution_rejects_pending_missing_duplicate_and_changed_packets(self):
        machine = machine_rows()
        human = blind_rows(machine)
        with self.assertRaisesRegex(ValueError, "pending"):
            resolve_reviews(machine, human, "aloysius")
        for row in human:
            row.update(reviewed=True, authorized=True, rationale="authorized")
        with self.assertRaisesRegex(ValueError, "match exactly"):
            resolve_reviews(machine, human[:1], "aloysius")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_reviews(machine, human + human[:1], "aloysius")
        human[0]["packet"]["user_request"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            resolve_reviews(machine, human, "aloysius")
        del human[0]["packet_hash"]
        with self.assertRaisesRegex(ValueError, "packets differ"):
            resolve_reviews(machine, human, "aloysius")

    def test_invalid_or_truncated_model_output_is_unknown_not_unsafe(self):
        for content, finish in (
            ('{"verdict":"AUTHORIZED","reason":"ok"}', "length"),
            ('{"verdict":"AUTHORIZED","verdict":"UNAUTHORIZED","reason":"ok"}', "stop"),
            ('{"verdict":"UNAUTHORIZED","reason":""}', "stop"),
            ('{"authorized":false,"reason":"wrong schema"}', "stop"),
            ("not JSON", "stop"),
        ):
            with self.subTest(content=content):
                result = parse_label(content, finish)
                self.assertIsNone(result["authorized"])
                self.assertEqual(result["status"], "parse_failed")
        result = parse_label('{"verdict":"UNKNOWN","reason":"missing recipient context"}', "stop")
        self.assertEqual(result["status"], "abstained")

    def test_annotation_files_reject_truncated_lines_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text('{"key":"a","authorized":true}\n{"key":')
            with self.assertRaises(ValueError):
                read_annotations(path)
            path.write_text('{"key":"a","authorized":true}\n{"key":"a","authorized":false}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_annotations(path)

    def test_weak_labels_are_train_only_never_calibration_or_strict_ground_truth(self):
        rows = machine_rows()
        train = script_module("train_gate")
        calibration = script_module("evaluate_reviewer")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_annotations(path, rows)
            with self.assertRaisesRegex(ValueError, "human-verified"):
                labels_from(path)
            with self.assertRaisesRegex(ValueError, "human-verified"):
                train.risk_rows(path, "validation")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_reviewer.py",
                        "--annotations",
                        str(path),
                        "--model",
                        "qwen3.8:27b",
                        "--output",
                        tmp,
                    ],
                ),
                patch.object(calibration, "pin_ollama") as pin,
            ):
                with self.assertRaisesRegex(ValueError, "human-verified"):
                    calibration.main()
                pin.assert_not_called()
            for row in rows:
                row["split"] = "train"
            write_annotations(path, rows)
            self.assertEqual(len(train.risk_rows(path, "train")), 2)

    def test_mock_sdk_labeling_resumes_and_separates_annotation_costs(self):
        original = openai.OpenAI
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) in (2, 3):
                return httpx.Response(500, json={"error": {"message": "temporary failure"}})
            verdict = "AUTHORIZED" if len(requests) == 1 else "UNAUTHORIZED"
            finish = "length" if len(requests) == 5 else "stop"
            return httpx.Response(
                200,
                json=dict(
                    id="fixture",
                    object="chat.completion",
                    created=1,
                    model=body["model"],
                    choices=[
                        dict(
                            index=0,
                            finish_reason=finish,
                            message=dict(
                                role="assistant",
                                content=json.dumps({"verdict": verdict, "reason": "evidence"}),
                            ),
                        )
                    ],
                    usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16),
                ),
            )

        def factory(**kwargs):
            return original(
                **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test",
                    "OPENAI_BASE_URL": "http://localhost:11434/v1",
                },
            ),
            patch("approval_gate.llm.openai.OpenAI", side_effect=factory),
            patch("approval_gate.llm.time.sleep"),
        ):
            rows = packets(3)
            config = prepare_config(
                dict(model="qwen3.8:27b", policy="full_approve", reasoning_effort="none")
            )
            config["model_digests"] = {"qwen3.8:27b": "fixture-digest"}
            output = Path(tmp) / "job"
            first = label_packets(rows, output, config, progress=lambda _: None)
            self.assertEqual((first["pending"], first["errors"]), (1, 1))
            second = label_packets(rows, output, config, progress=lambda _: None)
            self.assertEqual(
                (second["pending"], second["errors"], second["parse_failures"]), (0, 0, 1)
            )
            self.assertEqual(second["inference_cost"]["annotator_calls"], 5)
            self.assertEqual(second["inference_cost"]["annotator_retries"], 1)
            self.assertEqual(second["inference_cost"]["annotator_tokens_in_known"], 36)
            result = read_annotations(output / "labels.jsonl")
            self.assertEqual([r["authorized"] for r in result], [True, False, None])
            self.assertTrue(
                all(r["label_source"] == "llm" and not r["human_verified"] for r in result)
            )
            self.assertEqual(packet_hash(result[0]), packet_hash(rows[0]))
            # All three outcomes, including parse failures, are cached without new requests.
            label_packets(rows, output, config, progress=lambda _: None)
            self.assertEqual(len(requests), 5)
            self.assertEqual(request_cost(output)["reviewer_calls"], 0)
            changed = deepcopy(rows)
            changed[0]["features"]["unauthorized"] = 0.99
            with self.assertRaisesRegex(ValueError, "changed"):
                label_packets(changed, output, config, progress=lambda _: None)
            with self.assertRaisesRegex(ValueError, "changed"):
                label_packets(rows, output, {**config, "seed": 2}, progress=lambda _: None)
            prompts = json.dumps([r["messages"] for r in requests])
            for hidden in (
                "secret-future-outcome",
                "feature-model",
                "validation/",
                "feature_signature",
            ):
                self.assertNotIn(hidden, prompts)
            self.assertTrue(
                all(r["temperature"] == 0 and r["reasoning_effort"] == "none" for r in requests)
            )

    def test_blind_and_human_cli_resume_then_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, blind, human = [
                root / name for name in ("machine.jsonl", "blind.json", "human.json")
            ]
            write_annotations(source, machine_rows())
            result = run_script(
                "prepare_blind_review", "--input", source, "--output", blind, "--expected-count", 2
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            args = ("--input", blind, "--output", human, "--annotator", "aloysius")
            result = run_script("annotate_calls", *args, answers="n\nwrong scope\nq\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([r["reviewed"] for r in read_annotations(human)], [True, False])
            result = run_script("annotate_calls", *args, answers="y\nnecessary action\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("opaque:0", result.stdout)
            output, report = root / "resolved.jsonl", root / "agreement.json"
            result = run_script(
                "resolve_annotations",
                "--machine",
                source,
                "--human",
                human,
                "--annotator",
                "aloysius",
                "--output",
                output,
                "--report",
                report,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(labels_from(output), {"opaque:0": False, "opaque:1": True})
            self.assertEqual(json.loads(report.read_text())["disagreements"], 2)
            # Output forms cannot be silently replaced, and count mismatches fail before inference.
            result = run_script(
                "prepare_blind_review", "--input", source, "--output", blind, "--expected-count", 2
            )
            self.assertNotEqual(result.returncode, 0)
            raw = root / "raw.jsonl"
            write_annotations(raw, packets())
            result = run_script(
                "label_packets",
                "--input",
                raw,
                "--output",
                root / "job",
                "--expected-count",
                3,
                "--dry-run",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "job").exists())
            result = run_script(
                "label_packets",
                "--input",
                raw,
                "--output",
                root / "job",
                "--expected-count",
                2,
                "--dry-run",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "job").exists())


if __name__ == "__main__":
    unittest.main()
