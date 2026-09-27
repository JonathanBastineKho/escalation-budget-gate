"""Exercise the documented training commands and manifest expansion without inference."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from approval_gate.nodes.llm_features import FEATURES
from approval_gate.experiments import prepare_config, feature_extractor, validate_checkpoint_cases
from approval_gate.splits import validate_splits
from approval_gate.telemetry.events import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def features(value):
    return {name: value if name == "unauthorized" else 0.0 for name in FEATURES}


class TrainingWorkflowTests(unittest.TestCase):
    def test_checkpoint_preflight_rejects_signature_drift_and_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk.json"
            config = prepare_config(dict(model="fixture", policy="full_approve"))
            checkpoint = dict(
                kind="learned_risk",
                features=FEATURES,
                ablation="none",
                policy_model="fixture",
                training_families=["train/family"],
                validation_families=["validation/family"],
                feature_signature=feature_extractor(config).signature,
            )
            write_json(path, checkpoint)
            config = prepare_config({**config, "policy": "learned_risk", "checkpoint": str(path)})
            validate_checkpoint_cases(config, [{"split": "test", "family": "test/family"}])
            with self.assertRaisesRegex(ValueError, "test task family"):
                validate_checkpoint_cases(config, [{"split": "test", "family": "train/family"}])
            with self.assertRaisesRegex(ValueError, "feature prompt/model digest"):
                validate_checkpoint_cases({**config, "max_tokens": 64}, [])

    def test_risk_and_value_cli_and_leakage_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation"):
                rows = [
                    dict(
                        key=f"{split}:{index}",
                        split=split,
                        family=f"{split}/family-{index}",
                        features=features(index % 2),
                        authorized=not bool(index % 2),
                        policy_model="fixture",
                        feature_signature="fixture-signature",
                        label_source="llm" if split == "train" else "human",
                        human_verified=split == "validation",
                        reviewed=split == "validation",
                        annotator="qwen3.8:27b" if split == "train" else "fixture-human",
                        machine_annotation={"model": "qwen3.8:27b"},
                    )
                    for index in range(6)
                ]
                (root / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            risk = root / "risk.json"
            args = [
                "--kind",
                "risk",
                "--train",
                root / "train.jsonl",
                "--validation",
                root / "validation.jsonl",
                "--output",
                risk,
            ]
            result = run_script("train_gate.py", *args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = json.loads(risk.read_text())
            self.assertEqual(checkpoint["target"], "authorization")
            self.assertIn("calibrator", checkpoint)
            self.assertEqual(checkpoint["validation"]["risk"]["train_n"], 6)
            self.assertTrue(checkpoint["annotation_provenance"]["training_uses_weak_labels"])
            self.assertEqual(
                checkpoint["annotation_provenance"]["validation"]["sources"], {"human": 6}
            )
            # Immutable checkpoint protection occurs before fitting.
            result = run_script("train_gate.py", *args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("immutable", result.stderr)

            for split in ("train", "validation"):
                for action in ("AUTO_APPROVE", "AUTO_DENY", "ESCALATE"):
                    for index in range(4):
                        case_dir = root / split / "config" / "cases" / f"{action}-{index}"
                        directory = case_dir / "episode"
                        write_json(directory / "outcome.json", {"utility": bool(index % 2)})
                        write_json(case_dir / "selected.json", {"attempt": "episode"})
                        write_json(
                            directory / "training_example.json",
                            dict(
                                split=split,
                                family=f"{split}/family-{index}",
                                phase="training" if split == "train" else "diagnostic",
                                action=action,
                                features=features(index % 2),
                                target_ad=index % 2,
                                status="completed",
                                parse_failed=False,
                                policy_model="fixture",
                                feature_signature="fixture-signature",
                            ),
                        )
            voi = root / "voi.json"
            result = run_script(
                "train_gate.py",
                "--kind",
                "voi",
                "--target",
                "agentdojo",
                "--train",
                root / "train",
                "--validation",
                root / "validation",
                "--risk-checkpoint",
                risk,
                "--output",
                voi,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = json.loads(voi.read_text())
            self.assertEqual(checkpoint["target"], "agentdojo")
            self.assertEqual(set(checkpoint["heads"]), {"AUTO_APPROVE", "AUTO_DENY", "ESCALATE"})
            self.assertEqual(checkpoint["risk_guard"]["kind"], "learned_risk")

            rows = read_jsonl(root / "validation.jsonl")
            rows[0]["family"] = "train/family-0"
            (root / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            args[-1] = root / "leaky.json"
            result = run_script("train_gate.py", *args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("task-family leakage", result.stderr)
            self.assertFalse((root / "leaky.json").exists())


class ManifestTests(unittest.TestCase):
    def test_every_supplied_matrix_expands_without_inference(self):
        runner = script_module("run_experiments")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / "splits.json"
            result = run_script("create_splits.py", "--output", split_path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Fixture acknowledgement only. The user's real manifest remains untouched.
            result = run_script("create_splits.py", "--output", split_path, "--confirm-reviewed")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            splits = json.loads(split_path.read_text())
            self.assertEqual(splits["hash"], fingerprint(splits["tasks"]))
            with self.assertRaisesRegex(ValueError, "duplicate task"):
                validate_splits(
                    {**splits, "tasks": [*splits["tasks"], splits["tasks"][0]]}, "v1.2.1"
                )
            checkpoint = root / "checkpoint.json"
            write_json(checkpoint, {"fixture": True})
            case = next(r for r in splits["tasks"] if r["split"] == "test")
            bank = root / "bank.json"
            write_json(bank, {"cases": [{**case, "attack": "adaptive", "injection_task_id": None}]})
            for path in sorted((ROOT / "experiments").glob("*.yaml")):
                with self.subTest(matrix=path.name):
                    manifest = yaml.safe_load(path.read_text())
                    manifest["splits"] = str(split_path)
                    for strategy in manifest["strategies"]:
                        if "checkpoint" in strategy:
                            strategy["checkpoint"] = str(checkpoint)
                    if "case_manifest" in manifest:
                        manifest["case_manifest"] = str(bank)
                    configs, cases = runner.expand(manifest)
                    self.assertTrue(configs)
                    self.assertTrue(cases)
                    self.assertTrue(all(c["split"] == manifest["split"] for c in cases))
                    self.assertEqual(len(cases), len({fingerprint(c) for c in cases}))
                    self.assertTrue(all(c["reasoning_effort"] == "none" for c in configs))
                    if path.name == "pilot.yaml":
                        self.assertEqual(len(configs) * len(cases), 144)

    def test_incomplete_final_ledger_line_warns_but_middle_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"event":"started"}\n{"event":')
            with self.assertWarnsRegex(UserWarning, "accounting may be a lower bound"):
                self.assertEqual(read_jsonl(path), [{"event": "started"}])
            path.write_text('{"event":\n{"event":"completed"}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
