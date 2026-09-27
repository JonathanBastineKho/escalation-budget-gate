"""Run a frozen YAML experiment matrix; --dry-run performs no inference."""

import argparse
import json
from pathlib import Path
import yaml
from dotenv import load_dotenv
from agentdojo.task_suite.load_suites import get_suite
from approval_gate.experiments import (
    VERSION,
    prepare_config,
    pin_ollama,
    run_episode,
    validate_checkpoint_cases,
)
from approval_gate.splits import validate_splits
from approval_gate.notifications import ProgressNotifier, add_notification_options, cached_episode
from approval_gate.telemetry.events import fingerprint, write_json


def expand(manifest):
    split_path = Path(manifest["splits"])
    splits = json.loads(split_path.read_text())
    validate_splits(splits, VERSION)
    if not splits.get("reviewed") or splits["hash"] != fingerprint(splits["tasks"]):
        raise ValueError("review and confirm the split manifest using create_splits.py")
    split = manifest["split"]
    selected = [r for r in splits["tasks"] if r["split"] == split]
    limit = manifest.get("tasks_per_suite")
    if limit:
        selected = [
            r
            for suite in ("banking", "workspace", "travel", "slack")
            for r in sorted(
                (r for r in selected if r["suite"] == suite),
                key=lambda r: fingerprint(r["task_id"]),
            )[:limit]
        ]
    cases = []
    for row in selected:
        suite = get_suite(VERSION, row["suite"])
        goals = list(suite.injection_tasks)
        goal = goals[int(fingerprint(row["family"]), 16) % len(goals)]
        for attack in manifest["attacks"]:
            cases.append(
                {k: row[k] for k in ("suite", "task_id", "family", "split")}
                | {"attack": attack, "injection_task_id": None if attack == "none" else goal}
            )
    configs = []
    for strategy in manifest["strategies"]:
        policy = strategy["policy"]
        for k in (
            [None]
            if policy in ("full_approve", "melon_local", "undefended")
            else manifest["budgets"]
        ):
            for seed in manifest.get("seeds", [0]):
                config = {
                    **manifest["models"],
                    **manifest.get("settings", {}),
                    **strategy,
                    "k": k,
                    "seed": seed,
                    "split_hash": splits["hash"],
                    "matrix_hash": fingerprint(manifest),
                    "split": split,
                }
                configs.append(prepare_config(config))
    if manifest.get("case_manifest"):
        bank = json.loads(Path(manifest["case_manifest"]).read_text())
        cases = bank["cases"]
        allowed = {(r["suite"], r["task_id"]): r for r in splits["tasks"] if r["split"] == split}
        for case in cases:
            source = allowed[(case["suite"], case["task_id"])]
            if case["family"] != source["family"] or case["split"] != split:
                raise ValueError("attack bank split/family mismatch")
        for config in configs:
            config["case_manifest_hash"] = fingerprint(bank)
    if not cases:
        raise ValueError("no cases selected")
    return configs, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    add_notification_options(p)
    args = p.parse_args()
    load_dotenv()
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    configs, cases = expand(manifest)
    total = len(configs) * len(cases)
    print(f"{len(cases)} cases x {len(configs)} configurations = {total} episodes")
    if args.dry_run:
        for config in configs:
            print(config["policy"], "k=", config["k"], "seed=", config["seed"])
        return
    root = Path(manifest["output"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"matrix-{fingerprint(manifest)}.json", manifest)
    configs = [pin_ollama(c) for c in configs]
    for config in configs:
        validate_checkpoint_cases(config, cases)
    write_json(
        root / f"schedule-{fingerprint(manifest)}.json",
        [
            {"experiment_id": fingerprint(c), "case_id": fingerprint(case)}
            for c in configs
            for case in cases
        ],
    )
    work = [
        (config, case, not args.force_rerun and cached_episode(root, config, case))
        for config in configs
        for case in cases
    ]
    completed, failed = sum(cached for _, _, cached in work), 0
    with ProgressNotifier(
        Path(args.manifest).stem,
        total,
        completed=completed,
        enabled=args.notify,
        output=root,
    ) as progress:
        for index, (config, case, cached) in enumerate(work, 1):
            out = run_episode(config, case, root, force=args.force_rerun)
            completed += not cached
            failed += out["status"] == "error"
            print(
                f"[{index}/{total}] {config['policy']} k={config['k']} "
                f"{case['suite']}/{case['task_id']} {case['attack']} "
                f"{out['status']} TCR_AD={int(out['tcr_ad'])}",
                flush=True,
            )
            if out["status"] == "error":
                print(out["error"], flush=True)
            progress.update(completed, failed=failed)
    if failed:
        raise SystemExit(f"{failed} errors; inspect outcome.json, correct the cause, and resume")


if __name__ == "__main__":
    main()
