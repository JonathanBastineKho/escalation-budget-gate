"""Run isolated approve/deny/review continuations from training snapshots."""

import argparse
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from approval_gate.notifications import ProgressNotifier, add_notification_options, cached_episode
from approval_gate.experiments import (
    prepare_config,
    pin_ollama,
    run_episode,
    validate_checkpoint_cases,
)
from approval_gate.telemetry.events import fingerprint, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", required=True, help="training results root")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--continuation-policy",
        default="learned_risk",
        choices=["cascade", "llm_judge", "learned_risk", "learned_voi"],
    )
    p.add_argument("--checkpoint")
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow validation/test states; cannot train on these",
    )
    p.add_argument("--dry-run", action="store_true")
    add_notification_options(p)
    args = p.parse_args()
    load_dotenv()
    paths = sorted(Path(args.snapshots).glob("*/cases/*/*/snapshots/*.json"))
    # Only selected source attempts; obsolete reruns must not multiply samples.
    paths = [
        p
        for p in paths
        if (p.parents[2] / "selected.json").exists()
        and json.loads((p.parents[2] / "selected.json").read_text())["attempt"] == p.parents[1].name
    ]
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No selected snapshots found")
    print(f"{len(paths)} states, up to {len(paths) * 3} continuations")
    if args.dry_run:
        return
    work = []
    for path in paths:
        snap = json.loads(path.read_text())
        if snap["case"]["split"] != "train" and not args.diagnostic:
            raise ValueError("training rollouts must originate in train split")
        for action in ("AUTO_APPROVE", "AUTO_DENY", "ESCALATE"):
            if action == "ESCALATE" and snap["budget"] == 0:
                continue
            config = {
                **snap["config"],
                "policy": args.continuation_policy,
                "checkpoint": args.checkpoint,
                "capture_snapshots": False,
                "phase": "diagnostic" if args.diagnostic else "training",
                "source_snapshot": fingerprint(snap),
                "forced_action": action,
            }
            config = pin_ollama(prepare_config(config))
            case = {**snap["case"], "branch_action": action, "snapshot_id": fingerprint(snap)}
            validate_checkpoint_cases(config, [case])
            work.append((config, case, snap, cached_episode(args.output, config, case)))
    completed, failed = sum(cached for _, _, _, cached in work), 0
    with ProgressNotifier(
        Path(args.output).name,
        len(work),
        completed=completed,
        enabled=args.notify,
        output=args.output,
    ) as progress:
        for config, case, snap, cached in work:
            action = case["branch_action"]
            out = run_episode(config, case, args.output, snapshot=snap, action=action)
            folder = (
                Path(args.output)
                / out["experiment_id"]
                / "cases"
                / out["case_id"]
                / out["episode_id"]
            )
            write_json(
                folder / "training_example.json",
                {
                    "family": case["family"],
                    "split": case["split"],
                    "snapshot_id": fingerprint(snap),
                    "action": action,
                    "features": snap["policy_details"].get("features", {}),
                    "parse_failed": snap["policy_details"].get("parse_failed", False),
                    "target_ad": int(out["tcr_ad"]),
                    "status": out["status"],
                    "phase": config["phase"],
                    "episode_id": out["episode_id"],
                    "policy_model": config["policy_model"],
                    "feature_signature": snap["policy_details"].get("feature_signature"),
                },
            )
            print(case["suite"], case["task_id"], action, out["status"], out["tcr_ad"], flush=True)
            failed += out["status"] == "error"
            completed += not cached
            progress.update(completed, failed=failed)
    if failed:
        raise SystemExit(f"{failed} branch errors; costs retained")


if __name__ == "__main__":
    main()
