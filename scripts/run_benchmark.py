"""One strategy/budget/suite, with complete per-case records. No inferred safety labels."""

import argparse
import json
from dotenv import load_dotenv
from approval_gate.cli import add_model_options, budget, config_from_args
from approval_gate.notifications import ProgressNotifier, add_notification_options, cached_episode
from approval_gate.experiments import (
    POLICIES,
    cases_for,
    pin_ollama,
    prepare_config,
    run_episode,
    validate_checkpoint_cases,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_options(p)
    p.add_argument("--suite", default="banking")
    p.add_argument("--policy", choices=POLICIES, default="full_approve")
    p.add_argument("--k", type=budget, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attack", default="none")
    p.add_argument("--user-tasks", nargs="+")
    p.add_argument("--injection-tasks", nargs="+")
    p.add_argument("--logdir", default="results/experiments")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    add_notification_options(p)
    args = p.parse_args()
    load_dotenv()
    config = prepare_config(config_from_args(args))
    cases = cases_for(args.suite, args.user_tasks, args.attack, args.injection_tasks)
    if args.dry_run:
        print(json.dumps({"config": config, "cases": cases, "episodes": len(cases)}, indent=2))
        return
    config = pin_ollama(config)
    validate_checkpoint_cases(config, cases)
    work = [
        (case, not args.force_rerun and cached_episode(args.logdir, config, case)) for case in cases
    ]
    completed, failed = sum(cached for _, cached in work), 0
    with ProgressNotifier(
        f"{args.suite}/{args.policy}",
        len(cases),
        completed=completed,
        enabled=args.notify,
        output=args.logdir,
    ) as progress:
        for i, (case, cached) in enumerate(work, 1):
            out = run_episode(config, case, args.logdir, force=args.force_rerun)
            completed += not cached
            failed += out["status"] == "error"
            print(
                f"[{i}/{len(cases)}] {case['task_id']} {case['injection_task_id']} "
                f"status={out['status']} TCR_AD={int(out['tcr_ad'])} reviews={out['escalations']}",
                flush=True,
            )
            if out["status"] == "error":
                print(out["error"], flush=True)
            progress.update(completed, failed=failed)
    if failed:
        raise SystemExit(f"{failed} episodes failed; attempts and costs were retained")


if __name__ == "__main__":
    main()
