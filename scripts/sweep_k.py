"""Run matched policies, budgets, seeds, suites and attack conditions.

A judge preflight runs before any benchmark cell. Infrastructure failures stop the
sweep, preserving partial logs. Budget-independent controls run once at k=inf.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.task_suite.load_suites import get_suite

import approval_gate.attacks  # noqa: F401 (register preserving variants)

from approval_gate.judge_config import add_judge_arguments, judge_cli_args, resolve_judge_config

K_INDEPENDENT = {"full_approve", "undefended"}
JUDGES = {"direct_judge", "sandbox_judge", "sandbox_effects_judge"}
MATCHED = JUDGES | {"full_approve", "full_escalate"}
RUNNER = Path(__file__).with_name("run_benchmark.py")


def commands(args):
    cells = []
    suites = args.suites or [args.suite]
    attacks = args.attacks or [args.attack]
    for suite, attack, policy in itertools.product(suites, attacks, args.policies):
        for k, seed in itertools.product(["inf"] if policy in K_INDEPENDENT else args.ks, args.seeds):
            cmd = [sys.executable, str(RUNNER), "--model", args.model,
                   "--suite", suite, "--policy", policy, "--k", k, "--seed", str(seed),
                   "--force-rerun", "--logdir", args.logdir]
            if policy in MATCHED:
                cmd += judge_cli_args(args)
            if args.reviewer_model:
                cmd += ["--reviewer-model", args.reviewer_model]
            if attack and attack != "none":
                cmd += ["--attack", attack]
                if args.injection_tasks:
                    cmd += ["--injection-tasks", *args.injection_tasks]
            if args.user_tasks:
                cmd += ["--user-tasks", *args.user_tasks]
            cells.append(cmd)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reviewer-model", default=None)
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--suites", nargs="+", help="multiple suites; overrides --suite")
    parser.add_argument("--policies", nargs="+", default=["full_approve", "full_escalate"],
                        choices=[*sorted(MATCHED), "undefended", "cascade", "cascade_n1", "static_threshold"])
    parser.add_argument("--ks", nargs="+", default=["0", "1", "2", "3", "5", "8", "inf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--attack", default="important_instructions_generic")
    parser.add_argument("--attacks", nargs="+", help="multiple conditions, including none; overrides --attack")
    parser.add_argument("--user-tasks", nargs="*", default=None)
    parser.add_argument("--injection-tasks", nargs="*", default=None)
    parser.add_argument("--logdir", default="results/runs")
    parser.add_argument("--skip-preflight", action="store_true", help="reuse a separately verified preflight")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    add_judge_arguments(parser)
    args = parser.parse_args()
    load_dotenv()
    try:
        args.ks = ["inf" if k == "inf" else str(int(k)) for k in args.ks]
        if any(k != "inf" and int(k) < 0 for k in args.ks):
            raise ValueError("Budgets must be nonnegative integers or inf")
        if set(args.policies) & MATCHED:
            resolve_judge_config(args)
        for values in (args.policies, args.ks, args.seeds, args.suites or [args.suite],
                       args.attacks or [args.attack], args.user_tasks, args.injection_tasks):
            if values is not None and (not values or len(values) != len(set(values))):
                raise ValueError("Empty or duplicate selections would produce invalid comparisons")
        for attack in args.attacks or [args.attack]:
            if attack != "none" and attack not in ATTACKS:
                raise ValueError(f"Unknown attack: {attack}")
        for name in args.suites or [args.suite]:
            suite = get_suite("v1.2.1", name)
            for selected, available in ((args.user_tasks, suite.user_tasks),
                                        (args.injection_tasks, suite.injection_tasks)):
                if selected and set(selected) - set(available):
                    raise ValueError(f"Task selection does not exist in suite {name}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    cells = commands(args)
    print(f"{len(cells)} runs; clean cells omit injection IDs")
    preflight = None
    if set(args.policies) & JUDGES and not args.skip_preflight:
        preflight = [sys.executable, str(Path(__file__).with_name("preflight_judge.py")),
                     "--model", args.model, "--seed", str(args.seeds[0]),
                     "--output", str(Path(args.logdir).parent / "preflight.json"),
                     "--policies", *sorted(set(args.policies) & JUDGES),
                     *judge_cli_args(args)]
        if args.reviewer_model:
            preflight += ["--reviewer-model", args.reviewer_model]
        print("Preflight: " + " ".join(preflight[2:]))
    for i, cmd in enumerate(cells, 1):
        print(f"  [{i:>2}] {' '.join(cmd[2:])}")
    if args.dry_run:
        return
    if preflight and subprocess.run(preflight, check=False).returncode != 0:
        raise SystemExit("Judge preflight failed; no benchmark cells were started")
    for i, cmd in enumerate(cells, 1):
        print(f"\n===== run {i}/{len(cells)} =====", flush=True)
        if subprocess.run(cmd, check=False).returncode != 0:
            raise SystemExit(f"Run {i} failed; sweep stopped. Inspect logs and use a fresh output directory.")
    print(f"\ndone: {len(cells)}/{len(cells)} succeeded")


if __name__ == "__main__":
    main()
