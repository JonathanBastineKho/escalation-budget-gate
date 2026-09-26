"""Runs policies x k x seeds, one run_benchmark.py subprocess each.

Policies that never escalate (full_approve, undefended) run once at k=inf,
since k can't change their result.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

K_INDEPENDENT = {"full_approve", "undefended"}
RUNNER = Path(__file__).with_name("run_benchmark.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate-model", default=None)
    parser.add_argument("--policy-model", default=None)
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--policies", nargs="+", default=["full_approve", "full_escalate"])
    parser.add_argument("--ks", nargs="+", default=["0", "1", "2", "3", "5", "8", "inf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--attack", default="important_instructions_generic")
    parser.add_argument("--user-tasks", nargs="*", default=None)
    parser.add_argument("--injection-tasks", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args()

    cells: list[list[str]] = []
    for policy in args.policies:
        ks = ["inf"] if policy in K_INDEPENDENT else args.ks
        for k in ks:
            for seed in args.seeds:
                cmd = [
                    sys.executable, str(RUNNER),
                    "--model", args.model,
                    "--suite", args.suite,
                    "--policy", policy,
                    "--k", k,
                    "--seed", str(seed),
                    "--force-rerun",
                ]
                if args.gate_model:
                    cmd += ["--gate-model", args.gate_model]
                if args.policy_model:
                    cmd += ["--policy-model", args.policy_model]
                if args.attack and args.attack != "none":
                    cmd += ["--attack", args.attack]
                if args.user_tasks:
                    cmd += ["--user-tasks", *args.user_tasks]
                if args.injection_tasks:
                    cmd += ["--injection-tasks", *args.injection_tasks]
                cells.append(cmd)

    n_user = len(args.user_tasks) if args.user_tasks else "all"
    n_inj = len(args.injection_tasks) if args.injection_tasks else "all"
    print(f"{len(cells)} runs  |  each: user tasks={n_user} x injection tasks={n_inj}")
    for i, cmd in enumerate(cells, 1):
        print(f"  [{i:>2}] {' '.join(cmd[2:])}")
    if args.dry_run:
        return

    failed = []
    for i, cmd in enumerate(cells, 1):
        print(f"\n===== run {i}/{len(cells)} =====", flush=True)
        # check=False: one failed cell is recorded and the sweep carries on.
        if subprocess.run(cmd, check=False).returncode != 0:
            failed.append(i)
    print(f"\ndone: {len(cells) - len(failed)}/{len(cells)} succeeded" + (f", failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
