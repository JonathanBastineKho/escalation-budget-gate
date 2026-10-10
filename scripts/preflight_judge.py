"""Probe the actual production judge/rubric with direct and sandbox evidence.

Success means valid complete scores AND usage accounting, not good classification.
Each repeat makes four calls. The first infrastructure failure stops the probe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from approval_gate.authorization_eval import ALL_POLICIES, POLICIES, evaluate, preflight_dataset
from approval_gate.judge_config import add_judge_arguments, build_judge, resolve_judge_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reviewer-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=list(ALL_POLICIES), default=list(POLICIES))
    add_judge_arguments(parser)
    args = parser.parse_args()
    load_dotenv()
    try:
        config = resolve_judge_config(args)
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"Preflight: {2 * len(args.policies) * args.repeats} calls, "
              f"{config['backend']}, model={config['model']}")
        return
    if args.output.exists():
        parser.error("Output already exists; use a fresh preflight path")
    result = evaluate(preflight_dataset(), config, build_judge(config, args.seed), args.output,
                      seed=args.seed, repeats=args.repeats, purpose="preflight", policies=args.policies)
    if not result["healthy"]:
        print(f"Preflight FAILED; inspect {args.output}. No benchmark should run with these settings.")
        raise SystemExit(2)
    print(f"Preflight passed; classification quality is not validated. Report: {args.output}")


if __name__ == "__main__":
    main()
