"""Evaluate labeled call proposals without running an acting agent or reviewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from approval_gate.authorization_eval import ALL_POLICIES, POLICIES, evaluate, load_dataset, prepare_case
from approval_gate.judge_config import add_judge_arguments, build_judge, resolve_judge_config
from approval_gate.sandbox import AgentDojoPreview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reviewer-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["development", "heldout"], required=True)
    parser.add_argument("--dataset", type=Path, help="optional independently labeled fixture file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=list(ALL_POLICIES), default=list(POLICIES))
    add_judge_arguments(parser)
    args = parser.parse_args()
    load_dotenv()
    try:
        config = resolve_judge_config(args)
        dataset = load_dataset(args.split, args.dataset)
        for case in dataset["cases"]:
            runtime, env, _, _, call = prepare_case(case)
            preview = AgentDojoPreview(case["suite"], max_chars=config["preview_max_chars"],
                include_effects="sandbox_effects_judge" in args.policies).preview(
                runtime, env, call)
            if preview.status != "ok":
                raise ValueError(f"Fixture {case['id']} has an unavailable preview: {preview.status}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"Validated {len(dataset['cases'])} {args.split} fixtures; "
              f"{len(args.policies) * len(dataset['cases'])} judge calls planned, no inference performed")
        return
    if args.output.exists():
        parser.error("Output already exists; use a fresh evaluation path")
    result = evaluate(dataset, config, build_judge(config, args.seed), args.output, seed=args.seed,
                      policies=args.policies)
    print(json.dumps(result["summary"], indent=2))
    if not result["healthy"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
