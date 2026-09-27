"""Compare reviewers on saved decision-time contexts, never execute agent tools.

--prepare-only makes no LLM calls. Otherwise a model endpoint is required; offline
means replaying frozen traces rather than running the agent/environment again.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite
from dotenv import load_dotenv

from approval_gate.llm import build_client
from approval_gate.reviewer.replay import cases_from_run, load_cases, summarize
from approval_gate.reviewer.simulated import REVIEWER_MODES, SimulatedReviewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", type=Path, help="AgentDojo runs directory")
    source.add_argument("--cases", type=Path, help="frozen replay-case JSONL")
    parser.add_argument("--pipeline-glob", default="*")
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--benchmark-version", default="v1.2.1")
    parser.add_argument("--user-tasks", nargs="+")
    parser.add_argument("--include-reads", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--modes", choices=REVIEWER_MODES, nargs="+", default=list(REVIEWER_MODES))
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.prepare_only and not args.model:
        parser.error("--model is required unless --prepare-only")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.output.exists():
        parser.error("output already exists; use a new output name to preserve earlier results")
    cases = []
    if args.cases:
        cases = list(load_cases(args.cases))
    else:
        paths = sorted(args.runs.glob(f"{args.pipeline_glob}/{args.suite}/*/*/*.json"))
        for path in paths:
            if args.user_tasks and path.parent.parent.name not in args.user_tasks:
                continue
            cases.extend(cases_from_run(path, args.include_reads))
            if args.max_cases and len(cases) >= args.max_cases:
                break
    if args.max_cases:
        cases = cases[:args.max_cases]
    if not cases:
        parser.error("no matching tool calls found")
    suite = get_suite(args.benchmark_version, args.suite)
    schemas = {t.name: {**t.parameters.model_json_schema(), "description": t.description}
               for t in suite.tools}
    client = None
    if not args.prepare_only:
        load_dotenv()
        client = build_client(args.seed)
    reviewers = {mode: SimulatedReviewer(client, args.model or "prepare-only", mode=mode,
                                         tool_schemas=schemas) for mode in dict.fromkeys(args.modes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("x", encoding="utf-8") as output:
        for index, case in enumerate(cases):
            ctx = case.context()
            row = {"case_id": case.case_id, "source": case.source,
                   "expected_verdict": case.expected_verdict,
                   "tool_name": ctx.tool_call.function, "tool_args": dict(ctx.tool_call.args),
                   "model": args.model, "seed": args.seed}
            if args.prepare_only:
                row["requests"] = {m: r.build_messages(ctx) for m, r in reviewers.items()}
            else:
                # Alternate order to reduce consistent ordering effects in paired replay.
                modes = list(reviewers)
                if index % 2:
                    modes.reverse()
                row["reviews"] = {m: asdict(reviewers[m].review(ctx)) for m in modes}
                rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
    if args.prepare_only:
        print(f"Prepared {len(cases)} cases; no model calls or tool executions. {args.output}")
    else:
        summary = {"evaluation": "fixed-trace reviewer replay; NOT end-to-end TCR",
                   "model": args.model, "seed": args.seed, **summarize(rows)}
        summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
