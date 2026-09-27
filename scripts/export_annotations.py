"""Export reviewer-visible packets for independent offline authorization annotation."""

import argparse
import random
from pathlib import Path
from approval_gate.annotations import write_annotations
from approval_gate.evaluation import selected_episodes
from approval_gate.telemetry.events import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--require-count", action="store_true", help="fail unless --limit packets exist")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "validation", "test"])
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.require_count and args.limit is None:
        raise ValueError("--require-count requires --limit")
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Refusing to overwrite annotations; choose a new output")
    rows = []
    for directory, outcome in selected_episodes(args.runs):
        if args.split and outcome["split"] != args.split:
            continue
        invoked = {
            e["step_index"]
            for e in read_jsonl(directory / "executions.jsonl")
            if e["event"] == "started"
        }
        for d in read_jsonl(directory / "decisions.jsonl"):
            rows.append(
                dict(
                    key=f"{outcome['episode_id']}:{d['step_index']}",
                    family=outcome["family"],
                    split=outcome["split"],
                    suite=outcome["suite"],
                    task_id=outcome["task_id"],
                    trajectory=str(directory / "trace.json"),
                    packet=d["packet"],
                    features=d["features"],
                    feature_parse_failed=d["policy_details"].get("parse_failed", False),
                    policy_model=outcome["policy_model"],
                    feature_signature=d["policy_details"].get("feature_signature"),
                    executed=d["step_index"] in invoked,
                    authorized=None,
                    annotator=None,
                    rationale="",
                )
            )
    random.Random(args.seed).shuffle(rows)
    if args.require_count and len(rows) < args.limit:
        raise ValueError(
            f"Only {len(rows)} packets available; need {args.limit}. Collect more first."
        )
    if args.limit:
        rows = rows[: args.limit]
    write_annotations(output, rows)
    print(
        f"Exported {len(rows)} packets to {output}. Fill authorized=true/false; retain null if unresolved."
    )
    print(
        "The simulated reviewer's verdict is intentionally omitted to avoid anchoring annotations."
    )


if __name__ == "__main__":
    main()
