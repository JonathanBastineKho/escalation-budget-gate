"""Paired branch outcomes by tool tier; observational executed traces alone are insufficient."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from approval_gate.evaluation import selected_episodes, labels_from, authorization
from approval_gate.reporting import save_csv
from approval_gate.telemetry.events import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--annotations")
    p.add_argument("--target", choices=["agentdojo", "strict"], default="agentdojo")
    a = p.parse_args()
    labels = labels_from(a.annotations)
    states = defaultdict(dict)
    for directory, outcome in selected_episodes(a.runs):
        example = directory / "training_example.json"
        if not example.exists() or outcome["status"] == "error":
            continue
        row = json.loads(example.read_text())
        records = read_jsonl(directory / "decisions.jsonl")
        if not records:
            continue
        target = (
            outcome["tcr_ad"]
            if a.target == "agentdojo"
            else authorization(directory, outcome, labels)["strict_success"]
        )
        if target is not None:
            states[row["snapshot_id"]][row["action"]] = (float(target), records[0]["tool_tier"])
    grouped = defaultdict(list)
    for state in states.values():
        if not all(k in state for k in ("AUTO_APPROVE", "AUTO_DENY", "ESCALATE")):
            continue
        approve, deny, review = [state[k][0] for k in ("AUTO_APPROVE", "AUTO_DENY", "ESCALATE")]
        grouped[state["ESCALATE"][1]].append(
            (review - approve, review - deny, review - max(approve, deny))
        )
    rows = [
        dict(
            tool_tier=t,
            target=a.target,
            paired_states=len(v),
            review_minus_approve=sum(x[0] for x in v) / len(v),
            review_minus_deny=sum(x[1] for x in v) / len(v),
            review_minus_best_observed_branch=sum(x[2] for x in v) / len(v),
        )
        for t, v in grouped.items()
    ]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_csv(out, rows)
    print(
        f"{len(states)} states; {sum(len(v) for v in grouped.values())} complete triplets. "
        "Single branch samples are noisy; best-observed-branch comparison is optimistic for the alternatives."
    )


if __name__ == "__main__":
    main()
