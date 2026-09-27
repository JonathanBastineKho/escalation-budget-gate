"""Create task-family split candidates; human review is recorded explicitly."""

import argparse
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from agentdojo.task_suite.load_suites import get_suite
from approval_gate.experiments import VERSION
from approval_gate.splits import validate_splits
from approval_gate.telemetry.events import fingerprint, write_json


def normalize(text):
    return re.sub(r"\d+", "#", text.lower())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="experiments/splits.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confirm-reviewed", action="store_true")
    args = p.parse_args()
    path = Path(args.output)
    if args.confirm_reviewed:
        data = json.loads(path.read_text())
        validate_splits(data, VERSION)
        data["reviewed"] = True
        data["hash"] = fingerprint(data["tasks"])
        write_json(path, data)
        print(f"Recorded your family/split review in {path}")
        return
    if path.exists():
        raise SystemExit("Split file exists; edit it intentionally or choose a new output")
    rows = []
    rng = random.Random(args.seed)
    for name in ("banking", "workspace", "travel", "slack"):
        suite = get_suite(VERSION, name)
        tasks = list(suite.user_tasks.values())
        parent = list(range(len(tasks)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for i, a in enumerate(tasks):
            for j in range(i):
                if (
                    SequenceMatcher(None, normalize(a.PROMPT), normalize(tasks[j].PROMPT)).ratio()
                    >= 0.78
                ):
                    parent[find(i)] = find(j)
        groups = {}
        for i, task in enumerate(tasks):
            groups.setdefault(find(i), []).append(task)
        groups = list(groups.values())
        rng.shuffle(groups)
        counts = dict(train=0, validation=0, test=0)
        targets = dict(train=len(tasks) * 0.6, validation=len(tasks) * 0.2, test=len(tasks) * 0.2)
        for group in sorted(groups, key=len, reverse=True):
            split = max(targets, key=lambda k: targets[k] - counts[k])
            counts[split] += len(group)
            family = name + "/" + min(t.ID for t in group)
            rows += [
                dict(suite=name, task_id=t.ID, family=family, split=split, prompt=t.PROMPT)
                for t in group
            ]
    data = dict(
        benchmark_version=VERSION,
        seed=args.seed,
        reviewed=False,
        tasks=rows,
        note="Review semantic families and splits before confirming. Never move used training tasks into test.",
    )
    data["hash"] = fingerprint(rows)
    write_json(path, data)
    print(f"Wrote {path}; inspect task prompts/families/splits, then run --confirm-reviewed")


if __name__ == "__main__":
    main()
