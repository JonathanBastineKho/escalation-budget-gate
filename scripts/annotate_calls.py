"""Terminal annotation of exported call packets. No LLM requests."""

import argparse
import json
from pathlib import Path
from approval_gate.annotations import read_annotations, write_annotations, packet_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--annotator", required=True)
    a = p.parse_args()
    out = Path(a.output)
    source = read_annotations(a.input)
    rows = read_annotations(out) if out.exists() else source
    if {r["key"]: packet_hash(r) for r in source} != {r["key"]: packet_hash(r) for r in rows}:
        raise ValueError("resume file differs from the input packet set")
    if not a.annotator.strip():
        raise ValueError("provide an annotator name")
    if any(r.get("label_source") == "llm" or r.get("machine_annotation") for r in source + rows):
        raise ValueError("Use prepare_blind_review.py first to hide machine judgments")

    def save():
        write_annotations(out, rows)

    for i, row in enumerate(rows):
        if row.get("reviewed") is True:
            continue
        print(f"\n[{i + 1}/{len(rows)}] {row['key']}")
        print(json.dumps(row["packet"], indent=2, ensure_ascii=True))
        while True:
            answer = (
                input("Authorized? y / n / ? (unresolved) / q (save and exit): ").strip().lower()
            )
            if answer in ("y", "n", "?", "q"):
                break
        if answer == "q":
            save()
            return
        row["authorized"] = {"y": True, "n": False, "?": None}[answer]
        row["annotator"] = a.annotator
        row["rationale"] = ""
        while not row["rationale"].strip():
            row["rationale"] = input("Short rationale (required): ")
        row["reviewed"] = True
        row["label_source"] = "human"
        row["human_verified"] = row["authorized"] is not None
        row["packet_hash"] = packet_hash(row)
        save()
    save()
    print("Saved", out)


if __name__ == "__main__":
    main()
