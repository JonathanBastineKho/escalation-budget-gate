"""Join independent human decisions to original LLM labels; human answers determine final labels."""

import argparse
from pathlib import Path

from approval_gate.annotations import read_annotations, resolve_reviews, write_annotations
from approval_gate.telemetry.events import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--human", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.report).resolve():
        raise ValueError("output and report must be different files")
    if Path(args.output).exists() or Path(args.report).exists():
        raise SystemExit("Choose new output/report paths; resolved labels are immutable")
    rows, report = resolve_reviews(
        read_annotations(args.machine), read_annotations(args.human), args.annotator
    )
    write_annotations(args.output, rows)
    write_json(args.report, report)
    print(
        f"Resolved {len(rows)} reviews; {report['disagreements']} corrections, "
        f"{report['unresolved']} unresolved. Agreement report: {args.report}"
    )


if __name__ == "__main__":
    main()
