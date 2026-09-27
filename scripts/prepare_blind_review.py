"""Create a JSON review form without model judgments, features, or future outcomes."""

import argparse
from pathlib import Path

from approval_gate.annotations import blind_rows, read_annotations, write_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit("Refusing to overwrite a review form; choose a new output")
    rows = read_annotations(args.input)
    if args.expected_count < 1 or len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} packets, found {len(rows)}")
    write_annotations(args.output, blind_rows(rows))
    print(f"Saved {len(rows)} blind packets to {args.output}; no inference performed")


if __name__ == "__main__":
    main()
