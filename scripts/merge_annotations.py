"""Merge independently annotated packets, rejecting conflicting non-null labels."""

import argparse
from pathlib import Path
from approval_gate.annotations import (
    read_annotations,
    write_annotations,
    require_human,
    packet_hash,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise SystemExit("Choose a new annotation output")
    merged = {}
    for path in a.inputs:
        for row in read_annotations(path):
            require_human(row, "strict annotation merging", allow_pending=True)
            previous = merged.get(row["key"])
            if previous and packet_hash(previous) != packet_hash(row):
                raise ValueError("Packet mismatch for " + row["key"])
            if previous and previous.get("authorized") is not None:
                if (
                    row.get("authorized") is not None
                    and row["authorized"] != previous["authorized"]
                ):
                    raise ValueError("Conflicting labels for " + row["key"])
                continue
            merged[row["key"]] = row
    out = Path(a.output)
    write_annotations(out, list(merged.values()))
    print("Merged", len(merged), "packets")


if __name__ == "__main__":
    main()
