"""Recompute the small-study report from saved results without model inference."""

import argparse
import json
from pathlib import Path

from approval_gate.sandbox_study import build_report, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = [json.loads((args.root / name).read_text())
                for name in ("development.json", "calibration.json", "heldout.json")]
        print(render_report(build_report(*data)), end="")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
