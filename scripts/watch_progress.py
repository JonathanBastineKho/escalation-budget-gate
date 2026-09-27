"""Send Telegram milestones for an existing matrix run without restarting or modifying it."""

import argparse
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from approval_gate.notifications import (
    ProgressNotifier,
    add_notification_options,
    schedule_progress,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument(
        "--schedule", help="required if the results directory contains multiple schedules"
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument(
        "--once", action="store_true", help="print progress once without sending messages"
    )
    add_notification_options(parser)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must be between 1 and 60")
    load_dotenv()
    root = Path(args.runs)
    paths = [Path(args.schedule)] if args.schedule else sorted(root.glob("schedule-*.json"))
    if len(paths) != 1:
        raise ValueError("select exactly one existing schedule with --schedule")
    schedule = json.loads(paths[0].read_text())
    if not schedule or len({(r["experiment_id"], r["case_id"]) for r in schedule}) != len(schedule):
        raise ValueError("schedule is empty or has duplicate cases")
    completed, failed = schedule_progress(root, schedule)
    print(f"{root.name}: {completed}/{len(schedule)} finished; {failed} errors", flush=True)
    if args.once:
        return
    with ProgressNotifier(
        f"{root.name} (watcher)",
        len(schedule),
        completed=completed,
        failed=failed,
        enabled=args.notify,
        output=root,
    ) as progress:
        while completed < len(schedule):
            time.sleep(args.poll_seconds)
            completed, failed = schedule_progress(root, schedule)
            progress.update(completed, failed=failed)
            print(f"{root.name}: {completed}/{len(schedule)} finished; {failed} errors", flush=True)


if __name__ == "__main__":
    main()
