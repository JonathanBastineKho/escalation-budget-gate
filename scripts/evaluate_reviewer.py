"""Assess the frozen reviewer against offline annotations; label fields never enter its packet."""

import argparse
import dataclasses
import json
from pathlib import Path
from dotenv import load_dotenv
from approval_gate.annotations import read_annotations, require_human, label_summary
from approval_gate.experiments import prepare_config, pin_ollama
from approval_gate.llm import build_client
from approval_gate.reviewer.simulated import SimulatedReviewer
from approval_gate.reporting import request_cost
from approval_gate.notifications import ProgressNotifier, add_notification_options
from approval_gate.telemetry.events import EventLog, write_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--server-context-length", type=int, default=16384)
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    p.add_argument("--style", choices=["sanitized", "raw"], default="sanitized")
    add_notification_options(p)
    args = p.parse_args()
    annotations = read_annotations(args.annotations)
    for row in annotations:
        require_human(row, "reviewer calibration")
    rows = [r for r in annotations if type(r.get("authorized")) is bool]
    if not rows:
        raise ValueError("No human-verified packets")
    load_dotenv()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Use a new output directory to keep calibration runs distinct")
    config = pin_ollama(
        prepare_config(
            dict(
                model=args.model,
                policy="full_escalate",
                max_tokens=args.max_tokens,
                server_context_length=args.server_context_length,
                reasoning_effort=args.reasoning_effort,
                reviewer_style=args.style,
                phase="reviewer_calibration",
            )
        )
    )
    config["annotation_hash"] = fingerprint(rows)
    config["annotation_provenance"] = label_summary(annotations)
    log = EventLog(output, {"phase": "reviewer_calibration"})
    write_json(output / "manifest.json", config)
    client = build_client(
        role="reviewer", log=log, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort
    )
    reviewer = SimulatedReviewer(client, args.model, args.style)
    results = []
    with ProgressNotifier(
        "Reviewer calibration",
        len(rows),
        estimated_seconds_per_item=20,
        enabled=args.notify,
        output=output,
    ) as progress:
        try:
            for index, row in enumerate(rows, 1):
                result = reviewer.review_packet(row["packet"])
                out = {
                    "key": row["key"],
                    "authorized": row["authorized"],
                    **dataclasses.asdict(result),
                }
                log.emit("verdicts", **out)
                results.append(out)
                progress.update(index)
        finally:
            client.close()
        safe = [r for r in results if r["authorized"]]
        unsafe = [r for r in results if not r["authorized"]]
        metrics = dict(
            n=len(results),
            safe_n=len(safe),
            unsafe_n=len(unsafe),
            false_approval_rate=sum(r["verdict"] == "APPROVE" for r in unsafe) / len(unsafe)
            if unsafe
            else None,
            false_denial_rate=sum(r["verdict"] == "DENY" for r in safe) / len(safe)
            if safe
            else None,
            parse_failure_rate=sum(r["parse_failed"] for r in results) / len(results),
        )
        metrics["inference_cost"] = request_cost(output)
        write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
