"""Label collected packets with Ollama; resume completed packets and log offline LLM costs."""

import argparse
import json
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv
from approval_gate.annotations import read_annotations, packet_hash
from approval_gate.annotation_labeler import label_packets
from approval_gate.experiments import prepare_config, pin_ollama
from approval_gate.notifications import add_notification_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="job directory, including labels.jsonl")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--server-context-length", type=int, default=16384)
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="none"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="check packets without contacting Ollama"
    )
    add_notification_options(parser)
    args = parser.parse_args()
    rows = read_annotations(args.input)
    if args.expected_count < 1 or len(rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} packets, found {len(rows)}; collect/export more"
        )
    for row in rows:
        packet_hash(row)
        if row.get("authorized") is not None or row.get("annotator") or row.get("label_source"):
            raise ValueError("use an unlabeled export as input")
    if args.dry_run:
        print(
            f"{len(rows)} packets; {args.model}; normally one local LLM request per packet, plus retries"
        )
        return
    load_dotenv()
    endpoint = os.environ.get("OPENAI_BASE_URL", "")
    if urlsplit(endpoint).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            "set OPENAI_BASE_URL to your local Ollama endpoint, e.g. http://127.0.0.1:11434/v1"
        )
    config = pin_ollama(
        prepare_config(
            dict(
                model=args.model,
                policy="full_approve",
                phase="offline_annotation",
                seed=args.seed,
                max_tokens=args.max_tokens,
                server_context_length=args.server_context_length,
                reasoning_effort=args.reasoning_effort,
            )
        )
    )
    metrics = label_packets(rows, args.output, config, notify=args.notify)
    print(json.dumps(metrics, indent=2))
    if metrics["pending"] or metrics["errors"]:
        raise SystemExit("Annotation job incomplete; rerun the same command to resume")


if __name__ == "__main__":
    main()
