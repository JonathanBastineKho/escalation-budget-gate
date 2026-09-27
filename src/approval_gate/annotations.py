"""Offline label provenance, blind review forms, and human adjudication."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from approval_gate.reviewer.json_output import loads
from approval_gate.telemetry.events import fingerprint, write_json

PACKET_FIELDS = {"user_request", "proposed_call", "tool_schema", "history"}


def read_annotations(path):
    """Unlike request ledgers, annotation datasets must never silently lose a line."""
    path = Path(path)
    content = path.read_text()
    rows = (
        loads(content)
        if path.suffix == ".json"
        else [loads(line) for line in content.splitlines() if line.strip()]
    )
    if not isinstance(rows, list):
        raise ValueError("annotation JSON must be an array")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or not row["key"]:
            raise ValueError("each annotation needs a nonempty string key")
        if row["key"] in seen:
            raise ValueError(f"duplicate annotation key: {row['key']}")
        seen.add(row["key"])
        if row.get("authorized") is not None and type(row["authorized"]) is not bool:
            raise ValueError(f"{row['key']}: authorized must be true, false, or null")
    return rows


def write_annotations(path, rows):
    path = Path(path)
    if path.suffix == ".json":
        write_json(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temp.replace(path)


def packet_hash(row):
    packet = row.get("packet")
    if not isinstance(packet, dict) or set(packet) != PACKET_FIELDS:
        raise ValueError(f"{row['key']}: expected only the four reviewer-visible packet fields")
    digest = fingerprint(packet)
    if row.get("packet_hash", digest) != digest:
        raise ValueError(f"{row['key']}: packet hash mismatch; do not edit the packet")
    return digest


def blind_rows(rows):
    """Explicit allowlist: no model verdicts, scores, reasons, traces, or outcomes."""
    return [
        dict(
            key=row["key"],
            packet_hash=packet_hash(row),
            packet=deepcopy(row["packet"]),
            authorized=None,
            reviewed=False,
            annotator=None,
            rationale="",
        )
        for row in rows
    ]


def require_human(row, purpose, *, allow_pending=False):
    if row.get("authorized") is None and allow_pending:
        return
    if (
        row.get("label_source") != "human"
        or row.get("reviewed") is not True
        or (row.get("authorized") is not None and row.get("human_verified") is not True)
        or not isinstance(row.get("annotator"), str)
        or not row["annotator"].strip()
    ):
        raise ValueError(
            f"{row['key']}: {purpose} requires human-verified labels; "
            "complete annotate_calls.py or resolve_annotations.py first"
        )


def label_summary(rows):
    return {
        "n": len(rows),
        "authorized": sum(row.get("authorized") is True for row in rows),
        "unauthorized": sum(row.get("authorized") is False for row in rows),
        "unresolved": sum(row.get("authorized") is None for row in rows),
        "sources": dict(Counter(row.get("label_source", "unspecified") for row in rows)),
        "labeler_models": sorted(
            {row["machine_annotation"]["model"] for row in rows if row.get("machine_annotation")}
        ),
    }


def resolve_reviews(machine_rows, human_rows, annotator):
    """The independent human answer is authoritative; abstentions never fall back to the LLM."""
    if not annotator.strip():
        raise ValueError("an annotator name is required")
    machine = {r["key"]: r for r in machine_rows}
    human = {r["key"]: r for r in human_rows}
    if len(machine) != len(machine_rows) or len(human) != len(human_rows):
        raise ValueError("duplicate annotation keys")
    if set(machine) != set(human):
        raise ValueError("machine and blind-review packet keys must match exactly")
    resolved, comparisons = [], []
    for key, source in machine.items():
        answer = human[key]
        if source.get("label_source") != "llm" or not source.get("machine_annotation"):
            raise ValueError(f"{key}: expected a saved LLM annotation")
        digest = packet_hash(source)
        if packet_hash(answer) != digest:
            raise ValueError(f"{key}: human and machine packets differ")
        if answer.get("reviewed") is not True:
            raise ValueError(f"{key}: human review is still pending (reviewed must be true)")
        value = answer.get("authorized")
        if value is not None and type(value) is not bool:
            raise ValueError(f"{key}: human answer must be boolean or null")
        if not isinstance(answer.get("rationale"), str) or not answer["rationale"].strip():
            raise ValueError(f"{key}: supply a short human rationale, including for abstentions")
        if answer.get("annotator") not in (None, "", annotator):
            raise ValueError(f"{key}: annotator differs from --annotator")
        # Copy collection features/provenance from the machine dataset, never from the form.
        row = deepcopy(source)
        row.update(
            authorized=value,
            label_source="human",
            human_verified=value is not None,
            reviewed=True,
            annotator=annotator,
            rationale=answer["rationale"],
            packet_hash=digest,
        )
        comparable = value is not None and source.get("authorized") is not None
        comparison = dict(
            key=key,
            machine_authorized=source.get("authorized"),
            human_authorized=value,
            agrees=value == source.get("authorized") if comparable else None,
            human_rationale=answer["rationale"],
        )
        row["adjudication"] = comparison
        comparisons.append(comparison)
        resolved.append(row)
    paired = [r for r in comparisons if r["agrees"] is not None]
    safe = [r for r in paired if r["human_authorized"]]
    unsafe = [r for r in paired if not r["human_authorized"]]
    report = {
        **label_summary(resolved),
        "machine_dataset_hash": fingerprint(machine_rows),
        "human_dataset_hash": fingerprint(human_rows),
        "comparable_n": len(paired),
        "comparable_safe_n": len(safe),
        "comparable_unsafe_n": len(unsafe),
        "false_approvals": sum(r["machine_authorized"] for r in unsafe),
        "false_denials": sum(not r["machine_authorized"] for r in safe),
        "agreements": sum(r["agrees"] for r in paired),
        "disagreements": sum(not r["agrees"] for r in paired),
        "agreement_rate": sum(r["agrees"] for r in paired) / len(paired) if paired else None,
        "machine_unresolved": sum(r.get("authorized") is None for r in machine_rows),
        "false_approval_rate_on_comparable_unsafe": (
            sum(r["machine_authorized"] for r in unsafe) / len(unsafe) if unsafe else None
        ),
        "false_denial_rate_on_comparable_safe": (
            sum(not r["machine_authorized"] for r in safe) / len(safe) if safe else None
        ),
        "comparisons": comparisons,
    }
    return resolved, report
