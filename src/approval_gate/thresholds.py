"""Development-only selection of separate approval and denial thresholds."""

from __future__ import annotations

from itertools import product

from approval_gate.authorization_eval import summarize
from approval_gate.evidence import validate_checks
from approval_gate.judging import fingerprint, probabilities

DEFAULT_APPROVE = (.90, .95, .98, 1.0)
DEFAULT_DENY = (.80, .85, .90, .95)
CONSTRAINTS = {"max_false_approval_rate": 0.0, "max_false_denial_rate": .1, "min_coverage": .2}


def validate_evaluation(data: dict, split: str) -> list[str]:
    if (data.get("format") != 1 or data.get("purpose") != "authorization"
            or data.get("split") != split or not data.get("complete")
            or not data.get("healthy") or data.get("repeats") != 1):
        raise ValueError(f"Require a complete, healthy {split} authorization evaluation")
    policies = data.get("policies") or list(dict.fromkeys(r["policy"] for r in data["records"]))
    cases = data["case_ids"]
    if (not policies or len(set(policies)) != len(policies)
            or not cases or len(set(cases)) != len(cases)):
        raise ValueError("Duplicate or empty policy/case selection")
    observed = [(r["case_id"], r["policy"]) for r in data["records"]]
    if len(set(observed)) != len(observed) or set(observed) != set(product(cases, policies)):
        raise ValueError("Missing, duplicate or unmatched case/policy records")
    for case in cases:
        if len({r["expected"] for r in data["records"] if r["case_id"] == case}) != 1:
            raise ValueError("Paired labels disagree")
    for policy in policies:
        if {r["expected"] for r in data["records"] if r["policy"] == policy} != {
            "APPROVE", "DENY", "UNCERTAIN"
        }:
            raise ValueError("Each policy needs all three authorization classes")
    for r in data["records"]:
        if r["judge_status"] != "ok" or r["judge_usage_missing"]:
            raise ValueError("Unhealthy judgments cannot be calibrated or compared")
        probabilities(r["judge_probabilities"])
        if data.get("judge_config", {}).get("verdict_format") == "evidence":
            if validate_checks(r["judge_evidence"], r["state"]) != r["judge_allowed_labels"]:
                raise ValueError("Recorded evidence eligibility is inconsistent")
    return policies


def eligible(metrics: dict) -> bool:
    return (metrics["false_approval_rate"] <= CONSTRAINTS["max_false_approval_rate"]
            and metrics["false_denial_rate"] <= CONSTRAINTS["max_false_denial_rate"]
            and metrics["coverage"] >= CONSTRAINTS["min_coverage"])


def _best(candidates: list[dict], required: list[str]) -> dict | None:
    candidates = [c for c in candidates if all(c["eligible"][p] for p in required)]
    if not candidates:
        return None
    return max(candidates, key=lambda c: (min(c["metrics"][p]["coverage"] for p in required),
                                          c["thresholds"]["approve"], c["thresholds"]["deny"]))


def calibrate_separate(data: dict, approve_candidates=DEFAULT_APPROVE, deny_candidates=DEFAULT_DENY,
                       *, target_policy: str | None = None) -> dict:
    policies = validate_evaluation(data, "development")
    if target_policy is not None and target_policy not in policies:
        raise ValueError("Calibration target is absent from the development evaluation")
    for grid in (approve_candidates, deny_candidates):
        if not grid or any(type(t) not in (float, int) or not .5 < t <= 1 for t in grid):
            raise ValueError("Threshold candidates must be in (0.5, 1]")
    candidates = []
    for approve, deny in product(sorted(set(approve_candidates)), sorted(set(deny_candidates))):
        metrics = summarize(data["records"], approve, deny)
        candidates.append({"thresholds": {"approve": approve, "deny": deny},
                           "metrics": metrics, "eligible": {p: eligible(metrics[p]) for p in policies}})
    required = [target_policy] if target_policy else policies
    selected = _best(candidates, required)
    independent = {p: _best(candidates, [p]) for p in policies}
    return {
        "format": 2, "split": "development", "purpose": "authorization_calibration",
        "judge_identity": data["judge_identity"], "dataset_hash": data["dataset_hash"],
        "evaluation_hash": fingerprint(data), "case_ids": data["case_ids"], "policies": policies,
        "constraints": CONSTRAINTS, "selection_policies": required,
        "selection_rule": "Maximize minimum selected-policy coverage; ties favor higher cutoffs.",
        "thresholds": selected["thresholds"] if selected else None,
        "candidates": candidates,
        "independent_thresholds": {
            p: c["thresholds"] if c else None for p, c in independent.items()
        },
    }
