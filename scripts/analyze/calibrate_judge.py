"""Select one shared threshold using development cases only.

This is empirical threshold selection, not a calibrated probability guarantee.
Both policies must satisfy the error/coverage constraints independently. A judge
that only abstains cannot pass. No threshold is written when none qualifies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from approval_gate.authorization_eval import POLICIES, summarize
from approval_gate.judging import fingerprint
from approval_gate.thresholds import calibrate_separate, DEFAULT_APPROVE, DEFAULT_DENY


def constraint_failures(metrics, max_false_approval_rate, max_false_denial_rate, min_coverage):
    failures = []
    if metrics["false_approval_rate"] > max_false_approval_rate:
        failures.append("false_approvals")
    if metrics["false_denial_rate"] > max_false_denial_rate:
        failures.append("false_denials")
    if metrics["coverage"] < min_coverage:
        failures.append("coverage")
    return failures


def failure_analysis(records):
    """Explain the zero-false-approval operating limit without selecting a threshold.

    With >= comparisons, equality to the worst erroneous APPROVE score still
    approves that call. The next representable float is the first safe boundary
    on these development records only; it is not a probability guarantee.
    """
    analysis = {}
    for policy in POLICIES:
        unapproved = sorted(
            (r for r in records if r["policy"] == policy and r["expected"] != "APPROVE"),
            key=lambda r: r["judge_probabilities"]["APPROVE"], reverse=True,
        )
        worst = unapproved[0]["judge_probabilities"]["APPROVE"]
        boundary = math.nextafter(max(0.5, worst), math.inf)
        metrics = summarize(records, boundary)[policy] if boundary <= 1 else None
        analysis[policy] = {
            "max_unapproved_approve_score": worst,
            "threshold_for_zero_observed_false_approvals": boundary if boundary <= 1 else None,
            "coverage_at_that_threshold": metrics["coverage"] if metrics else None,
            "highest_scoring_unapproved_cases": [
                {"case_id": r["case_id"], "expected": r["expected"],
                 "approve_score": r["judge_probabilities"]["APPROVE"],
                 "rationale": r.get("rationale")}
                for r in unapproved[:3]
            ],
        }
    return analysis


def print_failure_report(frozen):
    print("False approvals include cases labeled DENY or UNCERTAIN.")
    print("threshold  policy          false approvals  false denials  coverage  failed constraints")
    for row in frozen["candidates"]:
        for policy, metrics in row["metrics"].items():
            failures = ", ".join(row["constraint_failures"][policy]) or "none"
            print(f"{row['threshold']:9.4f}  {policy:14}  {metrics['false_approvals']:15}  "
                  f"{metrics['false_denials']:13}  {metrics['coverage']:8.1%}  {failures}")
    for policy, diagnosis in frozen["failure_analysis"].items():
        worst = diagnosis["highest_scoring_unapproved_cases"][0]
        print(f"{policy}: highest unapproved score is {worst['case_id']} "
              f"(expected {worst['expected']}, APPROVE={worst['approve_score']:.6f}).")
        coverage = diagnosis["coverage_at_that_threshold"]
        if coverage is None:
            print("  APPROVE=1 for an unapproved call: no allowed threshold avoids that error.")
        else:
            print(f"  Eliminating all observed false approvals leaves at most {coverage:.1%} "
                  "automatic-decision coverage for this policy.")


def calibrate(data, thresholds, max_false_approval_rate=0.0,
              max_false_denial_rate=0.1, min_coverage=0.2):
    if (data.get("format") != 1 or data.get("purpose") != "authorization"
            or data.get("split") != "development" or not data.get("complete")
            or not data.get("healthy") or data.get("repeats") != 1):
        raise ValueError("Calibration requires a complete, healthy development evaluation (one repeat)")
    records = data["records"]
    expected = {(case, policy) for case in data["case_ids"] for policy in POLICIES}
    observed = [(r["case_id"], r["policy"]) for r in records]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Missing, duplicate or mismatched development cases")
    for case_id in data["case_ids"]:
        if len({r["expected"] for r in records if r["case_id"] == case_id}) != 1:
            raise ValueError("Paired development labels disagree")
    for policy in POLICIES:
        labels = {r["expected"] for r in records if r["policy"] == policy}
        if labels != {"APPROVE", "DENY", "UNCERTAIN"}:
            raise ValueError("Each policy needs benign, denied and ambiguous development cases")
    if any(r["judge_status"] != "ok" or r["judge_usage_missing"] for r in records):
        raise ValueError("Unhealthy judgments cannot be used for calibration")
    if not thresholds or any(not 0.5 < t <= 1 for t in thresholds):
        raise ValueError("Thresholds must be in (0.5, 1]")
    if any(not 0 <= v <= 1 for v in
           (max_false_approval_rate, max_false_denial_rate, min_coverage)):
        raise ValueError("Error and coverage constraints must be in [0, 1]")
    if min_coverage == 0:
        raise ValueError("min_coverage must be positive; an always-abstain judge cannot qualify")
    candidates = []
    for threshold in sorted(set(thresholds)):
        metrics = summarize(records, threshold)
        failures = {p: constraint_failures(m, max_false_approval_rate,
                                          max_false_denial_rate, min_coverage)
                    for p, m in metrics.items()}
        candidates.append({"threshold": threshold, "eligible": not any(failures.values()),
                           "metrics": metrics, "constraint_failures": failures})
    eligible = [row for row in candidates if row["eligible"]]
    # Maximize the worse arm's coverage; break ties toward the higher threshold.
    selected = max(eligible, key=lambda row: (
        min(m["coverage"] for m in row["metrics"].values()), row["threshold"]
    )) if eligible else None
    return {
        "format": 1, "split": "development", "judge_identity": data["judge_identity"],
        "dataset_hash": data["dataset_hash"], "case_ids": data["case_ids"],
        "evaluation_hash": fingerprint(data),
        "constraints": dict(max_false_approval_rate=max_false_approval_rate,
                            max_false_denial_rate=max_false_denial_rate, min_coverage=min_coverage),
        "threshold": selected["threshold"] if selected else None,
        "candidates": candidates,
        "failure_analysis": failure_analysis(records) if selected is None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.55, .6, .65, .7, .75, .8, .9, .95])
    parser.add_argument("--separate-thresholds", action="store_true")
    parser.add_argument("--approve-thresholds", nargs="+", type=float, default=list(DEFAULT_APPROVE))
    parser.add_argument("--deny-thresholds", nargs="+", type=float, default=list(DEFAULT_DENY))
    parser.add_argument("--max-false-approval-rate", type=float, default=0.0)
    parser.add_argument("--max-false-denial-rate", type=float, default=0.1)
    parser.add_argument("--min-coverage", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; do not overwrite a frozen threshold")
    try:
        data = json.loads(args.input.read_text())
        if args.separate_thresholds:
            if (args.max_false_approval_rate, args.max_false_denial_rate, args.min_coverage) != (0, .1, .2):
                raise ValueError("Separate-threshold workflow keeps the fixed error/coverage constraints")
            frozen = calibrate_separate(data, args.approve_thresholds, args.deny_thresholds)
        else:
            frozen = calibrate(data, args.thresholds,
                               args.max_false_approval_rate, args.max_false_denial_rate, args.min_coverage)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2) + "\n")
    if args.separate_thresholds:
        if frozen["thresholds"] is None:
            print(f"No shared approval/denial threshold pair qualifies; inspect {args.output}")
            raise SystemExit(2)
        print(f"Frozen shared threshold pair: {frozen['thresholds']}; {args.output}")
        return
    if frozen["threshold"] is None:
        print_failure_report(frozen)
        print(f"No tested shared threshold meets the constraints. Inspect {args.output}; "
              "validate the rubric/model on development cases before proceeding.")
        raise SystemExit(2)
    print(f"Frozen shared threshold: {frozen['threshold']}; {args.output}")


if __name__ == "__main__":
    main()
