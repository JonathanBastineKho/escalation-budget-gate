"""Run the small Qwen study: 6 preflight + 30 development + 30 held-out calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from approval_gate.authorization_eval import evaluate
from approval_gate.judge_config import add_judge_arguments, build_judge, resolve_judge_config
from approval_gate.sandbox_study import (
    STUDY_POLICIES, TARGET, build_probe_report, build_report, render_probe_report, render_report,
    small_dataset, small_development_probe, small_preflight, validate_fixture_previews,
    validate_small_fixtures,
)
from approval_gate.thresholds import DEFAULT_APPROVE, DEFAULT_DENY, calibrate_separate


def save(path: Path, value) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def stop(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="validate all fixtures without inference")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--development-probe", action="store_true",
                      help="test four development effect cases only; no calibration or held-out calls")
    mode.add_argument("--preflight-only", action="store_true",
                      help="check two development fixtures across all three policies (six calls), then stop")
    parser.add_argument("--probe-policies", nargs="+", choices=STUDY_POLICIES, default=None,
                        help="probe policies; defaults to the effects policy (four calls)")
    parser.add_argument("--approve-candidates", nargs="+", type=float, default=list(DEFAULT_APPROVE))
    parser.add_argument("--deny-candidates", nargs="+", type=float, default=list(DEFAULT_DENY))
    add_judge_arguments(parser)
    parser.set_defaults(model=None, reviewer_model=None, judge_model="qwen3.8:27b",
                        judge_verdict_format="evidence", judge_timeout=180,
                        judge_reasoning_effort="none", judge_max_output_tokens=2048,
                        judge_approve_threshold=.95, judge_deny_threshold=.85)
    args = parser.parse_args()
    load_dotenv()
    try:
        if (args.judge_backend != "chat" or "qwen" not in args.judge_model.lower()
                or args.judge_verdict_format != "evidence" or args.judge_calibration):
            raise ValueError("This study uses Qwen chat with evidence verdicts and a fresh calibration")
        for grid in (args.approve_candidates, args.deny_candidates):
            if not grid or any(not .5 < t <= 1 for t in grid):
                raise ValueError("Candidate thresholds must be in (0.5, 1]")
        config = resolve_judge_config(args)
        if args.probe_policies is not None and not args.development_probe:
            raise ValueError("--probe-policies requires --development-probe")
        if args.development_probe:
            args.probe_policies = args.probe_policies or [TARGET]
            if TARGET not in args.probe_policies or len(set(args.probe_policies)) != len(args.probe_policies):
                raise ValueError("Probe requires unique policies including sandbox_effects_judge")
            probe_data = small_development_probe(config)
        elif args.preflight_only:
            preflight_data = small_preflight(small_dataset("development"))
            validate_fixture_previews(preflight_data, config)
        else:
            datasets = validate_small_fixtures(config)
            preflight_data = small_preflight(datasets["development"])
        if args.output_dir.exists():
            raise ValueError("Output directory already exists; use a fresh run directory")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.development_probe:
        print(f"Validated {len(probe_data['cases'])} development probe fixtures. "
              f"Plan: {len(probe_data['cases']) * len(args.probe_policies)} Qwen calls; "
              "no calibration or held-out evaluation.", flush=True)
    elif args.preflight_only:
        print("Validated two development preflight fixtures across three policies. "
              "Plan: 6 Qwen calls; no development evaluation, calibration or held-out calls.", flush=True)
    else:
        print("Validated 10 development + 10 held-out fixtures across three policies.", flush=True)
        print("Plan: 6 preflight + 30 development + 30 held-out Qwen calls; no acting agent or reviewer.",
              flush=True)
    if args.dry_run:
        print("Dry run complete; no inference or result files created.")
        return
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    judge = build_judge(config, args.seed)
    if args.development_probe:
        result = evaluate(probe_data, config, judge, root / "probe.json", seed=args.seed,
                          policies=args.probe_policies, rotate_order=True, purpose="development_probe")
        report = build_probe_report(result)
        save(root / "probe_report.json", report)
        with (root / "probe_report.md").open("x") as stream:
            stream.write(render_probe_report(report))
        print(f"Report: {root / 'probe_report.md'}", flush=True)
        if not report["target_passed"]:
            stop(report["conclusion"])
        print(report["conclusion"])
        return
    kwargs = dict(seed=args.seed, policies=list(STUDY_POLICIES), rotate_order=True)
    print("Preflight: schema, grounded citations, effects and usage accounting", flush=True)
    preflight = evaluate(preflight_data, config, judge,
                         root / "preflight.json", purpose="preflight", **kwargs)
    if not preflight["healthy"]:
        failed = preflight["records"][-1]
        detail = failed["judge_status"]
        diagnostic = failed["judge_diagnostics"]
        if diagnostic.get("evidence_error"):
            detail += (f"; {diagnostic['evidence_error']} in {diagnostic.get('evidence_check', 'checks')}, "
                       f"citation {diagnostic.get('evidence_citation_index', 'n/a')}")
        if failed["judge_usage_missing"]:
            detail += "; usage missing"
        stop(f"Preflight failed at {failed['case_id']} ({failed['policy']}: {detail}); "
             f"inspect {root / 'preflight.json'}. No development calls made.")
    if args.preflight_only:
        print(f"Preflight passed; results: {root / 'preflight.json'}. "
              "No development evaluation, calibration or held-out calls made.")
        return
    print("Development evaluation", flush=True)
    development = evaluate(datasets["development"], config, judge, root / "development.json", **kwargs)
    if not development["healthy"]:
        stop(f"Development inference failed; inspect {root / 'development.json'}.")
    frozen = calibrate_separate(development, args.approve_candidates, args.deny_candidates,
                                target_policy=TARGET)
    save(root / "calibration.json", frozen)
    if frozen["thresholds"] is None:
        stop("No approval/denial threshold pair qualifies for the effects policy. "
             f"Held-out evaluation was not run. Inspect {root / 'calibration.json'}; "
             "keep the constraints fixed and revise on development cases.")
    print(f"Frozen common pair: {frozen['thresholds']}. "
          "Baseline-specific development pairs are retained for the sensitivity check.", flush=True)
    args.judge_calibration = str(root / "calibration.json")
    config = resolve_judge_config(args)
    heldout = evaluate(datasets["heldout"], config, judge, root / "heldout.json", **kwargs)
    if not heldout["healthy"]:
        stop(f"Held-out inference failed; inspect {root / 'heldout.json'}. No quality claim made.")
    report = build_report(development, frozen, heldout)
    save(root / "report.json", report)
    with (root / "report.md").open("x") as stream:
        stream.write(render_report(report))
    print(report["conclusion"])
    print(f"Report: {root / 'report.md'}")


if __name__ == "__main__":
    main()
