"""Report metrics from saved case and decision logs. Makes no LLM calls.

For every (source, suite, policy, attack, k) cell it computes:
  TCR and ASR       TCR = utility and not security (security=True means the attack succeeded)
  HITL load         reviewer calls per case, on successful cases, and the share of cases with none
  cost              mean tokens per case by component (agent, policy components, reviewer)
  escalation value  reviewer approvals among reviews (work recovered: without budget the call is
                    denied) and TCR gained over k=0 per review

A source is a results folder holding cases/ and decisions/. Each case keeps its latest record.
If a cell has several files (a rerun under a new config hash) the one with most cases is used.
Attacked cells are also pooled into attack="all" (successes over all attacked cases).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

K_ORDER = ["0", "1", "2", "3", "5", "8", "inf"]


def latest_cases(path: Path) -> dict:
    cases = {}
    for line in path.open():
        record = json.loads(line)
        cases[(record["task_id"], record.get("injection_task_id"))] = record
    return cases


def collect(sources: list[Path]) -> dict:
    cells = {}
    for source in sources:
        for path in sorted((source / "cases").glob("*.jsonl")):
            cases = latest_cases(path)
            if not cases:
                continue
            first = next(iter(cases.values()))
            k = "inf" if first.get("k_budget") is None else str(first["k_budget"])
            key = (source.name, first["suite"], first["policy"], first.get("attack_name") or "none", k)
            if key not in cells or len(cases) > len(cells[key]["cases"]):
                cells[key] = {"cases": cases, "decisions": source / "decisions" / path.name}
    return cells


def reviewer_verdicts(path: Path, run_ids: set) -> Counter:
    verdicts = Counter()
    if path.exists():
        for line in path.open():
            record = json.loads(line)
            if record["run_id"] in run_ids and record.get("decided_by_node") == "reviewer":
                verdicts[record.get("reviewer_verdict")] += 1
    return verdicts


def summarize(cases: list, verdicts: Counter) -> dict:
    n = len(cases)
    safe = [c for c in cases if c["utility"] and not c["security"]]
    reviews = [c.get("escalations") or 0 for c in cases]
    tokens_in, tokens_out = Counter(), Counter()
    for c in cases:
        tokens_in.update(c.get("tokens_in") or {})
        tokens_out.update(c.get("tokens_out") or {})
    components = sorted(set(tokens_in) - {"agent", "reviewer"})
    reviewed = verdicts["APPROVE"] + verdicts["DENY"]
    return {
        "n": n,
        "tcr": round(len(safe) / n * 100, 1),
        "utility": round(sum(bool(c["utility"]) for c in cases) / n * 100, 1),
        "asr": round(sum(bool(c["security"]) for c in cases) / n * 100, 1),
        "errors": sum(bool(c.get("error")) for c in cases),
        "reviews": sum(reviews),
        "reviews_per_case": round(sum(reviews) / n, 3),
        "reviews_on_successes": sum(c.get("escalations") or 0 for c in safe),
        "zero_review_pct": round(sum(r == 0 for r in reviews) / n * 100, 1),
        "reviewer_approved": verdicts["APPROVE"],
        "reviewer_denied": verdicts["DENY"],
        "approved_share": round(verdicts["APPROVE"] / reviewed * 100, 1) if reviewed else "",
        "agent_in": round(tokens_in["agent"] / n),
        "agent_out": round(tokens_out["agent"] / n),
        "policy_in": round(sum(tokens_in[x] for x in components) / n),
        "policy_out": round(sum(tokens_out[x] for x in components) / n),
        "policy_components": "; ".join(
            f"{x} {tokens_in[x] / n:.0f}/{tokens_out[x] / n:.0f}" for x in components),
        "reviewer_in": round(tokens_in["reviewer"] / n),
        "reviewer_out": round(tokens_out["reviewer"] / n),
    }


def build_rows(cells: dict) -> list[dict]:
    rows, pooled = [], defaultdict(lambda: {"cases": [], "verdicts": Counter(), "attacks": 0})
    for (source, suite, policy, attack, k), cell in cells.items():
        cases = list(cell["cases"].values())
        verdicts = reviewer_verdicts(cell["decisions"], {c["run_id"] for c in cases})
        rows.append({"source": source, "suite": suite, "policy": policy, "attack": attack, "k": k,
                     **summarize(cases, verdicts)})
        if attack != "none":
            group = pooled[(source, suite, policy, k)]
            group["cases"] += cases
            group["verdicts"] += verdicts
            group["attacks"] += 1
    for (source, suite, policy, k), group in pooled.items():
        if group["attacks"] > 1:
            rows.append({"source": source, "suite": suite, "policy": policy, "attack": "all", "k": k,
                         **summarize(group["cases"], group["verdicts"])})
    # A cell is complete when it has as many cases as the largest cell of its kind.
    full = defaultdict(int)
    for r in rows:
        full[(r["suite"], r["attack"])] = max(full[(r["suite"], r["attack"])], r["n"])
    base = {(r["source"], r["suite"], r["policy"], r["attack"]): r["tcr"] for r in rows if r["k"] == "0"}
    for r in rows:
        r["complete"] = r["n"] == full[(r["suite"], r["attack"])]
        k0 = base.get((r["source"], r["suite"], r["policy"], r["attack"]))
        r["tcr_gain_vs_k0"] = "" if k0 is None else round(r["tcr"] - k0, 1)
        r["gain_per_review"] = (round((r["tcr"] - k0) / r["reviews_per_case"], 1)
                                if k0 is not None and r["reviews_per_case"] else "")
    order = {"none": 0, "all": 1}
    rows.sort(key=lambda r: (r["suite"], r["source"], r["policy"], order.get(r["attack"], 2), r["attack"],
                             K_ORDER.index(r["k"]) if r["k"] in K_ORDER else 99))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path, help="results folders, e.g. results/v2 results/v3")
    parser.add_argument("--output", type=Path, default=Path("results/tables/report_metrics.csv"))
    parser.add_argument("--k", default="3", help="budget printed in the console summary")
    args = parser.parse_args()

    rows = build_rows(collect(args.sources))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        fields = ["source", "suite", "policy", "attack", "k", "n", "complete", "tcr", "utility", "asr",
                  "errors", "reviews", "reviews_per_case", "reviews_on_successes", "zero_review_pct",
                  "reviewer_approved", "reviewer_denied", "approved_share", "tcr_gain_vs_k0",
                  "gain_per_review", "agent_in", "agent_out", "policy_in", "policy_out",
                  "policy_components", "reviewer_in", "reviewer_out"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}\n")
    print(f"{'source':8} {'suite':10} {'policy':24} {'attack':31} {'n':>5}  TCR    reviews/case  approved%  policy tok in")
    for r in rows:
        if r["k"] == args.k:
            flag = "" if r["complete"] else " (incomplete)"
            print(f"{r['source']:8} {r['suite']:10} {r['policy']:24} {r['attack']:31} {r['n']:5}  {r['tcr']:5}  "
                  f"{r['reviews_per_case']:12}  {r['approved_share']!s:9}  {r['policy_in']}{flag}")


if __name__ == "__main__":
    main()
