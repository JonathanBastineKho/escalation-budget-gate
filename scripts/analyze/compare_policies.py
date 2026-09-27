"""Paired task-family bootstrap differences on a predeclared finite budget grid."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from approval_gate.reporting import interval, rate, save_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True)
    p.add_argument("--reference", default="learned_risk")
    p.add_argument("--budgets", nargs="+", type=int, default=[0, 1, 3])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = json.loads(Path(a.metrics).read_text())
    groups = defaultdict(list)
    for r in rows:
        groups[(r["comparison"], "clean" if r["attack"] == "none" else "all_attacked")].append(r)
    results = []

    def key(r):
        return (r["suite"], r["task_id"], r["attack"], r["injection_task_id"], r["seed"])

    for (comparison, attack), group in groups.items():
        refs = {r["strategy"] for r in group if r["policy"] == a.reference}
        if len(refs) != 1:
            raise ValueError("reference must identify exactly one strategy per comparison")
        reference = next(iter(refs))
        by_strategy = defaultdict(dict)
        for r in group:
            by_strategy[r["strategy"]][(r["k"], key(r))] = r
        for strategy, values in by_strategy.items():
            pairs = []
            for k in a.budgets:
                for (budget, case_key), row in values.items():
                    if budget != k and not (
                        budget is None
                        and row["policy"] in ("full_approve", "melon_local", "undefended")
                    ):
                        continue
                    ref = by_strategy[reference].get((k, case_key))
                    if ref is None:
                        ref = by_strategy[reference].get((None, case_key))
                    if ref is None:
                        raise ValueError("missing matched reference episode")
                    strict = (
                        float(row["strict_success"]) - float(ref["strict_success"])
                        if row["strict_success"] is not None and ref["strict_success"] is not None
                        else None
                    )
                    pairs.append(
                        dict(
                            suite=row["suite"],
                            family=row["family"],
                            k=k,
                            delta_ad=int(row["tcr_ad"]) - int(ref["tcr_ad"]),
                            delta_strict=strict,
                        )
                    )
            if not pairs or set(r["k"] for r in pairs) != set(a.budgets):
                raise ValueError("incomplete primary budget grid")
            for k in [*a.budgets, "mean"]:
                selected = [r for r in pairs if k == "mean" or r["k"] == k]
                low, high = interval(selected, "delta_ad", True)
                strict = all(r["delta_strict"] is not None for r in selected)
                ci = interval(selected, "delta_strict", True) if strict else (None, None)
                results.append(
                    dict(
                        comparison=comparison,
                        attack=attack,
                        strategy=strategy,
                        reference=reference,
                        k=k,
                        delta_ad=rate(selected, "delta_ad", True),
                        delta_ad_ci_low=low,
                        delta_ad_ci_high=high,
                        delta_strict=rate(selected, "delta_strict", True) if strict else None,
                        delta_strict_ci_low=ci[0],
                        delta_strict_ci_high=ci[1],
                        task_families=len({r["family"] for r in selected}),
                    )
                )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_csv(out, results)
    print("Wrote", out)


if __name__ == "__main__":
    main()
