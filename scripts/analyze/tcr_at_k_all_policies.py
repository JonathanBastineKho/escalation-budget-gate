"""TCR@k for every policy in the appendix table, from the CSVs in results_csv_only.

Writes two figures (clean tasks; both attacks pooled), each with one row per policy family
and one column per suite (workspace, travel, banking), plus the LaTeX body of the appendix
TCR table, all from the same numbers. Every (suite, attack, policy) cell is read from one
pinned source and must hold the suite's full case set; banking judges are three 48-case
batches per attacked cell and are summed. A cell with no source is left pending.

TCR is the AgentDojo proxy: utility and not security. Makes no LLM calls.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results_csv_only")
SUITES = ["workspace", "travel", "banking"]
CASES = {"workspace": (40, 560), "travel": (20, 140), "banking": (16, 144)}
ATTACKS = ["none", "important_instructions_generic", "tool_knowledge_generic"]
K_ORDER = ["0", "1", "3", "5", "8", "inf"]

# Table column order; a policy key is "<policy>" or "<policy>|<judge model>".
COLUMNS = ["full_approve", "full_escalate", "cascade", "cascade_n1", "cascade-llmfeat",
           "cascade-decision", "direct_judge|Q", "sandbox_judge|Q", "sandbox_effects_judge|Q",
           "direct_judge|D2", "sandbox_judge|D2", "sandbox_effects_judge|D2"]
LABEL = {"full_approve": "Full approve", "full_escalate": "Full escalate",
         "cascade": "Full cascade (ours)", "cascade_n1": "Node-1 only",
         "cascade-llmfeat": "LLM features", "cascade-decision": "Decision-2 scorer",
         "direct_judge|Q": "Direct", "sandbox_judge|Q": "Sandbox", "sandbox_effects_judge|Q": "Effects",
         "direct_judge|D2": "Direct", "sandbox_judge|D2": "Sandbox", "sandbox_effects_judge|D2": "Effects"}
# One row per family; full approve and full escalate are drawn in grey in every row as references.
FAMILIES = [("Baselines and\nfull cascade", ["cascade", "cascade_n1"]),
            ("Cascade\nvariants", ["cascade", "cascade-llmfeat", "cascade-decision"]),
            ("Qwen 27B\njudge", ["direct_judge|Q", "sandbox_judge|Q", "sandbox_effects_judge|Q"]),
            ("Decision-2\njudge", ["direct_judge|D2", "sandbox_judge|D2", "sandbox_effects_judge|D2"])]
# Same colours as the main-text figure: full cascade aqua, effects judge yellow.
COLOR = {"cascade": "#1baf7a", "cascade_n1": "#2a78d6", "cascade-llmfeat": "#2a78d6",
         "cascade-decision": "#eb6834", "direct_judge|Q": "#2a78d6", "sandbox_judge|Q": "#eb6834",
         "sandbox_effects_judge|Q": "#eda100", "direct_judge|D2": "#2a78d6",
         "sandbox_judge|D2": "#eb6834", "sandbox_effects_judge|D2": "#eda100"}
MARKER = {"cascade": "D", "sandbox_effects_judge|Q": "^", "sandbox_effects_judge|D2": "^",
          "cascade-decision": "o", "sandbox_judge|Q": "o", "sandbox_judge|D2": "o"}
REF = {"full_approve": dict(color="#898781", ls="--"), "full_escalate": dict(color="#b5b3ad", ls="-")}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def rm(path, suite, attack, policy, source=None, file_attacks=None):
    """report_metrics-style rows -> {k: (n, safe)}; tcr is percent rounded to 0.1."""
    out = {}
    for r in csv.DictReader(open(ROOT / path, newline="")):
        if r["suite"] != suite or r["policy"] != policy or r["attack"] not in (file_attacks or [attack]):
            continue
        if source is not None and not r["source"].startswith(source):
            continue
        n = int(r["n"])
        out[r["k"].removesuffix(".0")] = (n, round(float(r["tcr"]) * n / 100))
    return out


def batches(path, suite, attack, policy, prefix, exclude=()):
    """Sum disjoint batch runs (sources starting with prefix) -> {k: (n, safe)}."""
    out = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(ROOT / path, newline="")):
        if (r["suite"], r["attack"], r["policy"]) != (suite, attack, policy):
            continue
        if not r["source"].startswith(prefix) or r["source"].endswith(tuple(exclude)):
            continue
        n = int(r["n"])
        out[r["k"]][0] += n
        out[r["k"]][1] += round(float(r["tcr"]) * n / 100)
    return {k: tuple(v) for k, v in out.items()}


def jerome(suite, attack, policy, prefixes):
    out = {}
    for r in csv.DictReader(open(ROOT / "results_csv_jerome/tcr_all_results.csv", newline="")):
        if (r["suite"], r["policy"], r["attack"]) != (suite, policy, attack):
            continue
        if not any(r["source_runs"].startswith(p) for p in prefixes) or not r["n_cases"]:
            continue
        n = int(r["n_cases"])
        out[r["k"]] = (n, round(float(r["tcr_mean"]) * n))
    return out


RQ = "results_csv_rq/report_metrics_v2.csv"
ANANYA = "results_csv_ananya/report_metrics_ananya.csv"
JOJO = "results_csv_jojo/report_metrics_jojo.csv"
JOHN_D2 = "results_csv_john/report_metrics_john_decision2_workspace_s0_20261022.csv"
NOP_TS = "results_csv_nop/report_metrics_nop_ablation_travel_slack_s0_20261024.csv"
NOP_B = "results_csv_john/report_metrics_john_ablation_banking_s0_20261024_helped_by_nop.csv"
TEJA_TT = "results_csv_teja/travel_tool_knowledge.csv"
TK = ["tool_knowledge", "tool_knowledge_generic"]
TRAVEL_27B = {"none": "travel-27b-judge-clean-20261022-180004/",
              "important_instructions_generic": "travel-27b-judge-20261021-074148/",
              "tool_knowledge_generic": "travel-27b-judge-tool-knowledge-20261022-175724/"}


def source(suite, attack, col):
    """Pinned reader for one (suite, attack, column); None when no source exists."""
    policy, _, judge = col.partition("|")
    if col in ("full_approve", "full_escalate", "cascade", "cascade_n1"):
        if suite in ("workspace", "banking"):
            return lambda: rm(RQ, suite, attack, policy, source="v2")
        if attack == "none":
            return lambda: rm(ANANYA, "travel", "none", policy, source="results")
        if attack == "tool_knowledge_generic":
            return lambda: rm(TEJA_TT, "travel", attack, policy, file_attacks=TK)
        if policy == "cascade_n1":
            return lambda: rm("results_csv_jerome/report_metrics_jerome.csv", "travel", attack, policy,
                              source="results-from-nscc")
        dirs = {"full_approve": ["runs"], "full_escalate": ["full-escalate-travel-20261019-160220/"],
                "cascade": ["v3/", "cascade-travel-remaining-20261019-082222/"]}[policy]
        return lambda: jerome("travel", attack, policy, dirs)
    if policy in ("cascade-llmfeat", "cascade-decision"):
        path = {"workspace": JOJO, "travel": NOP_TS, "banking": NOP_B}[suite]
        src = "v3" if suite == "workspace" else "nop-"
        return lambda: rm(path, suite, attack, policy, source=src)
    if judge == "D2":
        if suite == "workspace":
            return lambda: rm(JOHN_D2, suite, attack, policy)
        if suite == "travel":
            return lambda: rm(RQ, suite, attack, policy, source="v2")
        return lambda: batches(ANANYA, suite, attack, policy, "Decision-2.0-Lux-9B")
    # Qwen 27B
    if suite == "workspace":
        if policy == "direct_judge":
            return None  # no workspace Qwen direct-judge export in results_csv_only
        return lambda: rm(JOJO, suite, attack, policy, source="v3cap")
    if suite == "travel":
        return lambda: jerome("travel", attack, policy, [TRAVEL_27B[attack]])
    if policy == "sandbox_effects_judge" and attack == "important_instructions_generic":
        return lambda: rm("results_csv_jojo/report_metrics_banking_jojo.csv", suite, attack, policy)
    # Banking sandbox/important has one complete run plus two partial ones; skip the partials.
    return lambda: batches(ANANYA, suite, attack, policy, "Qwen3.8-27B-AWQ-INT4",
                           exclude=("85232f5d381b", "ef07a98ec5d5") if policy == "sandbox_judge" else ())


def pct(safe, n):
    return float((Decimal(100 * safe) / n).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load():
    """{(suite, condition, col, k): (n, safe, tcr)}; condition 'attacks' pools both attacks."""
    cells = {}
    for suite in SUITES:
        for attack in ATTACKS:
            expected = CASES[suite][0 if attack == "none" else 1]
            for col in COLUMNS:
                read = source(suite, attack, col)
                if read is None:
                    continue
                got = read()
                for k in (["inf"] if col == "full_approve" else K_ORDER):
                    if k not in got:
                        raise SystemExit(f"missing {suite}/{attack}/{col} k={k}")
                    n, safe = got[k]
                    if n != expected:
                        raise SystemExit(f"{suite}/{attack}/{col} k={k}: {n} cases, expected {expected}")
                    cells[(suite, attack, col, k)] = (n, safe, pct(safe, n))
        for col in COLUMNS:
            for k in (["inf"] if col == "full_approve" else K_ORDER):
                parts = [cells.get((suite, a, col, k)) for a in ATTACKS[1:]]
                if all(parts):
                    n, safe = sum(p[0] for p in parts), sum(p[1] for p in parts)
                    cells[(suite, "attacks", col, k)] = (n, safe, pct(safe, n))
    return cells


def figure(cells, condition, out: Path):
    xs = list(range(len(K_ORDER)))
    xs[-1] += 0.6
    fig, axes = plt.subplots(len(FAMILIES), len(SUITES), figsize=(10.8, 2.5 * len(FAMILIES) + 0.4),
                             sharey=True, squeeze=False)
    for i, (family, cols) in enumerate(FAMILIES):
        for j, suite in enumerate(SUITES):
            ax = axes[i][j]
            fa = cells.get((suite, condition, "full_approve", "inf"))
            ax.plot([xs[0], xs[-1]], [fa[2]] * 2, lw=1.4, label="Full approve", zorder=2,
                    **REF["full_approve"])
            fe = [cells[(suite, condition, "full_escalate", k)][2] for k in K_ORDER]
            ax.plot(xs, fe, lw=1.4, label="Full escalate", zorder=2, **REF["full_escalate"])
            for c, col in enumerate(cols):
                if (suite, condition, col, K_ORDER[0]) not in cells:
                    continue  # no source for this policy in this suite
                ys = [cells[(suite, condition, col, k)][2] for k in K_ORDER]
                style = dict(color=COLOR[col], marker=MARKER.get(col, "s"), ms=5,
                             markeredgecolor="white", markeredgewidth=1.0)
                ax.plot(xs[:-1], ys[:-1], lw=1.8, label=LABEL[col], zorder=3, **style)
                ax.plot(xs[-2:], ys[-2:], lw=1.0, ls=":", color=COLOR[col], zorder=2)
                ax.plot([xs[-1]], [ys[-1]], ls="none", zorder=3, **style)
            n = cells[(suite, condition, "full_escalate", "0")][0]
            if i == 0:
                ax.set_title(f"{suite.capitalize()} (N={n})", fontsize=9, color=INK, loc="left")
            ax.set_xticks(xs, ["0", "1", "3", "5", "8", "∞"])
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{family}\nTCR (%)", color=INK, fontsize=8.5)
            if i == len(FAMILIES) - 1:
                ax.set_xlabel("Escalation budget k (per task)", color=INK, fontsize=8.5)
        # One legend per row, from the workspace panel (travel for the Qwen row, where
        # workspace has no direct judge).
        src = axes[i][1] if family.startswith("Qwen") else axes[i][0]
        h, lab = src.get_legend_handles_labels()
        axes[i][-1].legend(h, lab, loc="lower right", fontsize=7, frameon=True, framealpha=0.9,
                           edgecolor=GRID)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def latex_rows(cells) -> str:
    """Body rows of the appendix TCR table (suite blocks: workspace, travel, banking)."""
    names = {"none": "Clean", "important_instructions_generic": "Important",
             "tool_knowledge_generic": "Tool"}
    lines = []
    for s, suite in enumerate(SUITES):
        lines.append(f"% {'=' * 21} {suite.upper()} {'=' * 21}")
        lines.append(f"\\multirow{{18}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{suite.capitalize()}}}}}")
        for a, attack in enumerate(ATTACKS):
            n = CASES[suite][0 if attack == "none" else 1]
            lines.append(f" & \\multirow{{6}}{{*}}{{\\shortstack[l]{{{names[attack]}\\\\($N{{=}}{n}$)}}}}")
            for i, k in enumerate(K_ORDER):
                vals = []
                for col in COLUMNS:
                    if col == "full_approve" and k != "inf":
                        vals.append("--")
                        continue
                    cell = cells.get((suite, attack, col, k))
                    vals.append(f"{cell[2]:.1f}" if cell else "\\pending")
                kk = "$\\infty$" if k == "inf" else k
                lines.append(f" {'& ' if i == 0 else '& & '}{kk} & " + " & ".join(vals) + " \\\\")
            if a < 2:
                lines.append("\\cmidrule(l){2-15}")
        if s < len(SUITES) - 1:
            lines.append("\\midrule" if s == 0 else "\\pagebreak")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    cells = load()
    args.tables.mkdir(parents=True, exist_ok=True)
    with open(args.tables / "tcr_at_k_all_policies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "condition", "policy", "k", "n", "safe", "tcr"])
        for (suite, cond, col, k), (n, safe, tcr) in sorted(cells.items()):
            w.writerow([suite, cond, col, k, n, safe, tcr])
    (args.tables / "tcr_all_appendix_rows.tex").write_text(latex_rows(cells) + "\n")
    figure(cells, "none", args.figures / "tcr_at_k_all_policies_clean.png")
    figure(cells, "attacks", args.figures / "tcr_at_k_all_policies_attacks.png")
    print(f"{len(cells)} cells; wrote {args.tables}/tcr_at_k_all_policies.csv, tcr_all_appendix_rows.tex "
          f"and {args.figures}/tcr_at_k_all_policies_{{clean,attacks}}.png")


if __name__ == "__main__":
    main()
