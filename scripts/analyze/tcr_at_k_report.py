"""TCR@k curves and table for the report, from the CSVs in results_csv_only. Makes no LLM calls.

Policies: full approve and full escalate (the two baselines) and the full cascade, for
workspace, travel, slack and banking, clean and under both attacks; workspace also has the
Qwen 27B effects judge (the main-text figure). Each (suite, attack, policy)
cell is read from one pinned source below; every budget must hold the suite's full case
set or the script stops.

TCR is the AgentDojo proxy: utility and not security (security=True means the attack
succeeded). "Attacks" pools both attacks (safe completions over all attacked cases).

Travel's tool-knowledge runs are labelled `tool_knowledge`; the group treats that as the
same attack as `tool_knowledge_generic` (the injection differs only in how it addresses
the model), so it is mapped onto it here.
"""

from __future__ import annotations

import argparse
import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results_csv_only")
SUITES = ["workspace", "travel", "slack", "banking"]
CASES = {"workspace": (40, 560), "travel": (20, 140), "slack": (21, 105), "banking": (16, 144)}
ATTACKS = ["none", "important_instructions_generic", "tool_knowledge_generic"]
K_ORDER = ["0", "1", "3", "5", "8", "inf"]
POLICIES = ["full_approve", "full_escalate", "cascade"]

# Fixed colour and marker per policy, as in tcr_curve.py (colour-blind safe; markers
# mean the lines don't rely on colour alone).
STYLE = {
    "full_approve": {"color": "#2a78d6", "marker": "s", "label": "Full approve"},
    "full_escalate": {"color": "#eb6834", "marker": "o", "label": "Full escalate"},
    "cascade": {"color": "#1baf7a", "marker": "D", "label": "Full cascade (ours)"},
    "sandbox_effects_judge": {"color": "#eda100", "marker": "^", "label": "Effects judge (Qwen 27B)"},
}
# The effects judge is shown for workspace only (Qwen 27B, Jojo's larger-cap v3cap run).
EFFECTS = "sandbox_effects_judge"


def policies_of(suite: str) -> list[str]:
    return [*POLICIES, EFFECTS] if suite == "workspace" else POLICIES
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
CONDITION_LABEL = {"none": "Clean", "important_instructions_generic": "Important instructions",
                   "tool_knowledge_generic": "Tool knowledge", "attacks": "Both attacks pooled"}


def report_metrics(path, suite, attack, policy, source=None, file_attack=None):
    """Rows of a report_metrics-style CSV (tcr in percent, rounded to 0.1)."""
    out = {}
    for r in csv.DictReader(open(ROOT / path, newline="")):
        if (r["suite"], r["policy"], r["attack"]) != (suite, policy, file_attack or attack):
            continue
        if source is not None and r["source"] != source:
            continue
        k = r["k"].removesuffix(".0")
        n = int(r["n"])
        # 0.1 % steps are finer than one case for every n here, so this recovers the count.
        out[k] = (n, round(float(r["tcr"]) * n / 100))
    return out


def jerome(suite, attack, policy, sources):
    """Rows of Jerome's consolidated CSV (tcr_mean as a fraction), from the given run dirs."""
    out = {}
    for r in csv.DictReader(open(ROOT / "results_csv_jerome/tcr_all_results.csv", newline="")):
        if (r["suite"], r["policy"], r["attack"]) != (suite, policy, attack):
            continue
        if not any(r["source_runs"].startswith(s) for s in sources):
            continue
        n = int(r["n_cases"])
        out[r["k"]] = (n, round(float(r["tcr_mean"]) * n))
    return out


RQ = "results_csv_rq/report_metrics_v2.csv"
ANANYA = "results_csv_ananya/report_metrics_ananya.csv"
TEJA_TRAVEL_TOOL = "results_csv_teja/travel_tool_knowledge.csv"

# One pinned source per (suite, attack, policy).
SOURCES = {}
for suite in ("workspace", "banking"):
    for attack in ATTACKS:
        for policy in POLICIES:
            SOURCES[(suite, attack, policy)] = lambda s=suite, a=attack, p=policy: report_metrics(
                RQ, s, a, p, source="v2")
for policy in POLICIES:
    SOURCES[("travel", "none", policy)] = lambda p=policy: report_metrics(
        ANANYA, "travel", "none", p, source="results")
    SOURCES[("slack", "none", policy)] = lambda p=policy: report_metrics(
        ANANYA, "slack", "none", p, source="results")
    SOURCES[("slack", "tool_knowledge_generic", policy)] = lambda p=policy: report_metrics(
        ANANYA, "slack", "tool_knowledge_generic", p, source="results")
    SOURCES[("travel", "tool_knowledge_generic", policy)] = lambda p=policy: report_metrics(
        TEJA_TRAVEL_TOOL, "travel", "tool_knowledge_generic", p, file_attack="tool_knowledge")
    SOURCES[("slack", "important_instructions_generic", policy)] = lambda p=policy: jerome(
        "slack", "important_instructions_generic", p,
        ["slack-20261019-232826/baselines/", "slack-20261019-232826/cascade/"])
SOURCES[("travel", "important_instructions_generic", "full_approve")] = lambda: jerome(
    "travel", "important_instructions_generic", "full_approve", ["runs"])
SOURCES[("travel", "important_instructions_generic", "full_escalate")] = lambda: jerome(
    "travel", "important_instructions_generic", "full_escalate", ["full-escalate-travel-20261019-160220/"])
# Stitched from two run sets of the same configuration: v3 (k=0,1,3) and the remaining budgets.
SOURCES[("travel", "important_instructions_generic", "cascade")] = lambda: jerome(
    "travel", "important_instructions_generic", "cascade", ["v3/", "cascade-travel-remaining-20261019-082222/"])
# Qwen 27B effects judge, 40K/80K preview/input characters (v3cap; the default-cap v3 run lacks
# tool-knowledge k=3..inf).
for attack in ATTACKS:
    SOURCES[("workspace", attack, EFFECTS)] = lambda a=attack: report_metrics(
        "results_csv_jojo/report_metrics_jojo.csv", "workspace", a, EFFECTS, source="v3cap")


def pct(safe: int, n: int) -> float:
    """Percent to one decimal, halves rounded up (as in the report's tables)."""
    return float((Decimal(100 * safe) / n).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load() -> list[dict]:
    rows = []
    for (suite, attack, policy), read in SOURCES.items():
        cells = read()
        expected = CASES[suite][0 if attack == "none" else 1]
        budgets = ["inf"] if policy == "full_approve" else K_ORDER
        for k in budgets:
            if k not in cells:
                raise SystemExit(f"missing {suite}/{attack}/{policy} k={k}")
            n, safe = cells[k]
            if n != expected:
                raise SystemExit(f"{suite}/{attack}/{policy} k={k}: {n} cases, expected {expected}")
            rows.append({"suite": suite, "condition": attack, "policy": policy, "k": k,
                         "n": n, "safe": safe, "tcr": pct(safe, n)})
    # Pool both attacks per (suite, policy, k).
    for suite in SUITES:
        for policy in policies_of(suite):
            for k in (["inf"] if policy == "full_approve" else K_ORDER):
                parts = [r for r in rows if (r["suite"], r["policy"], r["k"]) == (suite, policy, k)
                         and r["condition"] != "none"]
                assert len(parts) == 2, (suite, policy, k)
                n, safe = sum(r["n"] for r in parts), sum(r["safe"] for r in parts)
                rows.append({"suite": suite, "condition": "attacks", "policy": policy, "k": k,
                             "n": n, "safe": safe, "tcr": pct(safe, n)})
    return rows


def panel(ax, rows, suite, condition, show_ylabel, show_xlabel, policies, ylim=None):
    xs = list(range(len(K_ORDER)))
    xs[-1] += 0.6  # gap before the unlimited-budget endpoint
    for policy in policies:
        style = STYLE[policy]
        cell = {r["k"]: r for r in rows
                if (r["suite"], r["condition"], r["policy"]) == (suite, condition, policy)}
        if policy == "full_approve":
            # Budget-independent: measured once, drawn flat across every k.
            y = cell["inf"]["tcr"]
            ax.plot([xs[0], xs[-1]], [y, y], color=style["color"], lw=2, ls="--",
                    label=style["label"], zorder=4)
            ax.plot([xs[-1]], [y], color=style["color"], marker=style["marker"], ms=6, ls="none",
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            continue
        ys = [cell[k]["tcr"] for k in K_ORDER]
        ax.plot(xs[:-1], ys[:-1], color=style["color"], lw=2, marker=style["marker"], ms=6,
                markeredgecolor="white", markeredgewidth=1.2, label=style["label"], zorder=3)
        # k=inf is a separate endpoint, joined by a dotted segment across the gap.
        ax.plot(xs[-2:], ys[-2:], color=style["color"], lw=1.2, ls=":", zorder=2)
        ax.plot([xs[-1]], [ys[-1]], color=style["color"], marker=style["marker"], ms=6, ls="none",
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    n = next(r["n"] for r in rows if (r["suite"], r["condition"]) == (suite, condition))
    zoom = f" · zoom {ylim[0]}–{ylim[1]}%" if ylim else ""
    ax.set_title(f"{suite.capitalize()} · {CONDITION_LABEL[condition]} (N={n}){zoom}",
                 fontsize=9, color=INK, loc="left")
    ax.set_xticks(xs, ["0", "1", "3", "5", "8", "∞"])
    if ylim:
        ax.set_ylim(*ylim)
        ax.set_yticks(range(ylim[0], ylim[1] + 1, 5))
    else:
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    if show_ylabel:
        ax.set_ylabel("TCR (%)", color=INK, fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Escalation budget k (per task)", color=INK, fontsize=9)


def figure(rows, conditions, out_base: Path, suites=SUITES, policies=POLICIES, by_row=True):
    """Small multiples. by_row: one row per condition, one column per suite; otherwise one
    row per suite, one column per condition."""
    shape = (len(conditions), len(suites)) if by_row else (len(suites), len(conditions))
    width = 3.2 * shape[1] + 0.4
    fig, axes = plt.subplots(*shape, figsize=(width, 2.6 * shape[0] + 0.5), sharey=True, squeeze=False)
    for i, condition in enumerate(conditions):
        for j, suite in enumerate(suites):
            r, c = (i, j) if by_row else (j, i)
            panel(axes[r][c], rows, suite, condition, c == 0, r == shape[0] - 1, policies)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(policies), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.38 / (2.6 * shape[0] + 0.5)))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    plt.close(fig)


def workspace_figure(rows, out_base: Path, ylim=None):
    """Main-text figure: workspace, clean and pooled attacks side by side. ylim=None shows the
    full 0-100% range; a range such as (75, 95) zooms in on the policies near the top."""
    policies = policies_of("workspace")
    conditions = ["none", "attacks"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), squeeze=False)
    for c, condition in enumerate(conditions):
        panel(axes[0][c], rows, "workspace", condition, c == 0, True, policies, ylim=ylim)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(policies), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    plt.close(fig)


def write_tables(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tcr_at_k_report.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["suite", "condition", "policy", "k", "n", "safe", "tcr"])
        writer.writeheader()
        writer.writerows(rows)
    # Wide view: one row per (suite, condition, policy), TCR (%) per budget.
    with open(out_dir / "tcr_at_k_report_wide.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["suite", "condition", "n", "policy", *[f"k={k}" for k in K_ORDER]])
        for suite in SUITES:
            for condition in [*ATTACKS, "attacks"]:
                for policy in policies_of(suite):
                    cell = {r["k"]: r for r in rows
                            if (r["suite"], r["condition"], r["policy"]) == (suite, condition, policy)}
                    n = next(iter(cell.values()))["n"]
                    values = ([cell["inf"]["tcr"]] * len(K_ORDER) if policy == "full_approve"
                              else [cell[k]["tcr"] for k in K_ORDER])
                    writer.writerow([suite, condition, n, policy, *values])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    rows = load()
    write_tables(rows, args.tables)
    # Main text: workspace with the effects judge. Appendix: all four suites, baselines and cascade.
    workspace_figure(rows, args.figures / "tcr_at_k_workspace")
    workspace_figure(rows, args.figures / "tcr_at_k_workspace_zoom", ylim=(75, 95))
    # Appendix figures leave out Slack (its values stay in the tables).
    appendix_suites = [s for s in SUITES if s != "slack"]
    figure(rows, ["none", "attacks"], args.figures / "tcr_at_k_main", suites=appendix_suites)
    figure(rows, ATTACKS, args.figures / "tcr_at_k_by_attack", suites=appendix_suites)
    print(f"{len(rows)} cells; wrote {args.tables}/tcr_at_k_report*.csv and {args.figures}/"
          "tcr_at_k_workspace{,_zoom}.png / tcr_at_k_main.png / tcr_at_k_by_attack.png")


if __name__ == "__main__":
    main()
