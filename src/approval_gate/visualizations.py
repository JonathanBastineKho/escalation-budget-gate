"""Offline figures and a local HTML gallery for matched experiment reports."""

from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np


POLICIES = {
    "full_approve": ("Full-approve", "#596579"),
    "full_escalate": ("Full-escalate", "#D55E00"),
    "cascade": ("Cascade", "#0072B2"),
    "llm_judge": ("Prompted alignment", "#CC79A7"),
    "learned_risk": ("Learned risk", "#009E73"),
    "learned_voi": ("Learned review value", "#7651A8"),
    "melon_local": ("MELON local", "#A66B00"),
    "undefended": ("Undefended", "#596579"),
}
BUDGET_INDEPENDENT = {"full_approve", "melon_local", "undefended"}
CHARTS = {
    "tcr": "Completion versus review budget",
    "cost": "Local API attempts by role",
    "hitl": "Simulated review load and budget exhaustion",
    "tradeoff": "Completion versus review and inference cost",
    "runtime": "Episode runtime",
    "audit": "Run status and authorization audit coverage",
}


def budget_key(value):
    return float("inf") if value in (None, "inf") else int(value)


def budget_label(row):
    if row["policy"] in BUDGET_INDEPENDENT:
        return "any"
    return "∞" if budget_key(row["k"]) == float("inf") else str(row["k"])


def strategy_label(row):
    base = POLICIES.get(row["policy"], (row["policy"], ""))[0]
    suffix = row["strategy"][len(row["policy"]) :].replace("_", " ")
    return base + suffix


def cell_label(row):
    return f"{strategy_label(row)} · k={budget_label(row)}"


def row_key(row):
    return tuple(row[k] for k in ("comparison", "suite", "attack", "strategy")) + (
        budget_key(row["k"]),
    )


def sort_key(row):
    policies = list(POLICIES)
    rank = policies.index(row["policy"]) if row["policy"] in policies else len(policies)
    return rank, row["strategy"], budget_key(row["k"])


def matched_groups(metrics, costs, hitl):
    """Join on all comparison dimensions; never mix duplicate alias/aggregate rows."""
    indexes = []
    for table in (costs, hitl):
        index = {row_key(r): r for r in table}
        if len(index) != len(table):
            raise ValueError("Duplicate configuration rows in chart input")
        indexes.append(index)
    groups = defaultdict(list)
    seen = set()
    for metric in metrics:
        if metric["attack"] not in ("clean", "all_attacked"):
            continue
        key = row_key(metric)
        if key in seen:
            raise ValueError("Duplicate metric row in chart input")
        seen.add(key)
        if any(key not in index for index in indexes):
            raise ValueError("Missing matching cost/HITL row in chart input")
        cost, load = (index[key] for index in indexes)
        if not (metric["n"] == cost["n"] == load["n"] and metric["n"] > 0):
            raise ValueError("Mismatched or empty chart denominators")
        groups[key[:3]].append((metric, cost, load))
    return {key: sorted(rows, key=lambda r: sort_key(r[0])) for key, rows in groups.items()}


def _colors(rows):
    # Keep each main policy's color stable across suites, attacks, and chart types.
    return {row["strategy"]: POLICIES.get(row["policy"], ("", "#333333"))[1] for row, _, _ in rows}


def _bar_axes(rows, count=1):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, count, figsize=(12, max(4.2, len(rows) * 0.38 + 2)), layout="constrained", squeeze=False
    )
    labels = [cell_label(r) for r, _, _ in rows]
    for ax in axes[0]:
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    if count > 1:
        axes[0, 1].set_yticklabels([])
    return fig, axes[0]


def _tcr(rows, colors):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    budgets = sorted({budget_key(r["k"]) for r, _, _ in rows})
    by_strategy = defaultdict(list)
    for row, _, _ in rows:
        by_strategy[row["strategy"]].append(row)
    for strategy, selected in by_strategy.items():
        color = colors[strategy]
        label = strategy_label(selected[0])
        independent = selected[0]["policy"] in BUDGET_INDEPENDENT
        if independent:
            xs, selected = np.arange(len(budgets)), [selected[0]] * len(budgets)
        else:
            xs = np.array([budgets.index(budget_key(r["k"])) for r in selected])
        ys = np.array([r["tcr_ad"] for r in selected])
        axes[0].plot(
            xs,
            ys,
            marker="s" if independent else "o",
            linestyle="--" if independent else "-",
            label=label,
            color=color,
        )
        axes[0].vlines(
            xs,
            [r["tcr_ad_ci_low"] for r in selected],
            [r["tcr_ad_ci_high"] for r in selected],
            color=color,
            alpha=0.6,
        )
        lower = [r["strict_lower_bound"] for r in selected]
        upper = [r["strict_upper_bound"] for r in selected]
        unresolved = any(r["tcr_strict"] is None for r in selected)
        if unresolved:
            axes[1].fill_between(xs, lower, upper, color=color, alpha=0.09)
            axes[1].plot(xs, lower, ":", color=color, alpha=0.75)
            axes[1].plot(xs, upper, ":", color=color, label=label)
            # Also show bounds for a one-budget report, where fill_between has no width.
            axes[1].vlines(xs, lower, upper, color=color, alpha=0.3, linewidth=4)
        known_y = [np.nan if r["tcr_strict"] is None else r["tcr_strict"] for r in selected]
        axes[1].plot(
            xs,
            known_y,
            marker="o",
            color=color,
            linestyle="--" if independent else "-",
            label=None if unresolved else label,
        )
        for x, row in zip(xs, selected):
            if row["tcr_strict"] is not None:
                axes[1].vlines(
                    x, row["tcr_strict_ci_low"], row["tcr_strict_ci_high"], color=color, alpha=0.6
                )
    for ax in axes:
        ax.set_xticks(range(len(budgets)), ["∞" if k == float("inf") else str(k) for k in budgets])
        ax.set_xlabel("Review budget k (∞ = no budget cap)")
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title("AgentDojo proxy · 95% task-family bootstrap CI")
    axes[0].set_ylabel("TCR_AD")
    axes[1].set_title("Strict TCR · dotted/shaded regions are label bounds")
    axes[1].set_ylabel("Strict safe task completion")
    if all(r["tcr_strict"] is None for r, _, _ in rows):
        axes[1].text(
            0.5,
            0.97,
            "Authorization audit incomplete: no point estimates",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    return fig


def _cost(rows, colors):
    fig, (ax,) = _bar_axes(rows)
    bottom = np.zeros(len(rows))
    for role, color in (("agent", "#4477AA"), ("policy", "#AA3377"), ("reviewer", "#EE9944")):
        values = np.array([c[role + "_calls"] / c["n"] for _, c, _ in rows])
        ax.barh(range(len(rows)), values, left=bottom, label=role.title(), color=color)
        bottom += values
    for i, total in enumerate(bottom):
        ax.annotate(
            f"{total:.2f}", (total, i), xytext=(4, 0), textcoords="offset points", va="center"
        )
    ax.set_xlim(0, max(1, max(bottom) * 1.17))
    ax.set_xlabel("Physical local API attempts per case (including request retries)")
    ax.set_title("Selected episodes · agent / policy / reviewer")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    return fig


def _hitl(rows, colors):
    from matplotlib.ticker import PercentFormatter

    fig, (ax, exhausted_ax) = _bar_axes(rows, 2)
    success = np.array([h["HITL_successful"] / h["n"] for _, _, h in rows])
    other = np.array([(h["HITL_all_selected"] - h["HITL_successful"]) / h["n"] for _, _, h in rows])
    ax.barh(range(len(rows)), success, color="#009E73", label="In utility-successful episodes")
    ax.barh(range(len(rows)), other, left=success, color="#AAB2BF", label="In other episodes")
    for i, total in enumerate(success + other):
        ax.annotate(
            f"{total:.2f}", (total, i), xytext=(3, 0), textcoords="offset points", va="center"
        )
    ax.set_xlim(0, max(1, max(success + other) * 1.25))
    ax.set_title("Simulated reviewer interventions")
    ax.set_xlabel("Reviews per case (denominator: all cases)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    exhausted = [h["budget_exhaustion_denied_cases"] / h["n"] for _, _, h in rows]
    exhausted_ax.barh(range(len(rows)), exhausted, color="#D55E00")
    exhausted_ax.set_xlim(0, 1.17)
    exhausted_ax.xaxis.set_major_formatter(PercentFormatter(1))
    exhausted_ax.set_xlabel("Share of episodes")
    exhausted_ax.set_title("At least one denial because k ran out")
    for i, (_, _, load) in enumerate(rows):
        exhausted_ax.annotate(
            f"{load['budget_exhaustion_denied_cases']}/{load['n']}",
            (exhausted[i], i),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
        )
    return fig


def _tradeoff(rows, colors):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    seen = set()
    for index, (metric, cost, load) in enumerate(rows):
        strategy = metric["strategy"]
        label = strategy_label(metric) if strategy not in seen else None
        seen.add(strategy)
        for ax, x in zip(axes, (load["mean_reviews_all"], cost["calls_per_case"])):
            y = metric["tcr_ad"]
            ax.vlines(
                x,
                metric["tcr_ad_ci_low"],
                metric["tcr_ad_ci_high"],
                color=colors[strategy],
                alpha=0.35,
            )
            ax.scatter(
                x, y, label=label, color=colors[strategy], s=48, edgecolors="white", zorder=3
            )
            ax.annotate(
                "k=" + budget_label(metric),
                (x, y),
                xytext=(5, 5 if index % 2 == 0 else -12),
                textcoords="offset points",
                fontsize=8,
                color=colors[strategy],
            )
    for ax in axes:
        ax.set_ylim(-0.06, 1.08)
        ax.set_xlim(left=-0.3, right=max(1, ax.get_xlim()[1] * 1.12))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("TCR_AD · AgentDojo proxy")
        ax.grid(alpha=0.18)
        ax.set_title("Higher completion with fewer calls is preferable")
    axes[0].set_xlabel("Mean simulated reviews per case")
    axes[1].set_xlabel("Mean local API attempts per case (all roles)")
    axes[0].legend(fontsize=8, loc="best")
    fig.supxlabel(
        "Vertical bars: 95% task-family bootstrap CI; k labels identify budget settings", fontsize=9
    )
    return fig


def _runtime(rows, colors):
    fig, (ax,) = _bar_axes(rows)
    bars = ax.barh(
        range(len(rows)),
        [c["seconds_per_case"] for _, c, _ in rows],
        color=[colors[r["strategy"]] for r, _, _ in rows],
    )
    ax.bar_label(bars, fmt="%.1f s", padding=4)
    ax.set_xlim(0, max(1, max(c["seconds_per_case"] for _, c, _ in rows) * 1.2))
    ax.set_xlabel("Mean recorded episode runtime (seconds/case)")
    ax.set_title("Includes agent, policy, reviewer and tool work; excludes between-episode gaps")
    return fig


def _audit(rows, colors):
    from matplotlib.ticker import PercentFormatter

    fig, (status_ax, audit_ax) = _bar_axes(rows, 2)
    left = np.zeros(len(rows))
    for key, label, color in (
        ("completed_episodes", "Completed", "#0072B2"),
        ("limited_episodes", "Iteration limit", "#E69F00"),
        ("execution_errors", "Processing error", "#CC6677"),
    ):
        values = np.array([r[key] / r["n"] for r, _, _ in rows])
        status_ax.barh(range(len(rows)), values, left=left, label=label, color=color)
        left += values
    status_ax.set_title("Terminal status (completed does not mean successful)")
    status_ax.set_xlabel("Share of episodes")
    status_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    known = np.array(
        [r["audited_executions"] / r["executions"] if r["executions"] else 0 for r, _, _ in rows]
    )
    unknown = np.array([1 - x if r["executions"] else 0 for x, (r, _, _) in zip(known, rows)])
    audit_ax.barh(range(len(rows)), known, label="Human-verified label", color="#009E73")
    audit_ax.barh(
        range(len(rows)), unknown, left=known, label="Authorization unresolved", color="#D5DAE2"
    )
    for i, (metric, _, _) in enumerate(rows):
        label = (
            f"{metric['audited_executions']}/{metric['executions']} audited"
            if metric["executions"]
            else "N/A: no calls executed"
        )
        audit_ax.text(0.02, i, label, va="center", fontsize=8, color="#17212B")
    audit_ax.set_title("Executed-call audit coverage (not a safety rate)")
    audit_ax.set_xlabel("Share of executed calls")
    audit_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    for ax in (status_ax, audit_ax):
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    return fig


def _gallery(output, sections, diagnostic):
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Approval policy experiment charts</title>",
        "<style>body{font:16px system-ui,sans-serif;background:#f4f6f9;color:#17212b;"
        "max-width:1400px;margin:auto;padding:24px}h1{margin-bottom:8px}p{line-height:1.6}"
        "a{color:#005da8}details{background:white;padding:18px;margin:18px 0;border-radius:10px}"
        "summary{font-weight:650;cursor:pointer}figure{margin:24px 0}img{width:100%;height:auto}"
        "figcaption{font-weight:600}nav{display:flex;gap:16px;flex-wrap:wrap}.note{background:#e7eef8;"
        "padding:14px;border-radius:8px}code{overflow-wrap:anywhere}</style>",
        "<h1>Approval policy experiment charts</h1>",
        "<p>Offline charts from saved outcomes and API ledgers. Each section keeps its comparison "
        "ID, suite and attack group separate. Open a section and click a chart to enlarge it. "
        "PNG and PDF exports are available for every figure.</p>",
        "<p class='note'><strong>Baseline interpretation:</strong> Full-approve automatically "
        "approves every tool call and uses no reviews. Full-escalate at k=∞ reviews every proposed "
        "call; at finite k, further escalation requests are denied when the budget is exhausted. "
        "Reviews are simulated by a fallible LLM, so full review is not a safety guarantee.</p>",
        "<p>TCR_AD is the AgentDojo proxy. Strict TCR needs human-verified authorization labels; "
        "dotted/shaded bounds are not point estimates or confidence intervals. Micro TCR weights "
        "cases equally; macro TCR weights suites equally. Cost, runtime and HITL plots use "
        "per-case means or counts, including in macro sections. Audit coverage counts executed "
        "call instances, which can repeat across configurations. API charts count selected "
        "attempts and their request retries; all_attempt_costs.csv includes other run attempts.</p>",
    ]
    if diagnostic:
        parts.append(
            "<p class='note'><strong>Diagnostic report:</strong> incomplete or unmatched case "
            "sets were permitted. The charts must not be treated as a matched policy comparison.</p>"
        )
    parts.append("<nav>")
    for name in ("tcr_at_k", "hitl_load", "api_costs", "all_attempt_costs"):
        parts.append(f"<a href='{name}.csv'>{name}.csv</a>")
    parts.append("</nav>")
    for (comparison, suite, attack), figures in sections:
        label = " / ".join((suite, attack, comparison))
        parts.append(
            f"<details{' open' if suite == 'micro' else ''}><summary>{escape(label)}</summary>"
        )
        for kind, stem in figures:
            caption = escape(CHARTS[kind])
            safe_stem = escape(stem, quote=True)
            parts.append(
                f"<figure><figcaption>{caption} · <a href='{safe_stem}.png'>PNG</a> · "
                f"<a href='{safe_stem}.pdf'>PDF</a></figcaption>"
                f"<a href='{safe_stem}.png'><img loading='lazy' src='{safe_stem}.png' "
                f"alt='{caption}'></a></figure>"
            )
        parts.append("</details>")
    parts.append("</html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def plot_reports(metrics, costs, hitl, output, *, diagnostic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = matched_groups(metrics, costs, hitl)
    renderers = dict(zip(CHARTS, (_tcr, _cost, _hitl, _tradeoff, _runtime, _audit)))
    sections = []
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "savefig.dpi": 170}):
        for key in sorted(groups, key=lambda k: (k[0], k[1] != "micro", k[1], k[2])):
            rows = groups[key]
            colors = _colors(rows)
            comparison, suite, attack = key
            figures = []
            ns = sorted({r["n"] for r, _, _ in rows})
            count = str(ns[0]) if len(ns) == 1 else f"{ns[0]}–{ns[-1]}"
            for kind, renderer in renderers.items():
                fig = renderer(rows, colors)
                try:
                    title = f"{suite} / {attack} · {CHARTS[kind]}"
                    detail = f"{count} episodes per configuration · comparison {comparison}"
                    if diagnostic:
                        detail += " · DIAGNOSTIC: unmatched/incomplete cases allowed"
                    fig.suptitle(title + "\n" + detail, fontsize=11)
                    stem = f"{kind}_{comparison}_{suite}_{attack}"
                    for ext in ("png", "pdf"):
                        fig.savefig(output / f"{stem}.{ext}", bbox_inches="tight")
                    figures.append((kind, stem))
                finally:
                    plt.close(fig)
            sections.append((key, figures))
    _gallery(output, sections, diagnostic)
