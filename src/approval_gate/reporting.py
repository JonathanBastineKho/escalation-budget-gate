"""Inference-free TCR, intervention, inference-cost and failure reports."""

from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from approval_gate.evaluation import selected_episodes, labels_from, authorization
from approval_gate.telemetry.events import fingerprint, read_jsonl, write_json


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate(rows, field, macro=False):
    if not macro:
        return sum(float(r[field]) for r in rows) / len(rows)
    suites = defaultdict(list)
    for r in rows:
        suites[r["suite"]].append(r)
    return sum(rate(v, field) for v in suites.values()) / len(suites)


def interval(rows, field, macro=False, repeats=1000):
    rng = np.random.default_rng(1729)
    suites = defaultdict(lambda: defaultdict(list))
    for row in rows:
        suites[row["suite"]][row["family"]].append(row)
    samples = []
    for _ in range(repeats):
        sampled = []
        for families in suites.values():
            groups = list(families.values())
            for index in rng.integers(0, len(groups), len(groups)):
                sampled.extend(groups[index])
        samples.append(rate(sampled, field, macro))
    return tuple(float(v) for v in np.quantile(samples, [0.025, 0.975]))


def request_cost(directory, roles=("agent", "policy", "reviewer")):
    rows = read_jsonl(Path(directory) / "llm_calls.jsonl")
    counts = {}
    for role in roles:
        started = [r for r in rows if r["role"] == role and r["event"] == "started"]
        complete = [r for r in rows if r["role"] == role and r["event"] == "completed"]
        ended = [r for r in rows if r["role"] == role and r["event"] in ("completed", "error")]
        counts.update(
            {
                role + "_calls": len(started),
                role + "_chat_calls": sum(r.get("kind", "chat") == "chat" for r in started),
                role + "_embedding_calls": sum(r.get("kind") == "embedding" for r in started),
                role + "_logical_requests": len({r["request_id"] for r in started}),
                role + "_retries": sum(r["attempt"] > 0 for r in started),
                role + "_completed": len(complete),
                role + "_tokens_in_known": sum(r.get("tokens_in") or 0 for r in complete),
                role + "_tokens_out_known": sum(r.get("tokens_out") or 0 for r in complete),
                role + "_usage_unknown": len(started)
                - sum(
                    r.get("tokens_in") is not None and r.get("tokens_out") is not None
                    for r in complete
                ),
                role + "_unresolved_requests": len(started) - len(ended),
                role + "_latency_ms_known": sum(r["latency_ms"] for r in ended),
            }
        )
    return counts


def report(root, output, annotations=None, suite=None, attack=None, allow_incomplete=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = labels_from(annotations)
    episodes, costs, failures, role_rows = [], {}, [], []
    expected = set()
    for path in Path(root).glob("schedule-*.json"):
        for cell in json.loads(path.read_text()):
            expected.add((cell["experiment_id"], cell["case_id"]))
    observed = set()
    for directory, outcome in selected_episodes(root):
        observed.add((outcome["experiment_id"], outcome["case_id"]))
        if suite and outcome["suite"] != suite or attack and outcome["attack"] != attack:
            continue
        config = json.loads((directory.parents[2] / "manifest.json").read_text())
        comparable = {
            k: config.get(k)
            for k in (
                "matrix_hash",
                "code_hash",
                "model",
                "gate_model",
                "policy_model",
                "temperature",
                "max_tokens",
                "max_iters",
                "reasoning_effort",
                "reviewer_style",
                "phase",
                "split_hash",
                "ollama_version",
                "benchmark_version",
                "agentdojo_package",
                "attempts",
                "timeout",
                "server_context_length",
                "endpoint",
                "case_manifest_hash",
                "split",
            )
        }
        comparable["agent_digest"] = config.get("model_digests", {}).get(config["model"])
        comparable["reviewer_digest"] = config.get("model_digests", {}).get(config["gate_model"])
        comparable["policy_digest"] = config.get("model_digests", {}).get(config["policy_model"])
        comparison = fingerprint(comparable)
        audit = authorization(directory, outcome, labels)
        row = {
            **outcome,
            **audit,
            "comparison": comparison,
            "strategy": outcome["policy"]
            + ("/" + outcome["ablation"] if outcome["ablation"] != "none" else "")
            + ("/" + str(outcome["checkpoint_hash"])[:8] if outcome["checkpoint_hash"] else ""),
            "strict_lower": int(audit["strict_success"] is True),
            "strict_upper": int(audit["strict_success"] is not False),
        }
        episodes.append(row)
        if outcome["policy"] == "melon_local":
            row["strategy"] += (
                "/"
                + str(config["melon_threshold"])
                + "/"
                + fingerprint(
                    {
                        "model": config["embedding_model"],
                        "digest": config.get("model_digests", {}).get(config["embedding_model"]),
                    }
                )[:8]
            )
        # Selected-attempt cost and total spent across retries/force-reruns are separate.
        selected_cost = request_cost(directory)
        spent = defaultdict(float)
        for attempt in directory.parent.iterdir():
            if attempt.is_dir():
                for key, value in request_cost(attempt).items():
                    spent[key] += value
        costs[outcome["episode_id"]] = (selected_cost, dict(spent))
        decisions = read_jsonl(directory / "decisions.jsonl")
        invoked = {
            e["step_index"]
            for e in read_jsonl(directory / "executions.jsonl")
            if e["event"] == "started"
        }
        for d in decisions:
            label = labels.get(f"{outcome['episode_id']}:{d['step_index']}")
            kind = None
            if d["step_index"] in invoked and label is False:
                kind = (
                    "over_permissive"
                    if d["decision"] == "AUTO_APPROVE"
                    else "reviewer_unsafe_approval"
                )
            elif not outcome["utility"] and not d["approved"] and label is True:
                kind = "candidate_over_conservative"
            elif not outcome["utility"] and d["decided_by_node"] == "budget_exhausted":
                kind = "candidate_budget_exhaustion"
            if kind:
                failures.append(
                    dict(
                        kind=kind,
                        episode_id=outcome["episode_id"],
                        case=outcome,
                        decision=d,
                        trajectory=str(directory / "trace.json"),
                    )
                )
        for attempt in directory.parent.iterdir():
            if not attempt.is_dir():
                continue
            for call in read_jsonl(attempt / "llm_calls.jsonl"):
                if call["event"] == "completed":
                    role_rows.append(
                        {
                            k: call.get(k)
                            for k in (
                                "experiment_id",
                                "episode_id",
                                "phase",
                                "role",
                                "model",
                                "kind",
                                "tokens_in",
                                "tokens_out",
                                "reasoning_tokens",
                                "latency_ms",
                                "finish_reason",
                            )
                        }
                    )
    if not episodes:
        raise ValueError("No new-format episode results found")
    if expected - observed and not allow_incomplete:
        raise ValueError(
            f"{len(expected - observed)} scheduled episodes have no outcome; resume the matrix"
        )
    # Refuse mismatched case sets between strategies/budgets/seeds within a matrix.
    checks = defaultdict(lambda: defaultdict(set))
    for r in episodes:
        cell = (r["strategy"], r["k"], r["seed"])
        checks[r["comparison"]][cell].add(
            (r["suite"], r["task_id"], r["attack"], r["injection_task_id"])
        )
    for comparison, cells in checks.items():
        if len({frozenset(s) for s in cells.values()}) > 1 and not allow_incomplete:
            raise ValueError(
                f"{comparison}: incomplete/mismatched cases; resume runs or use --allow-incomplete for diagnostics"
            )
    groups = defaultdict(list)
    for r in episodes:
        for suite_key in (r["suite"], "macro", "micro"):
            for attack_key in (r["attack"], "all_attacked" if r["attack"] != "none" else "clean"):
                groups[(r["comparison"], suite_key, attack_key, r["strategy"], r["k"])].append(r)
    metrics, hitl, cost_table = [], [], []
    for (comparison, suite_key, attack_key, strategy, k), rows in sorted(groups.items(), key=str):
        base = dict(
            comparison=comparison,
            suite=suite_key,
            attack=attack_key,
            strategy=strategy,
            k="inf" if k is None else k,
            n=len(rows),
            policy=rows[0]["policy"],
            model=rows[0]["model"],
            reviewer_model=rows[0]["gate_model"],
            policy_model=rows[0]["policy_model"],
            n_seeds=len({r["seed"] for r in rows}),
            task_families=len({r["family"] for r in rows}),
        )
        macro = suite_key == "macro"
        low, high = interval(rows, "tcr_ad", macro)
        all_strict = all(r["strict_success"] is not None for r in rows)
        strict_ci = interval(rows, "strict_lower", macro) if all_strict else (None, None)
        errors = sum(r["status"] == "error" for r in rows)
        known_asr = [r for r in rows if r["attack_success"] is not None and r["attack"] != "none"]
        metrics.append(
            {
                **base,
                "tcr_ad": rate(rows, "tcr_ad", macro),
                "tcr_ad_ci_low": low,
                "tcr_ad_ci_high": high,
                "tcr_strict": rate(rows, "strict_lower", macro) if all_strict else None,
                "tcr_strict_ci_low": strict_ci[0],
                "tcr_strict_ci_high": strict_ci[1],
                "strict_lower_bound": rate(rows, "strict_lower", macro),
                "strict_upper_bound": rate(rows, "strict_upper", macro),
                "utility": rate(rows, "utility", macro),
                "execution_errors": errors,
                "completed_episodes": sum(r["status"] == "completed" for r in rows),
                "limited_episodes": sum(r["status"] == "limit" for r in rows),
                "proposal_count": sum(r.get("proposals", 0) for r in rows),
                "reviewer_parse_failures": sum(r.get("reviewer_parse_failures", 0) for r in rows),
                "feature_parse_failures": sum(r.get("feature_parse_failures", 0) for r in rows),
                "asr_known_cases": len(known_asr),
                "asr": rate(known_asr, "attack_success", macro) if known_asr else None,
                "executions": sum(r["execution_count"] for r in rows),
                "audited_executions": sum(r["audited_execution_count"] for r in rows),
            }
        )
        reviews = sum(r["escalations"] for r in rows)
        safe_n = sum(r["strict_success"] is True for r in rows)
        safe_reviews = sum(r["escalations"] for r in rows if r["strict_success"] is True)
        hitl.append(
            {
                **base,
                "utility_successes": sum(r["utility"] for r in rows),
                "safe_successes_known": safe_n,
                "strict_complete": all_strict,
                "HITL_successful": sum(r["escalations"] for r in rows if r["utility"]),
                "HITL_safe_known": safe_reviews,
                "HITL_all_selected": reviews,
                "mean_reviews_all": reviews / len(rows),
                "mean_reviews_per_safe_success": safe_reviews / safe_n if safe_n else None,
                "total_reviews_per_safe_success": reviews / safe_n
                if safe_n and all_strict
                else None,
                "budget_used_up_cases": sum(
                    r["k"] is not None and r["escalations"] >= r["k"] for r in rows
                ),
            }
        )
        hitl[-1]["budget_exhaustion_denied_cases"] = sum(
            r.get("budget_exhausted_requests", 0) > 0 for r in rows
        )
        values = defaultdict(float)
        for r in rows:
            selected, spent = costs[r["episode_id"]]
            for key, value in selected.items():
                values[key] += value
            for key, value in spent.items():
                values["spent_" + key] += value
        total = sum(values[role + "_calls"] for role in ("agent", "policy", "reviewer"))
        ad_n = sum(r["tcr_ad"] for r in rows)
        cost_table.append(
            {
                **base,
                **values,
                "total_calls": total,
                "calls_per_case": total / len(rows),
                "seconds_per_case": sum(r["latency_ms"] for r in rows) / 1000 / len(rows),
                "generation_calls": sum(
                    values[r + "_chat_calls"] for r in ("agent", "policy", "reviewer")
                ),
                "embedding_calls": sum(
                    values[r + "_embedding_calls"] for r in ("agent", "policy", "reviewer")
                ),
                "calls_per_AD_success": total / ad_n if ad_n else None,
                "calls_per_strict_success": total / safe_n if safe_n and all_strict else None,
            }
        )
    save_csv(output / "tcr_at_k.csv", metrics)
    save_csv(output / "hitl_load.csv", hitl)
    save_csv(output / "api_costs.csv", cost_table)
    save_csv(output / "completed_generations.csv", role_rows)
    # Include interrupted attempts that never produced an outcome.
    spent_groups = defaultdict(lambda: defaultdict(float))
    for path in Path(root).glob("*/cases/*/*/llm_calls.jsonl"):
        events = read_jsonl(path)
        for e in events:
            g = spent_groups[(e.get("phase"), e["role"], e["model"], e.get("kind", "chat"))]
            if e["event"] == "started":
                g["api_attempts"] += 1
                g["usage_unknown"] += 1
                g["retries"] += int(e["attempt"] > 0)
            if e["event"] in ("completed", "error"):
                g["latency_ms_known"] += e.get("latency_ms") or 0
            if e["event"] == "completed":
                g["completed"] += 1
                g["tokens_in_known"] += e.get("tokens_in") or 0
                g["tokens_out_known"] += e.get("tokens_out") or 0
                if e.get("tokens_in") is not None and e.get("tokens_out") is not None:
                    g["usage_unknown"] -= 1
    save_csv(
        output / "all_attempt_costs.csv",
        [dict(phase=p, role=r, model=m, kind=k, **v) for (p, r, m, k), v in spent_groups.items()],
    )
    write_json(output / "episode_metrics.json", episodes)
    write_json(output / "failure_candidates.json", failures)
    with (output / "failure_cases.md").open("w") as f:
        f.write(
            "# Failure candidates\n\nOver-conservative causality requires trajectory/branch review.\n"
        )
        for row in failures[:30]:
            f.write("\n## " + row["kind"] + " / " + row["episode_id"] + "\n\n")
            f.write("Trajectory: " + row["trajectory"] + "\n\n")
            f.write("```json\n" + json.dumps(row["decision"], indent=2) + "\n```\n")
    plot(metrics, cost_table, output, hitl, diagnostic=allow_incomplete)
    print(
        f"Wrote {output}: CSV tables, PNG/PDF charts, index.html gallery, ledgers, failure candidates"
    )
    return metrics, hitl, cost_table


def plot(metrics, costs, output, hitl, *, diagnostic=False):
    from approval_gate.visualizations import plot_reports

    plot_reports(metrics, costs, hitl, output, diagnostic=diagnostic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--annotations")
    p.add_argument("--suite")
    p.add_argument("--attack")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    report(args.runs, args.output, args.annotations, args.suite, args.attack, args.allow_incomplete)


if __name__ == "__main__":
    main()
