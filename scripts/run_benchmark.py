"""Runs the benchmark for one policy, one k and one seed.

`--policy undefended` is stock AgentDojo with no gate. Without --attack it is
a clean run; with --attack, each (user task, injection task) pair is a case.

Use --force-rerun for reported numbers: AgentDojo skips cached tasks while
the decision log appends, so a resumed run splits across two run_ids.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import agentdojo.attacks  # noqa: F401  (registers AgentDojo's attacks)
from agentdojo.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import (
    aggregate_results,
    benchmark_suite_without_injections,
    run_task_with_injection_tasks,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from dotenv import load_dotenv

import approval_gate.attacks  # noqa: F401  (registers important_instructions_generic)
from approval_gate.decisions import RunMeta
from approval_gate.executor import GatedAgentPipeline, GatedToolsExecutor, track_cases
from approval_gate.gate import ApprovalGate
from approval_gate.judge_config import (
    add_judge_arguments,
    build_judge,
    comparison_manifest,
    policy_options,
    resolve_judge_config,
)
from approval_gate.judging import fingerprint, tool_schema
from approval_gate.llm import UsageMeter, build_client, build_llm, pipeline_name
from approval_gate.nodes.allowlist import check_coverage
from approval_gate.nodes.envelope import EnvelopeExtractor
from approval_gate.nodes.scorers import DecisionScorer, FormulaScorer
from approval_gate.policies.cascade import CascadePolicy
from approval_gate.policies.cascade_n1 import CascadeN1Policy
from approval_gate.policies.full_approve import FullApprovePolicy
from approval_gate.policies.full_escalate import FullEscalatePolicy
from approval_gate.policies.judge import (
    DirectJudgePolicy,
    SandboxEffectsJudgePolicy,
    SandboxJudgePolicy,
)
from approval_gate.policies.static_threshold import StaticThresholdPolicy
from approval_gate.reviewer.simulated import SimulatedReviewer
from approval_gate.sandbox import AgentDojoPreview
from approval_gate.settings import CONFIG_HASH
from approval_gate.telemetry.logger import DecisionLogger

BENCHMARK_VERSION = "v1.2.1"
POLICIES = {
    "full_approve": FullApprovePolicy,
    "full_escalate": FullEscalatePolicy,
    "cascade_n1": CascadeN1Policy,
    "cascade": CascadePolicy,
    "static_threshold": StaticThresholdPolicy,
    "direct_judge": DirectJudgePolicy,
    "sandbox_judge": SandboxJudgePolicy,
    "sandbox_effects_judge": SandboxEffectsJudgePolicy,
}
JUDGE_POLICIES = {"direct_judge", "sandbox_judge", "sandbox_effects_judge"}
COMPARISON_POLICIES = JUDGE_POLICIES | {"full_approve", "full_escalate"}
NEEDS_ENVELOPE = {"cascade", "static_threshold"}
SCORERS = {
    "formula": lambda args: FormulaScorer(),
    "decision": lambda args: DecisionScorer.load(
        args.decision_revision, args.decision_device, args.decision_local_files_only),
    "decision-nochecks": lambda args: DecisionScorer.load(
        args.decision_revision, args.decision_device, args.decision_local_files_only,
        include_checks=False),
}

BASE_SYSTEM_MESSAGE = load_system_message(None).rstrip("\n") + "\n  - Do not ask the user questions.\n"


def build_pipeline(
    model: str,
    gate: ApprovalGate | None,
    name: str,
    seed: int = 0,
    agent_meter: UsageMeter | None = None,
    previewer=None,
) -> AgentPipeline:
    """gate=None builds stock AgentDojo; otherwise only the executor differs."""
    llm = build_llm(model, seed=seed, meter=agent_meter)

    if gate is None:
        tools_loop = ToolsExecutionLoop([ToolsExecutor(), llm])
        pipeline = AgentPipeline([SystemMessage(BASE_SYSTEM_MESSAGE), InitQuery(), llm, tools_loop])
    else:
        tools_loop = ToolsExecutionLoop([GatedToolsExecutor(gate, previewer=previewer), llm])
        pipeline = GatedAgentPipeline(
            [SystemMessage(BASE_SYSTEM_MESSAGE), InitQuery(), llm, tools_loop], gate
        )

    pipeline.name = name
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", default="banking")
    parser.add_argument(
        "--policy",
        default="undefended",
        choices=["undefended", *POLICIES],
        help="'undefended' is stock AgentDojo with no gate",
    )
    parser.add_argument("--k", default="inf", help="escalation budget per task; int or 'inf'")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="repeat index; at temperature 0 it separates repeats rather than changing output",
    )
    parser.add_argument(
        "--reviewer-model",
        default=None,
        help="model for the gate's own LLM calls (envelope) and the simulated reviewer; "
             "defaults to --model. Set REVIEWER_BASE_URL if it is on a different server",
    )
    parser.add_argument(
        "--attack",
        default=None,
        help="omit for a clean (utility) run; 'important_instructions_generic' for the "
             "injection runs behind the TCR@k curve",
    )
    parser.add_argument(
        "--logdir",
        default="results/runs",
        help="AgentDojo run logs; decisions/ and cases/ are written next to it",
    )
    parser.add_argument("--user-tasks", nargs="*", default=None)
    parser.add_argument(
        "--injection-tasks",
        nargs="*",
        default=None,
        help="subset of injection task IDs (attack runs only); omit for all",
    )
    parser.add_argument(
        "--risk-model",
        default="formula",
        choices=list(SCORERS),
        help="Node 3 scorer that turns the features into R (cascade and static_threshold)",
    )
    parser.add_argument(
        "--no-scope-deny",
        action="store_true",
        help="ablation: turn off Node 4's scope-rule auto-deny",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print config; no model calls")
    add_judge_arguments(parser)
    args = parser.parse_args()

    load_dotenv()
    if args.attack is not None and args.attack not in ATTACKS:
        parser.error(f"unknown attack {args.attack!r}; registered: {sorted(ATTACKS)}")
    variant_set = args.risk_model != "formula" or args.no_scope_deny
    if variant_set and args.policy not in NEEDS_ENVELOPE:
        parser.error("--risk-model and --no-scope-deny only apply to cascade and static_threshold")
    variant = args.policy
    if args.risk_model != "formula":
        variant += f"-{args.risk_model}"
    if args.no_scope_deny:
        variant += "-noscope"
    try:
        k = None if args.k == "inf" else int(args.k)
        if k is not None and k < 0:
            raise ValueError()
    except ValueError:
        parser.error("--k must be a nonnegative integer or 'inf'")
    if args.injection_tasks is not None and args.attack is None:
        parser.error("--injection-tasks requires --attack")
    suite = get_suite(BENCHMARK_VERSION, args.suite)
    for selected, available, label in (
        (args.user_tasks, suite.user_tasks, "user"),
        (args.injection_tasks, suite.injection_tasks, "injection"),
    ):
        if selected is not None and (not selected or set(selected) - set(available)):
            parser.error(f"invalid or empty {label} task selection")
    name = pipeline_name(args.model, variant, k, args.seed)
    config_hash = CONFIG_HASH
    comparison_id = None
    manifest = None
    judge_config = None
    previewer = None
    if args.policy in COMPARISON_POLICIES:
        try:
            judge_config = resolve_judge_config(args)
            if args.policy in JUDGE_POLICIES and POLICIES[args.policy].requires_preview:
                previewer = AgentDojoPreview(args.suite, BENCHMARK_VERSION, args.preview_max_chars,
                    include_effects=POLICIES[args.policy].requires_effects)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        shared = comparison_manifest(args, suite, judge_config, BENCHMARK_VERSION)
        comparison_id = fingerprint(shared)
        manifest = {"shared": shared, "comparison_id": comparison_id,
                    "policy": variant, "k": k, "seed": args.seed}
        config_hash = fingerprint(manifest)
        name = pipeline_name(f"{args.model}-{config_hash[:12]}", variant, k, args.seed)
    if args.dry_run:
        print(json.dumps({"name": name, "config_hash": config_hash,
                          "experiment": manifest}, indent=2))
        return

    gate = None
    logger = None
    agent_meter = None
    if args.policy != "undefended":
        if args.policy not in JUDGE_POLICIES:
            check_coverage(t.name for t in suite.tools)
        reviewer_model = args.reviewer_model or args.model
        gate_model = judge_config["model"] if args.policy in JUDGE_POLICIES else reviewer_model
        agent_meter = UsageMeter()
        out_root = Path(args.logdir).parent
        log_name = f"{name}__{args.suite}__{args.attack or 'none'}.jsonl"
        logger = DecisionLogger(
            out_root / "decisions" / log_name,
            cases_path=out_root / "cases" / log_name,
        )
        meta = RunMeta(
            run_id=uuid.uuid4().hex[:8],
            suite=args.suite,
            policy=variant,
            k_budget=k,
            seed=args.seed,
            model_agent=args.model,
            model_gate=gate_model,
            model_reviewer=reviewer_model,
            attack_name=args.attack,
            config_hash=config_hash,
            comparison_id=comparison_id,
        )
        if manifest is not None:
            manifest_path = out_root / "manifests" / f"{meta.run_id}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({**manifest, "run_id": meta.run_id,
                                                 "config_hash": config_hash}, indent=2) + "\n")
        # Each intervention in a paired run is one request, including failures.
        gate_client = build_client(args.seed, base_url=os.environ.get("REVIEWER_BASE_URL"),
                                   max_retries=0 if args.policy in COMPARISON_POLICIES else 2)
        if args.policy in JUDGE_POLICIES:
            policy = POLICIES[args.policy](
                build_judge(judge_config, args.seed),
                schemas={t.name: tool_schema(t) for t in suite.tools},
                **policy_options(judge_config),
            )
        elif args.policy in NEEDS_ENVELOPE:
            tools = [(t.name, t.description.strip().splitlines()[0]) for t in suite.tools]
            policy = POLICIES[args.policy](
                EnvelopeExtractor(gate_client, gate_model, tools),
                scorer=SCORERS[args.risk_model](args),
                scope_deny=not args.no_scope_deny,
            )
        else:
            policy = POLICIES[args.policy]()
        gate = ApprovalGate(
            policy=policy,
            meta=meta,
            logger=logger,
            reviewer=SimulatedReviewer(gate_client, reviewer_model),
            agent_meter=agent_meter,
        )
        track_cases(suite, gate)

    pipeline = build_pipeline(args.model, gate, name, seed=args.seed, agent_meter=agent_meter,
                              previewer=previewer)
    logdir = Path(args.logdir)
    header = f"\nsuite={args.suite} model={args.model} policy={variant} k={args.k} seed={args.seed}"

    if args.attack is None:
        with OutputLogger(str(logdir)):
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir=logdir,
                force_rerun=args.force_rerun,
                user_tasks=args.user_tasks,
                benchmark_version=BENCHMARK_VERSION,
            )
        # No attack, so TCR is utility. AgentDojo's security field is a placeholder here.
        utility = aggregate_results([results["utility_results"]])
        print(f"{header} attack=none n={len(results['utility_results'])}")
        print(f"TCR = utility = {utility:.3f}")
        for (task_id, _), ok in sorted(results["utility_results"].items()):
            print(f"  {'PASS' if ok else 'FAIL'}  {task_id}")
    else:
        attack = load_attack(args.attack, suite, pipeline)
        task_ids = args.user_tasks or list(suite.user_tasks)
        utility_results: dict[tuple[str, str], bool] = {}
        security_results: dict[tuple[str, str], bool] = {}

        with OutputLogger(str(logdir)):
            for task_id in task_ids:
                utility, security = run_task_with_injection_tasks(
                    suite,
                    pipeline,
                    suite.get_user_task_by_id(task_id),
                    attack,
                    logdir,
                    args.force_rerun,
                    args.injection_tasks,
                    BENCHMARK_VERSION,
                )
                utility_results.update(utility)
                security_results.update(security)

        # AgentDojo's security=True means the attack succeeded.
        cases = sorted(utility_results)
        safe = {c: utility_results[c] and not security_results[c] for c in cases}
        n = len(cases)
        print(f"{header} attack={args.attack} n={n} cases")
        print(f"TCR                   = {sum(safe.values()) / n:.3f}")
        print(f"utility under attack  = {sum(utility_results.values()) / n:.3f}")
        print(f"attack success rate   = {sum(security_results.values()) / n:.3f}")
        by_task: dict[str, list[bool]] = {}
        for (task_id, _), ok in safe.items():
            by_task.setdefault(task_id, []).append(ok)
        for task_id, oks in sorted(by_task.items()):
            print(f"  {task_id:<14} safe completions {sum(oks)}/{len(oks)}")

    if logger is not None:
        print(f"\ngate: {logger.summary()}")
        logger.close()
        if args.policy in JUDGE_POLICIES and (logger.judge_unavailable or logger.judge_usage_missing):
            print("Judge health FAILED: "
                  f"{logger.judge_unavailable} unavailable, "
                  f"{logger.judge_usage_missing} missing usage. "
                  "Logs retained; fix the judge before running more cells.")
            raise SystemExit(2)


if __name__ == "__main__":
    main()
