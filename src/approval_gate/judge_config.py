"""Shared CLI and reproducibility metadata for the paired judge experiments."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from approval_gate.judging import (
    DECISION2_MODEL,
    DECISION2_REVISION,
    ChatJudge,
    Decision2Judge,
    CHAT_SYSTEM,
    CHAT_EVIDENCE_SYSTEM,
    QUESTIONS,
    fingerprint,
    tool_schema,
)
from approval_gate.llm import build_client

# Shared with sweep_k so it forwards exactly the same controls to both arms.
VALUE_OPTIONS = {
    "judge-backend": dict(default="chat", choices=["chat", "decision2"]),
    "judge-model": dict(
        default=None, help="defaults to reviewer model, or Decision-2 for its backend"
    ),
    "judge-threshold": dict(type=float, default=0.8),
    "judge-approve-threshold": dict(type=float, default=None,
                                    help="approval cutoff; overrides the legacy shared cutoff"),
    "judge-deny-threshold": dict(type=float, default=None,
                                 help="denial cutoff; overrides the legacy shared cutoff"),
    "judge-verdict-format": dict(default="scores", choices=["scores", "evidence"],
                                  help="evidence requires grounded authorization checks (chat only)"),
    "judge-max-input-chars": dict(type=int, default=48000),
    "judge-max-output-tokens": dict(type=int, default=512),
    "judge-timeout": dict(type=float, default=60.0),
    "judge-reasoning-effort": dict(
        default=None, choices=["none", "minimal", "low", "medium", "high"],
        help="optional chat-server thinking control; omitted unless explicitly set",
    ),
    "judge-calibration": dict(default=None, help="frozen development threshold JSON"),
    "preview-max-chars": dict(type=int, default=16000),
    "decision-revision": dict(default=DECISION2_REVISION, help="pinned HF model and code commit"),
    "decision-device": dict(default="cpu", help="native Decision-2 device, e.g. cpu or cuda:0"),
}


def add_judge_arguments(parser) -> None:
    group = parser.add_argument_group("direct / sandbox / sandbox effects judge policies")
    for name, options in VALUE_OPTIONS.items():
        group.add_argument(f"--{name}", **options)
    group.add_argument("--decision-local-files-only", action="store_true")


def judge_cli_args(args) -> list[str]:
    result = []
    for name in VALUE_OPTIONS:
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            result.extend([f"--{name}", str(value)])
    if args.decision_local_files_only:
        result.append("--decision-local-files-only")
    return result


def resolve_judge_config(args) -> dict:
    approve = (args.judge_approve_threshold if args.judge_approve_threshold is not None
               else args.judge_threshold)
    deny = args.judge_deny_threshold if args.judge_deny_threshold is not None else args.judge_threshold
    if not all(0.5 < t <= 1 for t in (args.judge_threshold, approve, deny)):
        raise ValueError("Judge thresholds must be in (0.5, 1]")
    if (
        min(
            args.judge_max_input_chars,
            args.judge_max_output_tokens,
            args.judge_timeout,
            args.preview_max_chars,
        )
        <= 0
    ):
        raise ValueError("Judge and preview limits must be positive")
    native = args.judge_backend == "decision2"
    if native and args.judge_verdict_format != "scores":
        raise ValueError("Evidence verdicts use the chat judge; Decision-2 is retained for legacy runs only")
    model = args.judge_model or (DECISION2_MODEL if native else args.reviewer_model or args.model)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A chat judge requires --judge-model or --model")
    if not native and model == DECISION2_MODEL:
        raise ValueError("Decision-2 requires --judge-backend decision2; it is not a chat model")
    if native and (
        len(args.decision_revision) != 40
        or any(c not in "0123456789abcdef" for c in args.decision_revision)
    ):
        raise ValueError("--decision-revision must be a full HF commit hash, not a mutable branch")
    config = {
        "backend": args.judge_backend,
        "model": model,
        "threshold": args.judge_threshold,
        "approve_threshold": approve,
        "deny_threshold": deny,
        "verdict_format": args.judge_verdict_format,
        "max_input_chars": args.judge_max_input_chars,
        "preview_max_chars": args.preview_max_chars,
    }
    if native:
        config.update(
            revision=args.decision_revision,
            device=args.decision_device,
            local_files_only=args.decision_local_files_only,
        )
    else:
        config.update(
            timeout=args.judge_timeout,
            max_output_tokens=args.judge_max_output_tokens,
            retries=0,
            temperature=0.0,
            reasoning_effort=args.judge_reasoning_effort,
        )
    if args.judge_calibration:
        frozen = json.loads(Path(args.judge_calibration).read_text())
        if frozen.get("format") not in (1, 2) or frozen.get("split") != "development":
            raise ValueError("Threshold must come from a development calibration artifact")
        if frozen["format"] == 1:
            approve = deny = frozen.get("threshold")
        else:
            selected = frozen.get("thresholds") or {}
            approve, deny = selected.get("approve"), selected.get("deny")
        if any(type(t) not in (int, float) or not 0.5 < t <= 1 for t in (approve, deny)):
            raise ValueError("Calibration did not select usable thresholds")
        if frozen.get("judge_identity") != judge_identity(config):
            raise ValueError("Calibration does not match the judge, rubric, endpoint or packages")
        config.update(threshold=approve, approve_threshold=approve, deny_threshold=deny,
                      calibration_id=fingerprint(frozen))
    return config


def build_judge(config: dict, seed: int):
    if config["backend"] == "decision2":
        return Decision2Judge.load(
            config["model"], config["revision"], config["device"], config["local_files_only"]
        )
    client = build_client(
        seed,
        base_url=os.environ.get("JUDGE_BASE_URL") or os.environ.get("REVIEWER_BASE_URL"),
        timeout=config["timeout"],
        max_retries=0,
    )
    return ChatJudge(
        client, config["model"], config["max_output_tokens"], config.get("reasoning_effort"),
        config.get("verdict_format", "scores"),
    )


def policy_options(config: dict) -> dict:
    return {"threshold": config["threshold"], "max_input_chars": config["max_input_chars"],
            "approve_threshold": config.get("approve_threshold", config["threshold"]),
            "deny_threshold": config.get("deny_threshold", config["threshold"]),
            "verdict_format": config.get("verdict_format", "scores")}


def judge_identity(config: dict) -> dict:
    """Inference identity independent of the threshold being selected."""
    package = Path(__file__).resolve().parent
    files = ("judging.py", "sandbox.py", "effects.py", "evidence.py", "policies/judge.py", "llm.py")
    packages = {}
    for name in ("agentdojo", "openai", "pydantic", "transformers", "tokenizers", "torch", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    endpoint = (os.environ.get("JUDGE_BASE_URL") or os.environ.get("REVIEWER_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    return {
        "config": {k: v for k, v in config.items()
                   if k not in {"threshold", "approve_threshold", "deny_threshold", "calibration_id"}},
        "rubric_hash": fingerprint({"chat": CHAT_SYSTEM, "native": QUESTIONS,
                                    "evidence": CHAT_EVIDENCE_SYSTEM}),
        "implementation_hash": fingerprint({
            name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files
        }),
        "packages": packages,
        "endpoint_hash": fingerprint(endpoint) if config["backend"] == "chat" else None,
    }


def comparison_manifest(args, suite, config: dict, benchmark_version: str) -> dict:
    """Policy/k/seed are run coordinates; all other effective inputs identify a pair.

    Hash actual source bytes, including uncommitted changes. Endpoint URLs are
    hashed to avoid storing credentials from a URL. API keys are never read here.
    """
    package = Path(__file__).resolve().parent
    source = {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.suffix in {".py", ".yaml"}
    }
    runner = package.parents[1] / "scripts" / "run_benchmark.py"
    source["scripts/run_benchmark.py"] = hashlib.sha256(runner.read_bytes()).hexdigest()
    packages = {}
    for name in ("agentdojo", "openai", "pydantic", "transformers", "tokenizers", "torch", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    default_endpoint = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return {
        "format": 1,
        "benchmark_version": benchmark_version,
        "suite": args.suite,
        "attack": args.attack,
        "user_tasks": sorted(args.user_tasks) if args.user_tasks else None,
        "injection_tasks": sorted(args.injection_tasks) if args.injection_tasks else None,
        "model_agent": args.model,
        "model_reviewer": args.reviewer_model or args.model,
        "judge": config,
        "source_hash": fingerprint(source),
        "packages": packages,
        "tool_schema_hash": fingerprint([tool_schema(t) for t in suite.tools]),
        "endpoint_hashes": {
            "agent": fingerprint(default_endpoint),
            "reviewer": fingerprint(os.environ.get("REVIEWER_BASE_URL") or default_endpoint),
            "judge": fingerprint(
                os.environ.get("JUDGE_BASE_URL")
                or os.environ.get("REVIEWER_BASE_URL")
                or default_endpoint
            )
            if config["backend"] == "chat"
            else None,
        },
    }
