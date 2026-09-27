"""Shared experiment options."""

import argparse


def budget(text):
    if text == "inf":
        return None
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("k must be nonnegative")
    return value


def add_model_options(p):
    p.add_argument("--model", required=True)
    p.add_argument("--gate-model")
    p.add_argument("--policy-model")
    p.add_argument("--checkpoint")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument(
        "--server-context-length",
        type=int,
        default=16384,
        help="record expected context length; configure Ollama separately",
    )
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    p.add_argument("--reviewer-style", choices=["sanitized", "raw"], default="sanitized")
    p.add_argument(
        "--ablation", choices=["none", "no_budget", "no_provenance", "tool_only"], default="none"
    )
    p.add_argument("--capture-snapshots", action="store_true")
    p.add_argument(
        "--phase", choices=["evaluation", "training", "pilot", "diagnostic"], default="evaluation"
    )


def config_from_args(args):
    keys = [
        "model",
        "policy",
        "k",
        "seed",
        "temperature",
        "max_tokens",
        "max_iters",
        "attempts",
        "timeout",
        "server_context_length",
        "reasoning_effort",
        "reviewer_style",
        "ablation",
        "capture_snapshots",
        "phase",
        "checkpoint",
    ]
    c = {key: getattr(args, key) for key in keys}
    c["gate_model"] = args.gate_model or args.model
    c["policy_model"] = args.policy_model or args.model
    return c
