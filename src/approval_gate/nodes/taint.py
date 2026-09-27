"""Node 3a: where each critical argument value first appeared."""

from __future__ import annotations

from approval_gate.nodes.arguments import appears_in, critical_args, flatten, normalize_text

USER = "USER"
DELEGATED = "DELEGATED"       # from the output of a tool call the user pointed at
TOOL_OUTPUT = "TOOL_OUTPUT"
AGENT = "AGENT"
UNKNOWN = "UNKNOWN"

_RANK = {USER: 0, DELEGATED: 1, AGENT: 2, UNKNOWN: 3, TOOL_OUTPUT: 4}


def _text(message) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(str(b.get("content", "")) for b in content if isinstance(b, dict))
    return str(content or "")


def _origin(value, prompt: str, outputs: list, agent: str) -> str:
    if appears_in(value, prompt):
        return USER
    for text, call in outputs:
        if appears_in(value, text):
            call_args = flatten(list(dict(call.args).values())) if call is not None else []
            if any(appears_in(a, prompt) for a in call_args):
                return DELEGATED
            return TOOL_OUTPUT
    if appears_in(value, agent):
        return AGENT
    return UNKNOWN


def taint_flags(ctx) -> dict[str, str]:
    prompt = normalize_text(ctx.user_prompt)
    outputs = [
        (normalize_text(_text(m)), m.get("tool_call"))
        for m in ctx.messages if m.get("role") == "tool"
    ]
    agent = " ".join(normalize_text(_text(m)) for m in ctx.messages if m.get("role") == "assistant")

    flags = {}
    for name, value in critical_args(dict(ctx.tool_call.args)).items():
        worst = USER
        for v in flatten(value):
            flag = _origin(v, prompt, outputs, agent)
            if _RANK[flag] > _RANK[worst]:
                worst = flag
        flags[name] = worst
    return flags
