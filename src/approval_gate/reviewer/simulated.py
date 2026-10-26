"""LLM-simulated human reviewer.

Tool outputs and call arguments are wrapped in untrusted-data markers by
default. These markers identify data for the reviewer; they do not guarantee
protection against prompt injection. style="raw" disables them for the ablation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from approval_gate.policies.base import CallContext
from approval_gate.reviewer.prompts import SYSTEM, USER_TEMPLATE

OPEN = "<<<UNTRUSTED_DATA>>>"
CLOSE = "<<<END_UNTRUSTED_DATA>>>"


def _spotlight(text: str, style: str = "sanitized") -> str:
    """Wrap untrusted text in markers. Any markers inside the text are
    removed first so it can't close the block early."""
    if style == "raw":
        return text
    return f"{OPEN}\n{text.replace(OPEN, '').replace(CLOSE, '')}\n{CLOSE}"


def _tool_output_text(content: object) -> str:
    """Join tool content blocks and trim whitespace around the resulting text."""
    if isinstance(content, list):
        content = "\n".join(
            str(block.get("content", "")) for block in content if isinstance(block, dict)
        )
    return str(content or "").strip()


def _prior_actions(messages, style: str = "sanitized") -> str:
    """Format prior calls, outputs, and errors using the requested marker style."""
    lines: list[str] = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                lines.append(
                    f"called {call.function} with:\n{_spotlight(str(dict(call.args)), style)}"
                )
        elif message.get("role") == "tool":
            text = _tool_output_text(message.get("content"))
            if text:
                lines.append(f"returned:\n{_spotlight(text, style)}")
            if message.get("error"):
                lines.append(f"failed, not executed:\n{_spotlight(str(message['error']), style)}")

    return "\n".join(lines) or "(no prior actions)"


@dataclass
class ReviewResult:
    verdict: str  # "APPROVE" | "DENY"
    reason: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    parse_failed: bool = False
    usage_known: bool = True


def _parse(text: str) -> tuple[str, str, bool]:
    """Only a JSON object with a valid verdict counts. Anything else is DENY,
    never guessed from prose ("do not approve" contains APPROVE)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().upper()
            if verdict in ("APPROVE", "DENY"):
                return verdict, str(parsed.get("reason", ""))[:200], False
    return "DENY", "reviewer output could not be parsed; failing closed", True


class SimulatedReviewer:
    def __init__(self, client, model: str, style: str = "sanitized") -> None:
        # client comes from llm.build_client, which already pins temperature.
        self.client = client
        self.model = model
        self.style = style

    def review(self, ctx: CallContext) -> ReviewResult:
        started = time.perf_counter()

        user_message = USER_TEMPLATE.format(
            user_prompt=ctx.user_prompt,
            prior_actions=_prior_actions(ctx.messages, style=self.style),
            tool_name=ctx.tool_call.function,
            tool_args=_spotlight(str(dict(ctx.tool_call.args)), self.style),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content or ""
        verdict, reason, parse_failed = _parse(text)
        usage = response.usage
        usage_known = (
            usage is not None
            and isinstance(getattr(usage, "prompt_tokens", None), int)
            and isinstance(getattr(usage, "completion_tokens", None), int)
        )

        return ReviewResult(
            verdict=verdict,
            reason=reason,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            parse_failed=parse_failed,
            usage_known=usage_known,
        )
