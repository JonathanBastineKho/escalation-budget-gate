"""LLM-simulated human reviewer.

Tool outputs and the proposed arguments are always wrapped in untrusted-data
markers, so an injection that fooled the agent can't instruct the reviewer
the same way. style="raw" turns that off for the ablation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from approval_gate.nodes.allowlist import classify
from approval_gate.policies.base import CallContext
from approval_gate.reviewer.prompts import (
    SYSTEM,
    TASK_SCOPED_SYSTEM,
    TASK_SCOPED_TEMPLATE,
    USER_TEMPLATE,
)

OPEN = "<<<UNTRUSTED_DATA>>>"
CLOSE = "<<<END_UNTRUSTED_DATA>>>"
REVIEWER_MODES = ("legacy", "task_scoped")
EFFECT_FIELDS = ("authorized_effect", "proposed_effect", "state_after_execution")


def _call_dict(call) -> dict:
    if isinstance(call, dict):
        return {k: call.get(k) for k in ("id", "function", "args")}
    return {"id": call.id, "function": call.function, "args": dict(call.args)}


def _evidence(ctx: CallContext) -> dict:
    """Whitelist observation fields; omit assistant rationales and all oracle data."""
    observations = []
    calls = {}
    for message in ctx.messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                data = _call_dict(call)
                if data["id"] is not None:
                    calls[data["id"]] = data
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            source = calls.get(call_id)
            if source is None and message.get("tool_call") is not None:
                source = _call_dict(message["tool_call"])
            observations.append({
                "source_call": source,
                "call_id": call_id,
                "error": message.get("error"),
                "output": message.get("content"),
            })
    return {"observations": observations, "proposed_call": _call_dict(ctx.tool_call)}


def _spotlight(text: str, style: str = "sanitized") -> str:
    """Wrap untrusted text in markers. Any markers inside the text are
    removed first so it can't close the block early."""
    if style == "raw":
        return text
    return f"{OPEN}\n{text.replace(OPEN, '').replace(CLOSE, '')}\n{CLOSE}"


def _prior_actions(messages, style: str = "sanitized") -> str:
    """Every tool call so far with its full output, outputs spotlighted."""
    lines: list[str] = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                lines.append(f"called {call.function}({dict(call.args)})")
        elif message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    str(b.get("content", "")) for b in content if isinstance(b, dict)
                )
            text = str(content or "").strip()
            if text:
                lines.append(f"returned:\n{_spotlight(text, style)}")

    return "\n".join(lines) or "(no prior actions)"


@dataclass
class ReviewResult:
    verdict: str          # "APPROVE" | "DENY"
    reason: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    parse_failed: bool = False
    assessment: dict = field(default_factory=dict)
    raw_response: str = ""
    validation_errors: list[str] = field(default_factory=list)
    finish_reason: str | None = None


def _json_payload(text: str) -> str:
    """Accept only a whole-response JSON fence, never extract JSON from prose."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", stripped,
                          flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def _validation_errors(text: str) -> list[str]:
    """Diagnostics only: never recover a verdict from malformed output."""
    if not text.strip():
        return ["empty_response"]
    try:
        parsed = json.loads(_json_payload(text))
    except json.JSONDecodeError as exc:
        return [f"invalid_json: {exc.msg} at line {exc.lineno}, column {exc.colno}"]
    if not isinstance(parsed, dict):
        return ["expected_json_object"]
    errors = []
    for name in ("verdict", "reason", *EFFECT_FIELDS, "effect_matches_authorization"):
        if name not in parsed:
            errors.append(f"missing_field: {name}")
        elif name == "verdict":
            if parsed[name] not in ("APPROVE", "DENY"):
                errors.append("invalid_verdict")
        elif name == "effect_matches_authorization":
            if type(parsed[name]) is not bool:
                errors.append(f"expected_boolean: {name}")
        elif not isinstance(parsed[name], str):
            errors.append(f"expected_string: {name}")
        elif name in EFFECT_FIELDS and not parsed[name].strip():
            errors.append(f"empty_field: {name}")
    return errors


def _parse(text: str, strict: bool = False) -> tuple[str, str, bool]:
    """Extract the verdict. An unreadable reply counts as DENY, and the
    reason says so, so parse failures can be counted in the log."""
    if strict:
        try:
            parsed = json.loads(_json_payload(text))
            if (isinstance(parsed, dict) and parsed.get("verdict") in ("APPROVE", "DENY")
                    and isinstance(parsed.get("reason"), str)
                    and all(isinstance(parsed.get(k), str) and parsed[k].strip()
                            for k in EFFECT_FIELDS)
                    and type(parsed.get("effect_matches_authorization")) is bool):
                if (parsed["verdict"] == "APPROVE"
                        and not parsed["effect_matches_authorization"]):
                    return "DENY", "reviewer effect check does not authorize the action", False
                return parsed["verdict"], parsed["reason"][:200], False
        except (json.JSONDecodeError, TypeError):
            pass
        return "DENY", "reviewer output could not be parsed; failing closed", True
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                return "DENY", "reviewer output is not an object; failing closed", True
            verdict = str(parsed.get("verdict", "")).strip().upper()
            if verdict in ("APPROVE", "DENY"):
                return verdict, str(parsed.get("reason", ""))[:200], False
        except json.JSONDecodeError:
            pass
    upper = text.upper()
    if "APPROVE" in upper and "DENY" not in upper:
        return "APPROVE", "recovered from unstructured output", False
    return "DENY", "reviewer output could not be parsed; failing closed", True


class SimulatedReviewer:
    def __init__(self, client, model: str, style: str = "sanitized",
                 mode: str = "task_scoped", tool_schemas: dict | None = None) -> None:
        # client comes from llm.build_client, which already pins temperature.
        self.client = client
        self.model = model
        self.style = style
        if mode not in REVIEWER_MODES:
            raise ValueError(f"unknown reviewer mode: {mode}")
        self.mode = mode
        self.tool_schemas = tool_schemas or {}

    def build_messages(self, ctx: CallContext) -> list[dict]:
        if self.mode == "task_scoped":
            info = classify(ctx.tool_call.function)
            metadata = {"tool": ctx.tool_call.function, "tier": str(info.tier),
                        "schema": self.tool_schemas.get(ctx.tool_call.function)}
            return [
                {"role": "system", "content": TASK_SCOPED_SYSTEM},
                {"role": "user", "content": TASK_SCOPED_TEMPLATE.format(
                    user_prompt=ctx.user_prompt,
                    tool_metadata=json.dumps(metadata),
                    evidence=_spotlight(json.dumps(_evidence(ctx), default=str), self.style),
                )},
            ]
        user_message = USER_TEMPLATE.format(
            user_prompt=ctx.user_prompt,
            prior_actions=_prior_actions(ctx.messages, style=self.style),
            tool_name=ctx.tool_call.function,
            tool_args=_spotlight(str(dict(ctx.tool_call.args)), self.style),
        )
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message}]

    def review(self, ctx: CallContext) -> ReviewResult:
        started = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(ctx),
        )
        text = response.choices[0].message.content or ""
        verdict, reason, parse_failed = _parse(text, strict=self.mode == "task_scoped")
        assessment = {}
        if self.mode == "task_scoped" and not parse_failed:
            parsed = json.loads(_json_payload(text))
            assessment = {k: parsed[k] for k in ("scope", "evidence", *EFFECT_FIELDS,
                                                   "effect_matches_authorization") if k in parsed}
        usage = response.usage

        return ReviewResult(
            verdict=verdict,
            reason=reason,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            parse_failed=parse_failed,
            assessment=assessment,
            raw_response=text,
            validation_errors=(_validation_errors(text) if self.mode == "task_scoped"
                               else ["legacy_parse_failure"] if parse_failed else []),
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )
