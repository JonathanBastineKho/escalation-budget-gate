"""Fallible online reviewer. Invalid verdicts deny and are recorded."""

from __future__ import annotations
import time
from dataclasses import dataclass
from approval_gate.reviewer.context import render_packet, review_packet
from approval_gate.reviewer.prompts import SYSTEM
from approval_gate.reviewer.json_output import loads


@dataclass
class ReviewResult:
    verdict: str
    reason: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float
    parse_failed: bool = False


def _parse(text: str) -> tuple[str, str, bool]:
    try:
        parsed = loads(text)
        if (
            isinstance(parsed, dict)
            and set(parsed) == {"verdict", "reason"}
            and parsed["verdict"] in ("APPROVE", "DENY")
            and isinstance(parsed["reason"], str)
        ):
            return parsed["verdict"], parsed["reason"][:1000], False
    except (ValueError, TypeError):
        pass
    return "DENY", "reviewer output could not be parsed; failing closed", True


class SimulatedReviewer:
    def __init__(self, client, model: str, style: str = "sanitized"):
        self.client, self.model, self.style = client, model, style

    def review(self, ctx) -> ReviewResult:
        return self.review_packet(review_packet(ctx))

    def review_packet(self, packet) -> ReviewResult:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": render_packet(packet, self.style)},
            ],
            response_format={"type": "json_object"},
        )
        verdict, reason, failed = _parse(response.choices[0].message.content or "")
        if response.choices[0].finish_reason == "length":
            verdict, reason, failed = "DENY", "truncated reviewer response", True
        usage = response.usage
        return ReviewResult(
            verdict,
            reason,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            (time.perf_counter() - started) * 1000,
            failed,
        )
