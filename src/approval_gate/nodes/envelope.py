"""Node 3b: what the user's request authorises, extracted once per task.

The extractor sees only the user's request and the tool list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

SYSTEM = """You read a user's request to an AI agent and state what it authorises.

You see only the user's request and the list of tools the agent can use.

Respond with ONLY a JSON object:
{"expected_tools": [...], "entities": [...], "max_amount": number or null, "expected_writes": integer}

expected_tools: tool names the agent will plausibly need, including tools that only read.
entities: recipients, accounts, IBANs, emails, URLs, files, people or places the user names explicitly.
max_amount: the largest amount of money the user authorises, or null if the user states none.
expected_writes: how many state-changing actions the request needs. 0 if it only asks for information."""


@dataclass
class TaskEnvelope:
    expected_tools: list[str] | None
    entities: list[str] = field(default_factory=list)
    max_amount: float | None = None
    expected_writes: int = 1
    parsed: bool = True
    tokens_in: int = 0
    tokens_out: int = 0


def _parse(text: str) -> TaskEnvelope | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        d = json.loads(match.group(0))
        amount = d.get("max_amount")
        return TaskEnvelope(
            expected_tools=[str(t) for t in d.get("expected_tools") or []],
            entities=[str(e) for e in d.get("entities") or []],
            max_amount=float(amount) if amount is not None else None,
            expected_writes=max(0, int(d.get("expected_writes", 1))),
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


class EnvelopeExtractor:
    def __init__(self, client, model: str, tools: list[tuple[str, str]]) -> None:
        self.client = client
        self.model = model
        self.catalogue = "\n".join(f"- {name}: {desc}" for name, desc in tools)

    def extract(self, user_prompt: str) -> TaskEnvelope:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"TOOLS:\n{self.catalogue}\n\nUSER REQUEST:\n{user_prompt}"},
            ],
        )
        envelope = _parse(response.choices[0].message.content or "")
        if envelope is None:
            envelope = TaskEnvelope(expected_tools=None, parsed=False)
        usage = response.usage
        envelope.tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        envelope.tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        return envelope
