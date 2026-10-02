"""The Policy interface shared by the baselines and the cascade."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from agentdojo.functions_runtime import FunctionCall

from approval_gate.decisions import Decision


@dataclass
class CallContext:
    """Everything a policy or the reviewer may see. Ground truth and attack
    labels are deliberately absent, so they cannot leak."""

    task_id: str
    user_prompt: str
    tool_call: FunctionCall
    messages: Any               # Sequence[ChatMessage] so far
    step_index: int
    k_remaining: int | None     # None means unbounded
    features: dict = field(default_factory=dict)   # policy -> log record


class Policy(abc.ABC):
    name: str

    def start_task(self, task_id: str, user_prompt: str) -> None:
        """Reset any per-task state."""

    def observe_execution(self, tool_call: FunctionCall, error: str | None) -> None:
        """Observe actual tool completion, not just approval. Default: no state."""

    def take_usage(self) -> tuple[dict, dict]:
        """Drain policy inference token counts for this decision."""
        return {}, {}

    def decision_details(self) -> dict:
        """Policy-specific audit information without benchmark ground truth."""
        return {}

    @abc.abstractmethod
    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        """Return (decision, name of the node that decided)."""

    def observe(self, ctx: CallContext, approved: bool, decided_by: str) -> None:
        """Told the final outcome, after any escalation has been resolved."""
