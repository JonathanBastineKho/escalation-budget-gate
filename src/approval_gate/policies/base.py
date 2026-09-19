"""The Policy interface shared by the baselines and the cascade."""

from __future__ import annotations

import abc
from dataclasses import dataclass
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


class Policy(abc.ABC):
    name: str

    def start_task(self, task_id: str, user_prompt: str) -> None:
        """Reset any per-task state."""

    @abc.abstractmethod
    def decide(self, ctx: CallContext) -> tuple[Decision, str]:
        """Return (decision, name of the node that decided)."""
