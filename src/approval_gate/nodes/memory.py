"""Node 2: task memory. Remembers every write proposed in the current task.

Only a reviewer's verdict counts as evidence; the policy's own decisions and
budget-exhaustion denials say nothing about the call.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentdojo.functions_runtime import FunctionCall

from approval_gate.nodes.arguments import destination_args, flatten, normalize

REVIEWER = "reviewer"


def _exact_key(call: FunctionCall) -> tuple:
    args = dict(call.args)
    return (call.function, tuple(sorted((k, normalize(v)) for k, v in args.items())))


def _values(destinations: dict) -> frozenset[str]:
    return frozenset(normalize(v) for value in destinations.values() for v in flatten(value))


@dataclass(frozen=True)
class Entry:
    exact: tuple
    destinations: frozenset[str]
    approved: bool
    decided_by: str


@dataclass(frozen=True)
class MemoryFacts:
    dest_denied_before: bool
    dest_approved_before: bool
    denials_so_far: int


class TaskMemory:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.entries: list[Entry] = []

    @property
    def proposed(self) -> int:
        return len(self.entries)

    @property
    def executed(self) -> int:
        return sum(e.approved for e in self.entries)

    def denied_repeat(self, call: FunctionCall) -> bool:
        """Approvals are never reused: repeating an approved write would run it twice."""
        key = _exact_key(call)
        return any(
            e.exact == key and e.decided_by == REVIEWER and not e.approved for e in self.entries
        )

    def record(self, call: FunctionCall, approved: bool, decided_by: str) -> None:
        dest = destination_args(call.function, dict(call.args))
        self.entries.append(Entry(_exact_key(call), _values(dest), approved, decided_by))

    def facts(self, destinations: dict) -> MemoryFacts:
        values = _values(destinations)
        reviewed = [e for e in self.entries if e.decided_by == REVIEWER]
        return MemoryFacts(
            dest_denied_before=any(not e.approved and e.destinations & values for e in reviewed),
            dest_approved_before=any(e.approved and e.destinations & values for e in reviewed),
            denials_so_far=sum(not e.approved for e in reviewed),
        )
