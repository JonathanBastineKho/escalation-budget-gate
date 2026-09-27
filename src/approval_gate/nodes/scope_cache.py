"""Node 2: per-task memory of decisions already made.

Approvals match the exact call; denials match the destination.
"""

from __future__ import annotations

from agentdojo.functions_runtime import FunctionCall

from approval_gate.decisions import Decision
from approval_gate.nodes.arguments import destination_args, normalize


def _exact_key(call: FunctionCall) -> tuple:
    args = dict(call.args)
    return (call.function, tuple(sorted((k, normalize(v)) for k, v in args.items())))


def _destination_key(call: FunctionCall) -> tuple:
    dest = destination_args(dict(call.args))
    return (call.function, tuple(sorted((k, normalize(v)) for k, v in dest.items())))


class ScopeCache:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._approved: set[tuple] = set()
        self._denied: set[tuple] = set()

    def lookup(self, call: FunctionCall) -> tuple[Decision, str] | None:
        if _destination_key(call) in self._denied:
            return Decision.AUTO_DENY, "cache_neg"
        if _exact_key(call) in self._approved:
            return Decision.AUTO_APPROVE, "cache_pos"
        return None

    def record(self, call: FunctionCall, approved: bool) -> None:
        if approved:
            self._approved.add(_exact_key(call))
        else:
            self._denied.add(_destination_key(call))
