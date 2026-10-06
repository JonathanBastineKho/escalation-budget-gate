"""Node 3: the features of a write call. A scorer turns them into R."""

from __future__ import annotations

from dataclasses import dataclass

from approval_gate.nodes.arguments import (
    TARGET,
    VALUE,
    appears_in,
    args_with_role,
    destination_args,
    flatten,
    kind,
    kinds_in,
    normalize_text,
)
from approval_gate.nodes.binding import canon
from approval_gate.nodes.envelope import TaskEnvelope
from approval_gate.nodes.memory import TaskMemory
from approval_gate.nodes.taint import DELEGATED, RANK, TOOL_OUTPUT, USER, taint_flags


@dataclass
class RiskFeatures:
    taint: dict[str, str]           # origin of every labelled argument
    origin: str                     # worst destination origin, after the delegation checks
    tool_unexpected: float          # 0.5 when the envelope could not be parsed
    destination_unnamed: float
    over_expected_writes: float
    value_from_tool_output: float   # e.g. an amount copied from something the agent read
    target_unbound: float           # a target not reachable from what the user named
    dest_denied_before: float
    dest_approved_before: float
    denials_so_far: int

    @property
    def deviation(self) -> float:
        return (self.tool_unexpected + self.destination_unnamed + self.over_expected_writes) / 3


def _delegated_ok(value, user_kinds: set[str], prompt: str) -> bool:
    """A document the user pointed to cannot replace a recipient the user named."""
    k = kind(value)
    return k is None or k not in user_kinds or appears_in(value, prompt)


def extract(ctx, envelope: TaskEnvelope, memory: TaskMemory, reachable: set[str]) -> RiskFeatures:
    tool = ctx.tool_call.function
    args = dict(ctx.tool_call.args)
    taint = taint_flags(ctx)
    prompt = normalize_text(ctx.user_prompt)
    user_kinds = kinds_in(ctx.user_prompt)

    tool_expected = envelope.expected_tools is None or tool in envelope.expected_tools
    over_expected = memory.proposed >= envelope.expected_writes
    # A document the user pointed to only speaks for the job the user gave, until that job
    # is done. Denied attempts did not do it, so this counts executed writes.
    in_scope = tool_expected and memory.executed < envelope.expected_writes

    def label(name: str) -> str:
        if taint.get(name) != DELEGATED:
            return taint.get(name, USER)
        if in_scope and all(_delegated_ok(v, user_kinds, prompt) for v in flatten(args[name])):
            return DELEGATED
        return TOOL_OUTPUT

    destinations = destination_args(tool, args)
    labels = {n: label(n) for n in destinations}
    origin = max(labels.values(), key=RANK.__getitem__, default=USER)

    entities = normalize_text(" ".join(envelope.entities))
    unnamed = any(
        lab not in (USER, DELEGATED)
        and not all(appears_in(v, entities) for v in flatten(args[n]))
        for n, lab in labels.items()
    )
    value_untrusted = any(label(n) == TOOL_OUTPUT for n in args_with_role(tool, args, VALUE))
    target_unbound = any(
        not appears_in(v, prompt) and canon(v) not in reachable
        for n in args_with_role(tool, args, TARGET) for v in flatten(args[n])
    )

    facts = memory.facts(tool, destinations)
    # A destination the user named stays authorized after a denied call to it:
    # the denial may have been about something else, such as the amount.
    not_user = {n: destinations[n] for n, lab in labels.items() if lab != USER}
    denied_before = memory.facts(tool, not_user).dest_denied_before
    return RiskFeatures(
        taint=taint,
        origin=origin,
        tool_unexpected=0.5 if envelope.expected_tools is None else float(not tool_expected),
        destination_unnamed=float(unnamed),
        over_expected_writes=float(over_expected),
        value_from_tool_output=float(value_untrusted),
        target_unbound=float(target_unbound),
        dest_denied_before=float(denied_before),
        dest_approved_before=float(facts.dest_approved_before),
        denials_so_far=facts.denials_so_far,
    )
