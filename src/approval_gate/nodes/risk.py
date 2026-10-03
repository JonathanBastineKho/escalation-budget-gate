"""Node 3: the features of a write call. A scorer turns them into R."""

from __future__ import annotations

from dataclasses import dataclass

from approval_gate.nodes.arguments import appears_in, destination_args, flatten, normalize_text
from approval_gate.nodes.envelope import TaskEnvelope
from approval_gate.nodes.memory import TaskMemory
from approval_gate.nodes.taint import DELEGATED, RANK, TOOL_OUTPUT, USER, taint_flags


@dataclass
class RiskFeatures:
    taint: dict[str, str]           # origin of every labelled argument
    origin: str                     # worst destination origin, after the delegation scope check
    tool_unexpected: float          # 0.5 when the envelope could not be parsed
    destination_unnamed: float
    over_expected_writes: float
    dest_denied_before: float
    dest_approved_before: float
    denials_so_far: int

    @property
    def deviation(self) -> float:
        return (self.tool_unexpected + self.destination_unnamed + self.over_expected_writes) / 3


def extract(ctx, envelope: TaskEnvelope, memory: TaskMemory) -> RiskFeatures:
    tool = ctx.tool_call.function
    args = dict(ctx.tool_call.args)
    taint = taint_flags(ctx)
    destinations = destination_args(tool, args)

    tool_expected = envelope.expected_tools is None or tool in envelope.expected_tools
    over_expected = memory.proposed >= envelope.expected_writes
    in_scope = tool_expected and memory.executed < envelope.expected_writes
    labels = {
        n: TOOL_OUTPUT if taint[n] == DELEGATED and not in_scope else taint[n]
        for n in destinations
    }
    origin = max(labels.values(), key=RANK.__getitem__, default=USER)

    entities = normalize_text(" ".join(envelope.entities))
    unnamed = any(
        label not in (USER, DELEGATED)
        and not all(appears_in(v, entities) for v in flatten(args[n]))
        for n, label in labels.items()
    )
    facts = memory.facts(destinations)

    not_user = {n: destinations[n] for n, label in labels.items() if label != USER}
    denied_before = memory.facts(not_user).dest_denied_before
    return RiskFeatures(
        taint=taint,
        origin=origin,
        tool_unexpected=0.5 if envelope.expected_tools is None else float(not tool_expected),
        destination_unnamed=float(unnamed),
        over_expected_writes=float(over_expected),
        dest_denied_before=float(denied_before),
        dest_approved_before=float(facts.dest_approved_before),
        denials_so_far=facts.denials_so_far,
    )
