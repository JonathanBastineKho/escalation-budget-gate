"""Node 3: combines taint and envelope fit into R, a risk probability.

The weights are provisional; they get fitted on a development split.
"""

from __future__ import annotations

from dataclasses import dataclass

from approval_gate.nodes.arguments import appears_in, destination_args, flatten, normalize_text
from approval_gate.nodes.envelope import TaskEnvelope
from approval_gate.nodes.taint import AGENT, DELEGATED, TOOL_OUTPUT, UNKNOWN, USER, taint_flags

TAINT_WEIGHT = {USER: 0.0, DELEGATED: 0.2, AGENT: 0.15, UNKNOWN: 0.3, TOOL_OUTPUT: 0.45}
DEVIATION_WEIGHT = {"tool": 0.3, "entity": 0.3, "amount": 0.1, "count": 0.3}


@dataclass
class RiskFeatures:
    R: float
    deviation: float
    taint: dict[str, str]
    taint_critical: bool


def assess(ctx, envelope: TaskEnvelope, writes_done: int) -> RiskFeatures:
    args = dict(ctx.tool_call.args)
    taint = taint_flags(ctx)
    watched = set(destination_args(args)) | ({"password"} & set(args))
    flags = [taint[n] for n in watched if n in taint]

    taint_term = max((TAINT_WEIGHT[f] for f in flags), default=0.0)
    taint_critical = TOOL_OUTPUT in flags

    entities = normalize_text(" ".join(envelope.entities))
    unexpected_destination = any(
        taint.get(n) not in (USER, DELEGATED)
        and not all(appears_in(v, entities) for v in flatten(args[n]))
        for n in watched
    )
    try:
        amount = float(args["amount"]) if "amount" in args else None
    except (TypeError, ValueError):
        amount = None

    d = {
        "tool": 0.5 if envelope.expected_tools is None
        else float(ctx.tool_call.function not in envelope.expected_tools),
        "entity": float(unexpected_destination),
        "amount": float(amount is not None and envelope.max_amount is not None
                        and amount > envelope.max_amount),
        "count": float(writes_done >= envelope.expected_writes),
    }
    deviation = sum(DEVIATION_WEIGHT[k] * v for k, v in d.items())
    R = min(0.95, 0.05 + taint_term + 0.5 * deviation)
    return RiskFeatures(R=R, deviation=deviation, taint=taint, taint_critical=taint_critical)
