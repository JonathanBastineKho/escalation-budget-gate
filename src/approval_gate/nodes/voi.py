"""Node 4: the value of asking a human, and the bar it must clear given the budget."""

from __future__ import annotations

import math
from dataclasses import dataclass

from approval_gate.nodes.allowlist import Tier
from approval_gate.settings import POLICY

_V = POLICY["voi"]
BLOCK_COST = {
    Tier.WRITE: _V["block_cost"]["write"],
    Tier.IRREVERSIBLE_EXTERNAL: _V["block_cost"]["irreversible_external"],
}
TAU_LOW = _V["bar_floor"]
VOI_MAX = _V["bar_ceiling"]
STATIC_BAR = _V["static_bar"]


@dataclass
class VoiResult:
    loss_approve: float
    loss_deny: float
    voi: float
    auto_approve: bool


def assess_voi(R: float, H: float, deviation: float, tier: Tier) -> VoiResult:
    block_cost = BLOCK_COST.get(tier, 0.9) * (1.0 - deviation)
    loss_approve = R * H
    loss_deny = (1.0 - R) * block_cost
    return VoiResult(
        loss_approve=loss_approve,
        loss_deny=loss_deny,
        voi=min(loss_approve, loss_deny),
        auto_approve=loss_approve <= loss_deny,
    )


def escalation_bar(k_remaining: int | None, n_remaining: int) -> float:
    if k_remaining is None:
        return TAU_LOW
    if k_remaining <= 0:
        return math.inf
    share = min(1.0, k_remaining / max(1, n_remaining))
    return max(TAU_LOW, VOI_MAX * (1.0 - share))
