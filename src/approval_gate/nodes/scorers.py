"""Node 3 scorers: turn RiskFeatures into R, the probability the call is unauthorized."""

from __future__ import annotations

from approval_gate.nodes.risk import RiskFeatures
from approval_gate.settings import POLICY


class FormulaScorer:
    """Hand-set weights from config/policy.yaml."""

    name = "formula"

    def __init__(self, weights: dict | None = None) -> None:
        self.w = weights or POLICY["formula"]

    def score(self, f: RiskFeatures) -> float:
        w = self.w
        z = (
            w["base"]
            + w["taint"][f.origin]
            + w["tool_unexpected"] * f.tool_unexpected
            + w["destination_unnamed"] * f.destination_unnamed
            + w["over_expected_writes"] * f.over_expected_writes
            + w["dest_denied_before"] * f.dest_denied_before
            + w["dest_approved_before"] * f.dest_approved_before
            + w["per_denial"] * min(f.denials_so_far, w["denial_cap"])
        )
        return min(max(z, w["r_min"]), w["r_max"])
