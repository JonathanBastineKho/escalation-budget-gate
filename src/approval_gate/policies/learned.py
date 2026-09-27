"""Prompted, calibrated risk, and learned downstream-value policies."""

from __future__ import annotations
import json
from pathlib import Path
from approval_gate.decisions import Decision
from approval_gate.nodes.llm_features import FEATURES, vector
from approval_gate.policies.base import Policy


def predict(head, values):
    import math

    z = head["bias"] + sum(
        w * (x - m) / s
        for w, x, m, s in zip(head["weights"], values, head["mean"], head["scale"], strict=True)
    )
    return 1 / (1 + math.exp(-max(-40, min(40, z))))


class SemanticPolicy(Policy):
    def __init__(self, extractor, kind="llm_judge", checkpoint=None):
        self.extractor, self.name = extractor, kind
        self.last_details = {}
        self.checkpoint = json.loads(Path(checkpoint).read_text()) if checkpoint else None
        if kind != "llm_judge":
            if self.checkpoint is None or self.checkpoint["kind"] != kind:
                raise ValueError(f"{kind} requires a matching checkpoint")
            if self.checkpoint["features"] != FEATURES:
                raise ValueError("checkpoint feature schema mismatch")
            if self.checkpoint["ablation"] != extractor.ablation:
                raise ValueError("checkpoint ablation mismatch; retrain the ablation")
            if self.checkpoint["policy_model"] != extractor.model:
                raise ValueError("checkpoint policy-model mismatch")
            if self.checkpoint.get("feature_signature") != getattr(extractor, "signature", None):
                raise ValueError(
                    "feature prompt/model digest/inference settings changed; recollect features"
                )

    def start_task(self, task_id, user_prompt):
        self.last_details = {}

    def decide(self, ctx):
        self.last_details = self.extractor.extract(ctx)
        self.last_details["feature_signature"] = getattr(self.extractor, "signature", None)
        if self.last_details["parse_failed"]:
            return Decision.ESCALATE, "feature_parse_failure"
        if self.name == "llm_judge":
            return Decision(self.last_details["suggested_decision"]), "llm_judge"
        x = vector(self.last_details["features"], self.extractor.ablation)
        checkpoint = self.checkpoint
        if self.name == "learned_risk":
            risk = predict(checkpoint["heads"]["risk"], x)
            if checkpoint.get("calibrator"):
                risk = predict(checkpoint["calibrator"], [risk])
            self.last_details["risk"] = risk
            if risk <= checkpoint["approve_threshold"]:
                return Decision.AUTO_APPROVE, "learned_risk"
            if risk >= checkpoint["deny_threshold"]:
                return Decision.AUTO_DENY, "learned_risk"
            return Decision.ESCALATE, "learned_risk"
        values = {a: predict(h, x) for a, h in checkpoint["heads"].items()}
        guard = checkpoint["risk_guard"]
        guard_x = vector(self.last_details["features"], guard["ablation"])
        risk = predict(guard["heads"]["risk"], guard_x)
        if guard.get("calibrator"):
            risk = predict(guard["calibrator"], [risk])
        self.last_details["risk"] = risk
        if risk > guard["approve_threshold"]:
            values.pop("AUTO_APPROVE", None)
        if ctx.k_remaining == 0:
            values.pop("ESCALATE", None)
        self.last_details["action_values"] = values
        if "ESCALATE" in values:
            self.last_details["voi"] = values["ESCALATE"] - max(
                value for action, value in values.items() if action != "ESCALATE"
            )
        # Conservative deterministic tie-breaking: denial, escalation, approval.
        preference = {"AUTO_DENY": 2, "ESCALATE": 1, "AUTO_APPROVE": 0}
        action = max(values, key=lambda a: (values[a], preference[a]))
        return Decision(action), "learned_voi"
