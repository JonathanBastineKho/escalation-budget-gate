"""The gate: owns the budget k, resolves escalations, writes the log.
Policies only decide; the gate enforces, so a policy cannot overspend."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agentdojo.functions_runtime import FunctionCall

from approval_gate.decisions import Decision, DecisionRecord, RunMeta, hash_args
from approval_gate.nodes.allowlist import classify
from approval_gate.policies.base import CallContext, Policy
from approval_gate.reviewer.context import review_packet

GENERIC_DENIAL = "Tool call denied by approval policy."


@dataclass
class GateOutcome:
    approved: bool
    denial_message: str | None
    record: DecisionRecord


class ApprovalGate:
    def __init__(
        self,
        policy: Policy,
        meta: RunMeta,
        logger=None,
        reviewer=None,
        denial_style: str = "generic",
        prompt_to_task_id: dict[str, str] | None = None,
    ) -> None:
        if meta.k_budget is not None and meta.k_budget < 0:
            raise ValueError("k must be nonnegative")
        self.policy = policy
        self.meta = meta
        self.logger = logger
        self.reviewer = reviewer
        self.denial_style = denial_style
        self.prompt_to_task_id = prompt_to_task_id or {}

        self.task_id = "unknown"
        self.user_prompt = ""
        self.k_remaining: int | None = meta.k_budget
        self.step_index = 0
        self.records = []
        self.executions = []
        self.force_decision = None

    def start_task(self, user_prompt: str, task_id: str | None = None) -> None:
        """Resets per-task state so budget and cached decisions never carry
        over between tasks."""
        self.user_prompt = user_prompt
        self.task_id = task_id or self.prompt_to_task_id.get(user_prompt, "unknown")
        self.k_remaining = self.meta.k_budget
        self.step_index = 0
        self.records = []
        self.executions = []
        self.force_decision = None
        self.policy.start_task(self.task_id, user_prompt)

    def _spend_budget(self) -> bool:
        if self.k_remaining is None:
            return True
        if self.k_remaining <= 0:
            return False
        self.k_remaining -= 1
        return True

    def decide(self, tool_call: FunctionCall, messages, tool_schema=None) -> GateOutcome:
        started = time.perf_counter()
        k_before = self.k_remaining

        ctx = CallContext(
            task_id=self.task_id,
            user_prompt=self.user_prompt,
            tool_call=tool_call,
            messages=messages,
            step_index=self.step_index,
            k_remaining=self.k_remaining,
            tool_schema=tool_schema or {},
        )
        if self.force_decision is not None:
            decision, node = self.force_decision, "counterfactual_override"
            self.force_decision = None
        else:
            decision, node = self.policy.decide(ctx)

        reviewer_verdict = None
        reviewer_reason = None
        parse_failed = False
        tokens_in: dict[str, int | None] = {}
        tokens_out: dict[str, int | None] = {}

        if decision is Decision.ESCALATE:
            if not self._spend_budget():
                # Fail closed: once k is used up, escalations become denials.
                approved = False
                node = "budget_exhausted"
            elif self.reviewer is None:
                raise RuntimeError("policy escalated but no reviewer is configured")
            else:
                if hasattr(self.logger, "emit"):
                    self.logger.emit(
                        "reviews",
                        step_index=self.step_index,
                        call_id=tool_call.id,
                        event="consumed",
                    )
                result = self.reviewer.review(ctx)
                reviewer_verdict = result.verdict
                reviewer_reason = result.reason
                parse_failed = result.parse_failed
                approved = result.verdict == "APPROVE"
                node = "reviewer"
                tokens_in["reviewer"] = result.tokens_in
                tokens_out["reviewer"] = result.tokens_out
        else:
            approved = decision is Decision.AUTO_APPROVE

        # Logged for every policy, so baselines can be grouped by tier too.
        tool_info = classify(tool_call.function)
        record = DecisionRecord(
            run_id=self.meta.run_id,
            task_id=self.task_id,
            suite=self.meta.suite,
            policy=self.meta.policy,
            k_budget=self.meta.k_budget,
            seed=self.meta.seed,
            model_agent=self.meta.model_agent,
            model_gate=self.meta.model_gate,
            attack_name=self.meta.attack_name,
            step_index=self.step_index,
            tool_name=tool_call.function,
            call_id=tool_call.id or str(self.step_index),
            approved=approved,
            reviewer_parse_failed=parse_failed,
            packet=review_packet(ctx),
            features=getattr(self.policy, "last_details", {}).get("features", {}),
            policy_details=getattr(self.policy, "last_details", {}),
            risk_R=getattr(self.policy, "last_details", {}).get("risk"),
            voi=getattr(self.policy, "last_details", {}).get("voi"),
            tool_tier=str(tool_info.tier),
            harm_H=tool_info.harm,
            args_hash=hash_args(dict(tool_call.args)),
            critical_args=dict(tool_call.args),
            k_remaining_before=k_before,
            decision=str(decision),
            decided_by_node=node,
            reviewer_verdict=reviewer_verdict,
            reviewer_reason=reviewer_reason,
            executed=False,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if self.logger is not None:
            self.logger.log(record)
        self.records.append(record)

        self.step_index += 1
        return GateOutcome(
            approved=approved,
            denial_message=None if approved else self._denial_message(reviewer_reason),
            record=record,
        )

    def _denial_message(self, reason: str | None) -> str:
        """A denial is returned to the agent as a tool error, never an abort,
        so the task can continue."""
        if self.denial_style == "generic" or not reason:
            return GENERIC_DENIAL
        return f"{GENERIC_DENIAL} Reason: {reason}"
