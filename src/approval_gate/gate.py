"""The gate: owns the budget k, resolves escalations, writes the log.
Policies only decide; the gate enforces, so a policy cannot overspend."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agentdojo.functions_runtime import FunctionCall

from approval_gate.decisions import Decision, DecisionRecord, RunMeta, hash_args
from approval_gate.nodes.allowlist import classify
from approval_gate.policies.base import CallContext, Policy

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

    def start_task(self, user_prompt: str) -> None:
        """Resets per-task state so budget and cached decisions never carry
        over between tasks."""
        self.user_prompt = user_prompt
        self.task_id = self.prompt_to_task_id.get(user_prompt, "unknown")
        self.k_remaining = self.meta.k_budget
        self.step_index = 0
        self.policy.start_task(self.task_id, user_prompt)

    def _spend_budget(self) -> bool:
        if self.k_remaining is None:
            return True
        if self.k_remaining <= 0:
            return False
        self.k_remaining -= 1
        return True

    def decide(self, tool_call: FunctionCall, messages) -> GateOutcome:
        started = time.perf_counter()
        k_before = self.k_remaining

        ctx = CallContext(
            task_id=self.task_id,
            user_prompt=self.user_prompt,
            tool_call=tool_call,
            messages=messages,
            step_index=self.step_index,
            k_remaining=self.k_remaining,
        )
        decision, node = self.policy.decide(ctx)
        policy_node = node

        reviewer_verdict = None
        reviewer_reason = None
        reviewer_parse_failed = False
        reviewer_assessment = {}
        tokens_in, tokens_out = self.policy.take_usage()

        if decision is Decision.ESCALATE:
            if not self._spend_budget():
                # Fail closed: once k is used up, escalations become denials.
                approved = False
                node = "budget_exhausted"
            elif self.reviewer is None:
                raise RuntimeError("policy escalated but no reviewer is configured")
            else:
                result = self.reviewer.review(ctx)
                reviewer_verdict = result.verdict
                reviewer_reason = result.reason
                reviewer_parse_failed = getattr(result, "parse_failed", False)
                reviewer_assessment = getattr(result, "assessment", {})
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
            tool_tier=str(tool_info.tier),
            harm_H=tool_info.harm,
            args_hash=hash_args(dict(tool_call.args)),
            critical_args=dict(tool_call.args),
            k_remaining_before=k_before,
            decision=str(decision),
            decided_by_node=node,
            reviewer_verdict=reviewer_verdict,
            reviewer_reason=reviewer_reason,
            executed=approved,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
            policy_node=policy_node,
            policy_details=self.policy.decision_details(),
            model_policy=self.meta.model_policy,
            reviewer_mode=getattr(self.reviewer, "mode", None),
            reviewer_parse_failed=reviewer_parse_failed,
            reviewer_assessment=reviewer_assessment,
        )
        if self.logger is not None:
            self.logger.log(record)

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
