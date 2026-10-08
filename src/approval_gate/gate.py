"""The gate: owns the budget k, resolves escalations, writes the log.
Policies only decide; the gate enforces, so a policy cannot overspend."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, replace

from agentdojo.functions_runtime import FunctionCall

from approval_gate.decisions import CaseRecord, Decision, DecisionRecord, RunMeta, hash_args
from approval_gate.llm import UsageMeter
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
        agent_meter: UsageMeter,
        logger=None,
        reviewer=None,
        denial_style: str = "generic",
    ) -> None:
        self.policy = policy
        self.meta = meta
        self.agent_meter = agent_meter
        self.logger = logger
        self.reviewer = reviewer
        self.denial_style = denial_style

        self.task_id = "unknown"
        self.injection_task_id: str | None = None
        self.user_prompt = ""
        self.k_remaining: int | None = meta.k_budget
        self.step_index = 0
        self._start_case_counters()

    def _start_case_counters(self) -> None:
        self._attempts = 0
        self._gated_calls = 0
        self._escalations = 0
        self._tokens_in: Counter[str] = Counter()
        self._tokens_out: Counter[str] = Counter()
        self._policy_costs: Counter[str] = Counter()
        # Snapshots of agent usage at case start, and at the last gated call.
        self._agent_case_start = replace(self.agent_meter)
        self._agent_mark = replace(self.agent_meter)

    def start_case(self, user_task_id: str, injection_task_id: str | None) -> None:
        """Called once per AgentDojo case, before its first query(). The budget
        resets on the case's first start_task, so retries share one budget."""
        self.task_id = user_task_id
        self.injection_task_id = injection_task_id
        self._start_case_counters()

    def end_case(self, utility: bool | None, security: bool | None, error: str | None) -> None:
        # Whole-case agent usage, including the final turn that proposed no call.
        start = self._agent_case_start
        tokens_in = {**self._tokens_in, "agent": self.agent_meter.tokens_in - start.tokens_in}
        tokens_out = {**self._tokens_out, "agent": self.agent_meter.tokens_out - start.tokens_out}
        record = CaseRecord(
            run_id=self.meta.run_id,
            task_id=self.task_id,
            injection_task_id=self.injection_task_id,
            suite=self.meta.suite,
            policy=self.meta.policy,
            k_budget=self.meta.k_budget,
            seed=self.meta.seed,
            attack_name=self.meta.attack_name,
            attempts=self._attempts,
            gated_calls=self._gated_calls,
            escalations=self._escalations,
            utility=utility,
            security=None if self.injection_task_id is None else security,
            error=error,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            config_hash=self.meta.config_hash,
            comparison_id=self.meta.comparison_id,
            model_agent=self.meta.model_agent,
            model_gate=self.meta.model_gate,
            model_reviewer=self.meta.model_reviewer,
            **self._policy_costs,
        )
        if self.logger is not None:
            self.logger.log_case(record)

    def start_task(self, user_prompt: str) -> None:
        """Called on every attempt. AgentDojo retries a case when the agent ends
        without text, keeping the environment, so only the first attempt resets
        the budget and the policy's memory."""
        self._attempts += 1
        if self._attempts > 1:
            return
        self.user_prompt = user_prompt
        self.k_remaining = self.meta.k_budget
        self.step_index = 0
        self.policy.start_task(self.task_id, user_prompt)

    def _agent_tokens_since_last_call(self) -> tuple[int, int]:
        """Agent tokens spent since the previous gated call: the model turn that
        proposed this call. Zero for later calls in the same turn."""
        delta = (
            self.agent_meter.tokens_in - self._agent_mark.tokens_in,
            self.agent_meter.tokens_out - self._agent_mark.tokens_out,
        )
        self._agent_mark = replace(self.agent_meter)
        return delta

    def _spend_budget(self) -> bool:
        if self.k_remaining is None:
            return True
        if self.k_remaining <= 0:
            return False
        self.k_remaining -= 1
        return True

    def decide(self, tool_call: FunctionCall, messages, *, preview=None) -> GateOutcome:
        started = time.perf_counter()
        k_before = self.k_remaining

        ctx = CallContext(
            task_id=self.task_id,
            user_prompt=self.user_prompt,
            tool_call=tool_call,
            messages=messages,
            step_index=self.step_index,
            k_remaining=self.k_remaining,
            preview=preview,
        )
        decision, node = self.policy.decide(ctx)

        reviewer_verdict = None
        reviewer_reason = None
        reviewer_error = None
        reviewer_usage_missing = 0
        tokens_in: dict[str, int] = {}
        tokens_out: dict[str, int] = {}
        agent_in, agent_out = self._agent_tokens_since_last_call()
        if agent_in or agent_out:
            tokens_in["agent"] = agent_in
            tokens_out["agent"] = agent_out

        if decision is Decision.ESCALATE:
            if not self._spend_budget():
                # Fail closed: once k is used up, escalations become denials.
                approved = False
                node = "budget_exhausted"
            elif self.reviewer is None:
                raise RuntimeError("policy escalated but no reviewer is configured")
            else:
                node = "reviewer"
                try:
                    result = self.reviewer.review(ctx)
                except Exception as exc:
                    # The intervention was attempted and still costs budget.
                    # Keep the judge/preview accounting even if review fails.
                    approved = False
                    reviewer_verdict = "DENY"
                    reviewer_reason = "Reviewer unavailable; failing closed"
                    reviewer_error = type(exc).__name__
                    reviewer_usage_missing = 1
                else:
                    reviewer_verdict = result.verdict
                    reviewer_reason = result.reason
                    approved = result.verdict == "APPROVE"
                    reviewer_error = "parse_failed" if getattr(result, "parse_failed", False) else None
                    reviewer_usage_missing = int(not getattr(result, "usage_known", True))
                    tokens_in["reviewer"] = result.tokens_in
                    tokens_out["reviewer"] = result.tokens_out
        else:
            approved = decision is Decision.AUTO_APPROVE

        self.policy.observe(ctx, approved, node)
        for component, n in ctx.features.pop("tokens_in", {}).items():
            tokens_in[component] = tokens_in.get(component, 0) + n
        for component, n in ctx.features.pop("tokens_out", {}).items():
            tokens_out[component] = tokens_out.get(component, 0) + n

        # Logged for every policy, so baselines can be grouped by tier too.
        tool_info = classify(tool_call.function)
        record = DecisionRecord(
            run_id=self.meta.run_id,
            task_id=self.task_id,
            injection_task_id=self.injection_task_id,
            suite=self.meta.suite,
            policy=self.meta.policy,
            k_budget=self.meta.k_budget,
            seed=self.meta.seed,
            model_agent=self.meta.model_agent,
            model_gate=self.meta.model_gate,
            model_reviewer=self.meta.model_reviewer,
            attack_name=self.meta.attack_name,
            step_index=self.step_index,
            tool_name=tool_call.function,
            tool_call_id=tool_call.id,
            tool_tier=str(tool_info.tier),
            harm_H=tool_info.harm,
            args_hash=hash_args(dict(tool_call.args)),
            critical_args=dict(tool_call.args),
            k_remaining_before=k_before,
            decision=str(decision),
            decided_by_node=node,
            reviewer_verdict=reviewer_verdict,
            reviewer_reason=reviewer_reason,
            reviewer_error=reviewer_error,
            reviewer_usage_missing=reviewer_usage_missing,
            executed=approved,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000
            + (preview.latency_ms if preview is not None else 0.0),
            config_hash=self.meta.config_hash,
            comparison_id=self.meta.comparison_id,
            speculative_used=preview is not None and preview.calls > 0,
            preview_status=preview.status if preview is not None else None,
            preview_observation=preview.observation if preview is not None else None,
            preview_error=preview.error if preview is not None else None,
            preview_calls=preview.calls if preview is not None else 0,
            preview_latency_ms=preview.latency_ms if preview is not None else 0.0,
            preview_effects=preview.effects if preview is not None else None,
        )
        for key, value in ctx.features.items():
            if hasattr(record, key):
                setattr(record, key, value)
        if self.logger is not None and not approved:
            self.logger.log(record)

        self._gated_calls += 1
        if node == "reviewer":
            self._escalations += 1
        self._policy_costs.update({
            "judge_calls": record.judge_calls,
            "judge_unavailable": int(record.judge_status not in (None, "ok")),
            "judge_usage_missing": record.judge_usage_missing,
            "judge_latency_ms": record.judge_latency_ms,
            "preview_calls": record.preview_calls,
            "preview_failures": int(record.preview_status not in (None, "ok")),
            "preview_latency_ms": record.preview_latency_ms,
            "gate_latency_ms": record.latency_ms,
            "reviewer_failures": int(record.reviewer_error is not None),
            "reviewer_usage_missing": record.reviewer_usage_missing,
            "non_read_only_calls": int(record.tool_tier != "read_only"),
            "non_read_only_executed": int(record.tool_tier != "read_only" and approved),
        })
        # Agent totals come from the meter in end_case; other components add up here.
        self._tokens_in.update({c: n for c, n in tokens_in.items() if c != "agent"})
        self._tokens_out.update({c: n for c, n in tokens_out.items() if c != "agent"})
        self.step_index += 1
        return GateOutcome(
            approved=approved,
            denial_message=None if approved else self._denial_message(reviewer_reason),
            record=record,
        )

    def record_result(self, outcome: GateOutcome, error: str | None) -> None:
        """Called after an approved call has run."""
        outcome.record.tool_error = error
        if self.logger is not None:
            self.logger.log(outcome.record)

    def _denial_message(self, reason: str | None) -> str:
        """A denial is returned to the agent as a tool error, never an abort,
        so the task can continue."""
        if self.denial_style == "generic" or not reason:
            return GENERIC_DENIAL
        return f"{GENERIC_DENIAL} Reason: {reason}"
