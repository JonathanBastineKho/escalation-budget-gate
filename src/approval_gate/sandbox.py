"""One-call preview for the pinned, in-memory AgentDojo simulator only.

This is not OS isolation for arbitrary Python/network tools. Only the stock
functions in the selected built-in suite are eligible. Live environment,
arguments, history, policy and review budget are never passed to a preview call.
The default returns only the normal tool observation. Optional audited adapters
also export affected records, never an environment dump.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
import json
import time

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from approval_gate.effects import effect_receipt, effect_snapshot


@dataclass(frozen=True)
class Preview:
    status: str
    observation: str | None = None
    error: str | None = None
    calls: int = 0
    latency_ms: float = 0.0
    effects: dict | None = None


def has_nested_call(value) -> bool:
    if isinstance(value, FunctionCall):
        return True
    if isinstance(value, dict):
        return any(has_nested_call(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_nested_call(v) for v in value)
    return False


class AgentDojoPreview:
    def __init__(
        self, suite_name: str, benchmark_version: str = "v1.2.1", max_chars: int = 16000,
        *, include_effects: bool = False,
    ) -> None:
        if version("agentdojo") != "0.1.35" or benchmark_version != "v1.2.1":
            raise ValueError("Preview adapter audited for AgentDojo 0.1.35 / v1.2.1 only")
        if suite_name not in {"banking", "workspace", "travel", "slack"}:
            raise ValueError("Preview supports only the four built-in simulator suites")
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        suite = get_suite(benchmark_version, suite_name)
        self.functions = {t.name: deepcopy(t) for t in suite.tools}
        self.environment_type = type(suite.load_and_inject_default_environment({}))
        self.max_chars = max_chars
        self.suite_name, self.include_effects = suite_name, include_effects

    def preview(self, runtime, env, call: FunctionCall) -> Preview:
        started = time.perf_counter()
        calls = 0
        try:
            if type(runtime) is not FunctionsRuntime or type(env) is not self.environment_type:
                return Preview(
                    "unsupported",
                    error="Runtime/environment is outside the audited simulator",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            expected = self.functions.get(call.function)
            actual = runtime.functions.get(call.function)
            if (
                expected is None
                or type(actual) is not type(expected)
                or actual != expected
                or has_nested_call(call.args)
            ):
                return Preview(
                    "unsupported",
                    error="Tool/runtime is outside the audited simulator",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            # A fresh runtime also isolates registry mutations. Stock simulator
            # functions use their injected environment rather than external I/O.
            clone = deepcopy(env)
            args = deepcopy(dict(call.args))
            before = None
            if self.include_effects:
                try:
                    before = effect_snapshot(self.suite_name, clone, call)
                except ValueError:
                    return Preview("effects_unavailable", error="No audited effect adapter",
                                   latency_ms=(time.perf_counter() - started) * 1000)
            calls = 1
            result, error = FunctionsRuntime([expected]).run_function(clone, call.function, args)
            observation = tool_result_to_str(result)
            effects = (effect_receipt(before, effect_snapshot(self.suite_name, clone, call), call)
                       if before is not None else None)
            effect_chars = len(json.dumps(effects, ensure_ascii=False)) if effects else 0
            if len(observation) + len(error or "") + effect_chars > self.max_chars:
                return Preview(
                    "too_large",
                    error="Preview exceeds evidence limit; not truncated",
                    calls=calls,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            return Preview(
                "tool_error" if error else "ok",
                observation,
                error,
                calls,
                (time.perf_counter() - started) * 1000,
                effects,
            )
        except Exception as exc:
            return Preview(
                "failed",
                error=f"Preview unavailable: {type(exc).__name__}",
                calls=calls,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
