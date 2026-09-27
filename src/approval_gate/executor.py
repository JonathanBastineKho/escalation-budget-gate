"""The integration point with AgentDojo: a ToolsExecutor that asks the gate
before running each tool call."""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable, Sequence

from agentdojo.agent_pipeline import AgentPipeline, ToolsExecutor
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from approval_gate.gate import ApprovalGate


def _tool_result_message(tool_call, content: str, error: str | None) -> ChatToolResultMessage:
    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(content)],
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        error=error,
    )


class GatedToolsExecutor(ToolsExecutor):
    def __init__(
        self,
        gate: ApprovalGate,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
    ) -> None:
        super().__init__(tool_output_formatter)
        self.gate = gate

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        if len(messages) == 0 or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args

        results = []
        for tool_call in messages[-1]["tool_calls"]:
            # Malformed calls skip the gate so they never cost budget.
            if tool_call.function == EMPTY_FUNCTION_NAME:
                results.append(_tool_result_message(
                    tool_call, "", "Empty function name provided. Provide a valid function name."))
                continue
            if tool_call.function not in (t.name for t in runtime.functions.values()):
                results.append(_tool_result_message(
                    tool_call, "", f"Invalid tool {tool_call.function} provided."))
                continue

            # Same coercion as ToolsExecutor; done before the gate so it always
            # sees the same argument types.
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            outcome = self.gate.decide(tool_call, messages)
            if not outcome.approved:
                results.append(_tool_result_message(tool_call, "", outcome.denial_message))
                continue

            result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            results.append(_tool_result_message(tool_call, self.output_formatter(result), error))

        return query, runtime, env, [*messages, *results], extra_args


def track_cases(suite, gate: ApprovalGate) -> None:
    """Wraps suite.run_task_with_pipeline, which AgentDojo calls once per
    (user task, injection task) case, so the gate knows the case ids and can
    write a case record with the outcome. It may call pipeline.query() several
    times inside, which is why this can't live in GatedAgentPipeline."""
    # Bind the class method, not the instance attribute: get_suite() caches
    # suites, so a second run in one process would otherwise wrap our wrapper.
    run_case = type(suite).run_task_with_pipeline.__get__(suite)

    def run_tracked(agent_pipeline, user_task, injection_task, injections, *args, **kwargs):
        gate.start_case(user_task.ID, injection_task.ID if injection_task is not None else None)
        try:
            utility, security = run_case(
                agent_pipeline, user_task, injection_task, injections, *args, **kwargs
            )
        except Exception as e:
            # AgentDojo handles some of these (e.g. context overflow) upstream.
            gate.end_case(None, None, error=type(e).__name__)
            raise
        gate.end_case(utility, security, error=None)
        return utility, security

    suite.run_task_with_pipeline = run_tracked


class GatedAgentPipeline(AgentPipeline):
    """Resets the gate at the start of each task. AgentDojo reuses one
    pipeline for every task and calls query() once per task."""

    def __init__(self, elements, gate: ApprovalGate) -> None:
        super().__init__(elements)
        self.gate = gate

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        self.gate.start_task(query)
        return super().query(query, runtime, env, messages, extra_args)
