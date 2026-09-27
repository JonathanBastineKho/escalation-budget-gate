"""The integration point with AgentDojo: a ToolsExecutor that asks the gate
before running each tool call."""

from __future__ import annotations

from ast import literal_eval
from copy import deepcopy
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
        observer=None,
    ) -> None:
        super().__init__(tool_output_formatter)
        self.gate = gate
        self.observer = observer

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        pending = extra_args.get("pending_calls")
        if pending is None:
            if len(messages) == 0 or messages[-1]["role"] != "assistant":
                return query, runtime, env, messages, extra_args
            pending = messages[-1].get("tool_calls") or []
        if not pending:
            return query, runtime, env, messages, extra_args

        results = []
        for call_index, tool_call in enumerate(pending):
            # Malformed calls skip the gate so they never cost budget.
            if tool_call.function == EMPTY_FUNCTION_NAME:
                results.append(
                    _tool_result_message(
                        tool_call,
                        "",
                        "Empty function name provided. Provide a valid function name.",
                    )
                )
                continue
            if tool_call.function not in (t.name for t in runtime.functions.values()):
                results.append(
                    _tool_result_message(
                        tool_call, "", f"Invalid tool {tool_call.function} provided."
                    )
                )
                continue

            # Same coercion as ToolsExecutor; done before the gate so it always
            # sees the same argument types.
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            function = next(t for t in runtime.functions.values() if t.name == tool_call.function)
            visible_messages = [*messages, *results]
            outcome = self.gate.decide(
                tool_call,
                visible_messages,
                {
                    "name": function.name,
                    "description": getattr(function, "description", ""),
                    "parameters": function.parameters.model_json_schema(),
                },
            )
            if self.observer:
                self.observer(
                    env.model_copy(deep=True),
                    deepcopy(visible_messages),
                    deepcopy(pending[call_index:]),
                    outcome.record,
                )
            if not outcome.approved:
                results.append(_tool_result_message(tool_call, "", outcome.denial_message))
                continue

            event = dict(
                step_index=outcome.record.step_index,
                call_id=outcome.record.call_id,
                tool_name=tool_call.function,
                arguments=dict(tool_call.args),
            )
            outcome.record.executed = True
            self.gate.executions.append(deepcopy(tool_call))
            if hasattr(self.gate.logger, "emit"):
                self.gate.logger.emit("executions", event="started", **event)
            try:
                result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            except Exception as exc:
                if hasattr(self.gate.logger, "emit"):
                    self.gate.logger.emit(
                        "executions",
                        event="finished",
                        success=False,
                        error_type=type(exc).__name__,
                        **event,
                    )
                raise
            if hasattr(self.gate.logger, "emit"):
                self.gate.logger.emit(
                    "executions", event="finished", success=error is None, error=error, **event
                )
            results.append(_tool_result_message(tool_call, self.output_formatter(result), error))

        return query, runtime, env, [*messages, *results], extra_args


class GatedAgentPipeline(AgentPipeline):
    """Resets the gate at the start of each task. AgentDojo reuses one
    pipeline for every task and calls query() once per task."""

    def __init__(self, elements, gate: ApprovalGate) -> None:
        super().__init__(elements)
        self.gate = gate

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        self.gate.start_task(query)
        return super().query(query, runtime, env, messages, extra_args)
