"""Role-aware clients with one explicit retry layer and request accounting."""

from __future__ import annotations
import os
import time
import uuid
from types import SimpleNamespace
import openai
from agentdojo.agent_pipeline import OpenAILLM
from agentdojo.agent_pipeline.llms.openai_llm import (
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)


class _Completions:
    def __init__(
        self,
        inner,
        *,
        seed=0,
        temperature=0.0,
        role="agent",
        log=None,
        max_tokens=4096,
        attempts=2,
        reasoning_effort=None,
        kind="chat",
    ):
        self.inner, self.seed, self.temperature = inner, seed, temperature
        self.role, self.log, self.max_tokens = role, log, max_tokens
        self.attempts, self.reasoning_effort = attempts, reasoning_effort
        self.kind = kind

    def create(self, **kwargs):
        if self.kind == "chat":
            kwargs.update(temperature=self.temperature, seed=self.seed, max_tokens=self.max_tokens)
        if self.kind == "chat" and self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        request_id = uuid.uuid4().hex
        for attempt in range(self.attempts):
            started = time.perf_counter()
            base = dict(
                request_id=request_id,
                role=self.role,
                model=kwargs["model"],
                attempt=attempt,
                kind=self.kind,
            )
            if self.log:
                self.log.emit("llm_calls", event="started", **base)
            try:
                response = self.inner.create(**kwargs)
            except Exception as exc:
                if self.log:
                    self.log.emit(
                        "llm_calls",
                        event="error",
                        **base,
                        error_type=type(exc).__name__,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                retryable = isinstance(
                    exc,
                    (
                        openai.APIConnectionError,
                        openai.APITimeoutError,
                        openai.RateLimitError,
                        openai.InternalServerError,
                    ),
                )
                if not retryable or attempt + 1 == self.attempts:
                    raise
                time.sleep(min(2**attempt, 4))
                continue
            if self.log:
                usage = response.usage
                details = getattr(usage, "completion_tokens_details", None)
                self.log.emit(
                    "llm_calls",
                    event="completed",
                    **base,
                    tokens_in=getattr(usage, "prompt_tokens", None),
                    tokens_out=0
                    if self.kind == "embedding"
                    else getattr(usage, "completion_tokens", None),
                    reasoning_tokens=getattr(details, "reasoning_tokens", None),
                    finish_reason="embedding"
                    if self.kind == "embedding"
                    else response.choices[0].finish_reason,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            return response
        raise RuntimeError("attempts must be positive")


class DeterministicClient:
    def __init__(self, client, seed=0, temperature=0.0, **kwargs):
        self._client = client
        self.chat = SimpleNamespace(
            completions=_Completions(
                client.chat.completions, seed=seed, temperature=temperature, **kwargs
            )
        )
        self.embeddings = _Completions(
            client.embeddings, seed=seed, temperature=temperature, kind="embedding", **kwargs
        )

    def __getattr__(self, name):
        return getattr(self._client, name)


def build_client(
    seed=0,
    temperature=0.0,
    *,
    role="agent",
    log=None,
    max_tokens=4096,
    attempts=2,
    timeout=300,
    reasoning_effort=None,
):
    if attempts < 1 or max_tokens < 1 or timeout <= 0:
        raise ValueError("attempts, max_tokens and timeout must be positive")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unset: create .env and make sure load_dotenv() ran")
    raw = openai.OpenAI(
        api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"), timeout=timeout, max_retries=0
    )
    return DeterministicClient(
        raw,
        seed,
        temperature,
        role=role,
        log=log,
        max_tokens=max_tokens,
        attempts=attempts,
        reasoning_effort=reasoning_effort,
    )


class MeasuredLLM(OpenAILLM):
    """Reuse AgentDojo message adapters, bypass its nested tenacity retry."""

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        kwargs = dict(
            model=self.model, messages=[_message_to_openai(m, self.model) for m in messages]
        )
        tools = [_function_to_openai(f) for f in runtime.functions.values()]
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        response = self.client.chat.completions.create(**kwargs)
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("agent output limit reached; see llm_calls.jsonl")
        output = _openai_to_assistant_message(response.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args or {}


def build_llm(model, seed=0, temperature=0.0, **kwargs):
    return MeasuredLLM(build_client(seed, temperature, **kwargs), model, temperature=temperature)


def pipeline_name(model, policy, k=None, seed=0):
    budget = "inf" if k is None else str(k)
    return f"{model.replace('/', '_')}-{policy}-k{budget}-s{seed}"
