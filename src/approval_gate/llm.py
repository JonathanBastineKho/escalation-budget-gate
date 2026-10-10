"""Model clients for any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter).

We build OpenAILLM directly because AgentPipeline.from_config only accepts
model names from AgentDojo's fixed ModelsEnum.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import openai
from agentdojo.agent_pipeline import OpenAILLM


@dataclass
class UsageMeter:
    """Running token totals for one client. AgentDojo's OpenAILLM discards
    response.usage, so this is the only record of the agent's tokens."""

    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, usage) -> None:
        if usage is not None:
            self.tokens_in += usage.prompt_tokens
            self.tokens_out += usage.completion_tokens


class _Completions:
    """Forces temperature and seed onto every request.

    AgentDojo sends `temperature=temperature or NOT_GIVEN`, and 0.0 is falsy,
    so temperature=0.0 is silently dropped and the server samples at its own
    default (0.8 on Ollama). AgentDojo also never sends a seed.
    """

    def __init__(self, inner, seed: int, temperature: float, meter: UsageMeter | None) -> None:
        self._inner, self._seed, self._temperature = inner, seed, temperature
        self._meter = meter

    def create(self, **kwargs):
        kwargs["temperature"] = self._temperature
        kwargs["seed"] = self._seed
        try:
            response = self._inner.create(**kwargs)
        except openai.BadRequestError as e:
            # AgentDojo skips a case on overflow only if it recognises the error;
            # vLLM words it differently, which crashed the whole arm instead.
            if "maximum context length" not in str(e):
                raise
            body = dict(e.body) if isinstance(e.body, dict) else {}
            body["code"] = "context_length_exceeded"
            raise openai.BadRequestError(str(e), response=e.response, body=body) from e
        if self._meter is not None:
            self._meter.add(response.usage)
        return response


class _Chat:
    def __init__(self, inner, seed: int, temperature: float, meter: UsageMeter | None) -> None:
        self.completions = _Completions(inner.completions, seed, temperature, meter)


class DeterministicClient:
    """openai.OpenAI proxy that pins temperature and seed on every call, and
    counts tokens into `meter` when one is given."""

    def __init__(
        self,
        client: openai.OpenAI,
        seed: int,
        temperature: float = 0.0,
        meter: UsageMeter | None = None,
    ) -> None:
        self._client = client
        self.chat = _Chat(client.chat, seed, temperature, meter)

    def __getattr__(self, name):
        return getattr(self._client, name)


def build_client(
    seed: int = 0,
    temperature: float = 0.0,
    meter: UsageMeter | None = None,
    base_url: str | None = None,
    timeout: float = 1800,
    max_retries: int = 2,
) -> DeterministicClient:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unset: create .env and make sure load_dotenv() ran")
    raw = openai.OpenAI(
        api_key=api_key,
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        timeout=timeout,
        max_retries=max_retries,
    )
    return DeterministicClient(raw, seed=seed, temperature=temperature, meter=meter)


def build_llm(
    model: str, seed: int = 0, temperature: float = 0.0, meter: UsageMeter | None = None
) -> OpenAILLM:
    return OpenAILLM(build_client(seed, temperature, meter), model, temperature=temperature)


def pipeline_name(model: str, policy: str, k: int | None = None, seed: int = 0) -> str:
    """AgentDojo uses pipeline.name as both a directory name and its result
    cache key, so everything that changes a run must appear in it."""
    budget = "inf" if k is None else str(k)
    return f"{model.replace('/', '_')}-{policy}-k{budget}-s{seed}"
