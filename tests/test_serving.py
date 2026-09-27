import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import httpx
import openai
from agentdojo.functions_runtime import FunctionCall, TaskEnvironment
from approval_gate.experiments import prepare_config, run_episode, rng_state, restore_rng
from approval_gate.telemetry.events import EventLog, read_jsonl
from approval_gate.reporting import request_cost
from approval_gate.llm import _Completions
from approval_gate.reviewer.simulated import _parse
from approval_gate.policies.melon_local import MelonLocalAgent


class ServingTests(unittest.TestCase):
    def test_sdk_path_accounts_for_all_three_roles(self):
        original = openai.OpenAI
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            model = body["model"]
            if model == "policy":
                text = json.dumps(
                    dict(
                        alignment=0.5,
                        unauthorized=0.5,
                        uncertainty=0.8,
                        necessity=0.5,
                        indirect=0.5,
                        decision="ESCALATE",
                        reason="check",
                    )
                )
                message = dict(role="assistant", content=text)
            elif model == "reviewer":
                message = dict(
                    role="assistant", content='{"verdict":"APPROVE","reason":"authorized"}'
                )
            elif any(m["role"] == "tool" for m in body["messages"]):
                message = dict(role="assistant", content="Done.")
            else:
                message = dict(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        dict(
                            id="call_1",
                            type="function",
                            function=dict(name="get_balance", arguments="{}"),
                        )
                    ],
                )
            return httpx.Response(
                200,
                json=dict(
                    id="test",
                    object="chat.completion",
                    created=1,
                    model=model,
                    choices=[dict(index=0, message=message, finish_reason="stop")],
                    usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16),
                ),
            )

        def factory(**kwargs):
            return original(
                **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
            )

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(
                os.environ, {"OPENAI_API_KEY": "test", "OPENAI_BASE_URL": "http://mock.local/v1"}
            ),
            patch("approval_gate.llm.openai.OpenAI", side_effect=factory),
            patch("approval_gate.experiments.evaluate", return_value=(True, False)),
        ):
            config = prepare_config(
                dict(
                    model="agent",
                    gate_model="reviewer",
                    policy_model="policy",
                    policy="llm_judge",
                    k=1,
                )
            )
            case = dict(
                suite="banking",
                task_id="user_task_0",
                family="b/0",
                split="train",
                attack="none",
                injection_task_id=None,
            )
            result = run_episode(config, case, d)
            self.assertEqual(result["status"], "completed", result["error"])
            path = (
                Path(d)
                / result["experiment_id"]
                / "cases"
                / result["case_id"]
                / result["episode_id"]
            )
            cost = request_cost(path)
            self.assertEqual(
                [cost[r + "_calls"] for r in ("agent", "policy", "reviewer")], [2, 1, 1]
            )
            self.assertEqual(result["escalations"], 1)
            self.assertTrue(all(r["temperature"] == 0 for r in seen))
            packets = read_jsonl(path / "decisions.jsonl")
            self.assertEqual(len(packets[0]["features"]), 10)

    def test_embeddings_are_separate_and_receive_no_generation_options(self):
        captured = []

        class Inner:
            def create(self, **kwargs):
                captured.append(kwargs)
                return NS(usage=NS(prompt_tokens=10), data=[])

        with tempfile.TemporaryDirectory() as d:
            client = _Completions(Inner(), kind="embedding", role="policy", log=EventLog(d))
            client.create(model="embed", input=["hello"])
            self.assertNotIn("temperature", captured[0])
            self.assertNotIn("max_tokens", captured[0])
            cost = request_cost(Path(d))
            self.assertEqual(cost["policy_embedding_calls"], 1)
            self.assertEqual(cost["policy_chat_calls"], 0)

    def test_duplicate_reviewer_keys_fail_closed(self):
        self.assertEqual(_parse('{"verdict":"DENY","verdict":"APPROVE","reason":"x"}')[0], "DENY")

    def test_rng_json_round_trip(self):
        import random
        import numpy as np

        state = json.loads(json.dumps(rng_state()))
        expected = (random.random(), np.random.random())
        restore_rng(state)
        self.assertEqual((random.random(), np.random.random()), expected)

    def test_melon_mask_never_executes_tools(self):
        class Env(TaskEnvironment):
            writes: int = 0

        proposal = FunctionCall(function="get_balance", args={}, id="x")

        class Agent:
            def query(self, q, r, e, m):
                return (
                    q,
                    r,
                    e,
                    [*m, {"role": "assistant", "content": None, "tool_calls": [proposal]}],
                    {},
                )

        class Embeddings:
            def create(self, model, input):
                return NS(data=[NS(index=i, embedding=[1.0, 0.0]) for i, _ in enumerate(input)])

        history = [
            {"role": "system", "content": []},
            {
                "role": "tool",
                "tool_call": proposal,
                "content": [{"type": "text", "content": "untrusted"}],
                "error": None,
            },
        ]
        with tempfile.TemporaryDirectory() as d:
            env = Env()
            detector = MelonLocalAgent(
                Agent(), Agent(), NS(embeddings=Embeddings()), "embed", 0.85, EventLog(d)
            )
            result = detector.query("q", object(), env, history)
            self.assertEqual(env.writes, 0)
            self.assertEqual(result[3][-1]["tool_calls"], [])
            self.assertTrue(read_jsonl(Path(d) / "defense_checks.jsonl")[0]["blocked"])


if __name__ == "__main__":
    unittest.main()
