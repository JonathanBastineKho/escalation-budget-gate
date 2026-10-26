"""Fixed policy inputs; no model clients, live tools, or saved experiment logs."""

from unittest.mock import Mock

import pytest
from agentdojo.functions_runtime import FunctionCall

from approval_gate.nodes.envelope import TaskEnvelope
from approval_gate.policies.base import CallContext
from approval_gate.policies.cascade import CascadePolicy


@pytest.fixture
def make_context():
    def make(tool="send_money", *, args=None, budget=3, **overrides):
        values = {
            "task_id": "fixture-task",
            "user_prompt": "Send 25 to Alice once for lunch.",
            "tool_call": FunctionCall(
                id="fixture-call",
                function=tool,
                args={"recipient": "Alice", "amount": 25} if args is None else args,
            ),
            "messages": [],
            "step_index": 0,
            "k_remaining": budget,
            **overrides,
        }
        return CallContext(**values)

    return make


@pytest.fixture
def make_cascade():
    def make(*, policy_class=CascadePolicy, expected_writes=4, **options):
        extractor = Mock()
        extractor.extract.return_value = TaskEnvelope(
            expected_tools=["send_money", "read_file"],
            entities=["Alice"],
            expected_writes=expected_writes,
            tokens_in=17,
            tokens_out=5,
        )
        policy = policy_class(extractor, **options)
        policy.start_task("fixture-task", "Send 25 to Alice once for lunch.")
        return policy

    return make
