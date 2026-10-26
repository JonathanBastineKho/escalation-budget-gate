"""Contracts shared by every policy implementation."""

import pytest

from approval_gate.policies.base import Policy


def test_call_context_features_are_not_shared_between_calls(make_context):
    first, second = make_context(), make_context()
    first.features["tokens_in"] = {"judge": 10}

    assert second.features == {}
    assert first.features is not second.features


def test_policy_subclass_must_implement_decision():
    class MissingDecision(Policy):
        pass

    with pytest.raises(TypeError, match="abstract"):
        MissingDecision()
