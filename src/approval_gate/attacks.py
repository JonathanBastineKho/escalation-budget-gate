"""AgentDojo's important_instructions attack, addressing the model generically.

The built-in version looks up the model's name from the pipeline name and
raises for models AgentDojo doesn't list (every Qwen model). Its
no_model_name variant runs the same lookup first, so it fails too.
"""

from __future__ import annotations

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME, FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import (
    ImportantInstructionsAttack,
    ToolKnowledgeAttack,
)


@register_attack
class ImportantInstructionsGeneric(FixedJailbreakAttack):
    name = "important_instructions_generic"

    def __init__(self, task_suite, target_pipeline) -> None:
        super().__init__(ImportantInstructionsAttack._JB_STRING, task_suite, target_pipeline)
        self.user_name = "Emma Johnson"
        self.model_name = DEFAULT_MODEL_NAME


@register_attack
class ToolKnowledgeGeneric(ToolKnowledgeAttack):
    """tool_knowledge, which also names the exact tool calls to make. Skips its
    parents' __init__, which run the failing model-name lookup."""

    name = "tool_knowledge_generic"

    def __init__(self, task_suite, target_pipeline) -> None:
        FixedJailbreakAttack.__init__(self, ToolKnowledgeAttack._JB_STRING, task_suite, target_pipeline)
        self.user_name = "Emma Johnson"
        self.model_name = DEFAULT_MODEL_NAME
