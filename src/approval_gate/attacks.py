"""AgentDojo's important_instructions attack, addressing the model generically.

The built-in version looks up the model's name from the pipeline name and
raises for models AgentDojo doesn't list (every Qwen model). Its
no_model_name variant runs the same lookup first, so it fails too.
"""

from __future__ import annotations

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME, FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack


@register_attack
class ImportantInstructionsGeneric(FixedJailbreakAttack):
    name = "important_instructions_generic"

    def __init__(self, task_suite, target_pipeline) -> None:
        super().__init__(ImportantInstructionsAttack._JB_STRING, task_suite, target_pipeline)
        self.user_name = "Emma Johnson"
        self.model_name = DEFAULT_MODEL_NAME
