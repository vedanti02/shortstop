"""Model class whose bash tool description comes from the config: the first tool change.

With the default config the tool is identical to mini-swe-agent's BASH_TOOL, so the baseline
arm uses this class too and the arms differ only in their YAML.
"""

from __future__ import annotations

import litellm
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

_DEFAULT_TOOL = BASH_TOOL["function"]["description"]
_DEFAULT_COMMAND = BASH_TOOL["function"]["parameters"]["properties"]["command"]["description"]


class ToolModelConfig(LitellmModelConfig):
    tool_description: str = _DEFAULT_TOOL
    command_description: str = _DEFAULT_COMMAND


def build_bash_tool(tool_description: str, command_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": command_description}},
                "required": ["command"],
            },
        },
    }


class ToolModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=ToolModelConfig, **kwargs)
        self.bash_tool = build_bash_tool(self.config.tool_description, self.config.command_description)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=[self.bash_tool],
            **(self.config.model_kwargs | kwargs),
        )
