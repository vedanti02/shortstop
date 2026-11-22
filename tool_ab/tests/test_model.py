from unittest.mock import patch

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from shortstop_plugin.model import ToolModel


def captured_tools(**config):
    model = ToolModel(model_name="hosted_vllm/qwen3-coder", **config)
    with patch("shortstop_plugin.model.litellm.completion") as completion:
        model._query([{"role": "user", "content": "hi"}])
    return completion.call_args.kwargs["tools"]


def test_default_tool_is_exactly_upstream_bash_tool():
    assert captured_tools() == [BASH_TOOL]


def test_descriptions_come_from_the_config():
    (tool,) = captured_tools(tool_description="Run bash.", command_description="One command.")
    assert tool["function"]["name"] == "bash"
    assert tool["function"]["description"] == "Run bash."
    assert tool["function"]["parameters"]["properties"]["command"]["description"] == "One command."
