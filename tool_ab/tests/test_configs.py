"""Each arm must differ from the baseline in exactly the intended setting."""

from pathlib import Path

import minisweagent
import pytest
import yaml
from jinja2 import StrictUndefined, Template
from run_arm import ARMS, arm_config, filter_regex
from shortstop_plugin.agent import StoppingAgentConfig
from shortstop_plugin.model import ToolModelConfig

UPSTREAM = Path(minisweagent.__file__).parent / "config/benchmarks/swebench.yaml"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) and v else {key: v})
    return out


def render(template, **variables):
    return Template(template, undefined=StrictUndefined).render(**variables)


def test_baseline_prompt_is_the_upstream_prompt():
    ours = arm_config("baseline")["agent"]["instance_template"]
    upstream = yaml.safe_load(UPSTREAM.read_text())["agent"]["instance_template"]
    assert render(ours, task="Fix bug 1", extra_instructions="") == render(upstream, task="Fix bug 1")


@pytest.mark.parametrize("arm,expected", [
    ("descriptions", {"model.tool_description", "model.command_description"}),
    ("search", {"agent.setup_scripts", "agent.extra_instructions"}),
    ("compressed", {"agent.compression.max_chars", "agent.compression.max_line_chars",
                    "agent.compression.head_fraction"}),
])
def test_each_arm_changes_only_its_own_settings(arm, expected):
    base, treated = flatten(arm_config("baseline")), flatten(arm_config(arm))
    changed = {k for k in base.keys() | treated.keys() if base.get(k) != treated.get(k)}
    assert changed - {"agent.agent_name"} == expected  # the name is a label, not a setting
    assert treated["agent.agent_name"] == f"qwen-{arm}" and base["agent.agent_name"] == "qwen-baseline"


@pytest.mark.parametrize("arm", list(ARMS))
def test_arm_configs_are_valid(arm):
    config = arm_config(arm)
    StoppingAgentConfig(**config["agent"])
    model = {k: v for k, v in config["model"].items() if k != "model_class"}
    ToolModelConfig(**model)


def test_search_prompt_documents_the_tool():
    agent = arm_config("search")["agent"]
    prompt = render(agent["instance_template"], task="t", extra_instructions=agent["extra_instructions"])
    assert "search -f NAME" in prompt
    assert prompt.index("Search Tool") < prompt.index("## Submission")


def test_filter_regex_matches_exactly_the_listed_tasks():
    import re
    pattern = filter_regex(["django__django-11099", "astropy__astropy-12907"])
    assert re.match(pattern, "django__django-11099")
    assert not re.match(pattern, "django__django-110999")
    assert not re.match(pattern, "sympy__sympy-1")
