"""Check that the served model answers with a parsed bash tool call, the way mini-swe-agent needs.

    python tool_ab/serving/check_endpoint.py

Reads HOSTED_VLLM_API_BASE and HOSTED_VLLM_API_KEY from tool_ab/.env. Sends one short
request (the first request after a cold start can take several minutes).
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOL_AB = Path(__file__).resolve().parents[1]


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(TOOL_AB / ".env")
    sys.path.insert(0, str(TOOL_AB))
    from shortstop_plugin.model import ToolModel

    model = ToolModel(model_name="hosted_vllm/qwen3-coder", cost_tracking="ignore_errors",
                      model_kwargs={"timeout": 900})
    response = model._query([
        {"role": "system", "content": "You can run bash commands with the bash tool."},
        {"role": "user", "content": "List the files in the current directory."},
    ])
    message = response.choices[0].message
    calls = message.tool_calls or []
    usage = response.usage
    print(f"tool calls: {[c.function.arguments for c in calls]}")
    print(f"tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
    if not calls or calls[0].function.name != "bash":
        raise SystemExit("FAIL: no bash tool call was parsed; check --tool-call-parser")
    print("OK")


if __name__ == "__main__":
    main()
