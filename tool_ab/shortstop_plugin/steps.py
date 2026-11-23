"""mini-swe-agent messages to steps in the shared trajectory format.

One step is one model call plus everything the environment said back before the next call.
This is the same code as `to_steps` in coverage/download_swebench.py. One difference in the
data: mini-swe-agent 1.17.5 logged a reply that failed to parse as an assistant message (a
step), while 2.4.6 logs it only as the user-side error message, with the reply's usage under
extra.response. Such a failed call is not a step here (so steps still match the plug-in's
draws), but its tokens are added to the step that follows it, or to the last step if the run
ended on it, so no tokens are lost.
"""

from __future__ import annotations

from typing import Any

from shortstop import Step

UNKNOWN_OUTCOME = -1  # used for live prefixes, before the run has been evaluated


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(p.get("text") if isinstance(p, dict) else p) for p in content)
    return str(content)


def _is_model_call(msg: dict) -> bool:
    return msg.get("role") == "assistant" or msg.get("object") == "response"


def _action(msg: dict) -> str:
    parts = [_text(msg.get("content"))]
    extra = msg.get("extra") or {}
    if extra.get("actions"):
        parts += [a.get("command", "") for a in extra["actions"] if isinstance(a, dict)]
    else:
        parts += [(c.get("function") or {}).get("arguments", "") for c in msg.get("tool_calls") or []]
    return "\n".join(p for p in parts if p)


def _response(msg: dict) -> dict:
    response = (msg.get("extra") or {}).get("response")
    return response if isinstance(response, dict) else {}  # litellm logs repr() when it cannot dump


def _usage(msg: dict) -> tuple[int, int] | None:
    usage = msg.get("usage") or _response(msg).get("usage")
    if not isinstance(usage, dict):
        return None
    tokens_in = usage.get("prompt_tokens", usage.get("input_tokens"))
    tokens_out = usage.get("completion_tokens", usage.get("output_tokens"))
    if tokens_in is None or tokens_out is None:
        return None
    return int(tokens_in), int(tokens_out)


def _failed_call_usage(msg: dict) -> tuple[int, int] | None:
    """Usage of a model reply that failed to parse, logged on the error message that replaced it."""
    if _is_model_call(msg) or not _response(msg):
        return None
    return _usage(msg)


def _add_tokens(step: dict, tokens: tuple[int, int]) -> dict:
    return {**step, "tokens_in": step["tokens_in"] + tokens[0], "tokens_out": step["tokens_out"] + tokens[1]}


def messages_to_steps(messages: list[dict]) -> list[dict]:
    """Steps as dicts with tokens_in, tokens_out, action, observation.

    Token counts come from the logged API usage, or are estimated as characters / 4. Tokens of
    failed calls go to the next step, or to the last step if no step follows.
    """
    steps: list[dict] = []
    chars_so_far = 0
    pending = (0, 0)  # tokens of failed calls not yet given to a step
    for msg in messages:
        if _is_model_call(msg):
            action = _action(msg)
            tokens = _usage(msg) or (chars_so_far // 4, len(action) // 4)
            step = {"tokens_in": tokens[0], "tokens_out": tokens[1], "action": action, "observation": ""}
            steps.append(_add_tokens(step, pending))
            pending = (0, 0)
            chars_so_far += len(action)
        elif msg.get("role") == "exit":
            continue
        else:
            failed = _failed_call_usage(msg)
            if failed:
                pending = (pending[0] + failed[0], pending[1] + failed[1])
            text = _text(msg.get("content", msg.get("output")))
            chars_so_far += len(text)
            if steps:
                sep = "\n" if steps[-1]["observation"] else ""
                steps[-1] = {**steps[-1], "observation": steps[-1]["observation"] + sep + text}
    if steps and pending != (0, 0):
        steps[-1] = _add_tokens(steps[-1], pending)
    return steps


def to_records(steps: list[dict], *, task_id: str, agent: str, outcome: int) -> list[dict]:
    """Add the identifying fields so each step is one line of the shared JSONL format."""
    return [
        {"task_id": task_id, "agent": agent, "step": i, **step, "outcome": outcome}
        for i, step in enumerate(steps, start=1)
    ]


def to_step_objects(steps: list[dict], *, task_id: str, agent: str) -> list[Step]:
    """Live prefix as shortstop.Step objects for a predictor; the outcome is not known yet."""
    records = to_records(steps, task_id=task_id, agent=agent, outcome=UNKNOWN_OUTCOME)
    return [Step(**r) for r in records]
