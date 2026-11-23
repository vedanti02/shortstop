from shortstop_plugin.steps import messages_to_steps, to_records

MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "task"},
    {
        "role": "assistant", "content": "look around",
        "extra": {"actions": [{"command": "ls"}], "response": {"usage": {
            "prompt_tokens": 100, "completion_tokens": 20}}},
    },
    {"role": "tool", "content": "a.py\nb.py"},
    {"role": "assistant", "content": "read", "extra": {"actions": [{"command": "cat a.py"}]}},
    {"role": "tool", "content": "print(1)"},
    {"role": "user", "content": "format warning"},
    {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted"}},
]


def test_one_step_per_model_call_with_its_observations():
    steps = messages_to_steps(MESSAGES)
    assert len(steps) == 2
    assert steps[0]["action"] == "look around\nls"
    assert steps[0]["observation"] == "a.py\nb.py"
    assert steps[1]["observation"] == "print(1)\nformat warning"


def test_token_usage_is_read_or_estimated():
    steps = messages_to_steps(MESSAGES)
    assert (steps[0]["tokens_in"], steps[0]["tokens_out"]) == (100, 20)
    assert steps[1]["tokens_in"] > 0  # no usage logged, so estimated from characters


def failed_call(prompt, completion):
    """A reply that failed to parse, as mini-swe-agent 2.4.6 logs it."""
    return {"role": "user", "content": "Tool call error: Unknown tool",
            "extra": {"interrupt_type": "FormatError",
                      "response": {"usage": {"prompt_tokens": prompt, "completion_tokens": completion}}}}


def call(prompt, completion):
    return {"role": "assistant", "content": "act", "extra": {
        "actions": [{"command": "ls"}],
        "response": {"usage": {"prompt_tokens": prompt, "completion_tokens": completion}}}}


def test_failed_calls_are_not_steps_but_their_tokens_go_to_the_next_step():
    messages = [{"role": "system", "content": "s"}, call(100, 10), {"role": "tool", "content": "out"},
                failed_call(200, 20), failed_call(300, 30), call(400, 40)]
    steps = messages_to_steps(messages)
    assert len(steps) == 2  # still one step per draw
    assert (steps[1]["tokens_in"], steps[1]["tokens_out"]) == (900, 90)
    assert (steps[0]["tokens_in"], steps[0]["tokens_out"]) == (100, 10)
    assert "Tool call error" in steps[0]["observation"]


def test_a_failed_call_that_ends_the_run_counts_toward_the_last_step():
    messages = [call(100, 10), {"role": "tool", "content": "out"}, failed_call(200, 20),
                {"role": "exit", "content": "RepeatedFormatError", "extra": {}}]
    steps = messages_to_steps(messages)
    assert len(steps) == 1
    assert (steps[0]["tokens_in"], steps[0]["tokens_out"]) == (300, 30)


def test_to_records_fills_the_shared_format():
    records = to_records(messages_to_steps(MESSAGES), task_id="t1", agent="qwen-baseline", outcome=1)
    assert [r["step"] for r in records] == [1, 2]
    assert all(r["task_id"] == "t1" and r["outcome"] == 1 for r in records)
    assert set(records[0]) == {
        "task_id", "agent", "step", "tokens_in", "tokens_out", "action", "observation", "outcome"}


def test_a_response_logged_as_a_string_is_ignored():
    messages = [call(100, 10), {"role": "user", "content": "err", "extra": {"response": "ModelResponse(...)"}},
                call(50, 5)]
    steps = messages_to_steps(messages)
    assert [(s["tokens_in"], s["tokens_out"]) for s in steps] == [(100, 10), (50, 5)]
