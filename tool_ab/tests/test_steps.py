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


def test_to_records_fills_the_shared_format():
    records = to_records(messages_to_steps(MESSAGES), task_id="t1", agent="qwen-baseline", outcome=1)
    assert [r["step"] for r in records] == [1, 2]
    assert all(r["task_id"] == "t1" and r["outcome"] == 1 for r in records)
    assert set(records[0]) == {
        "task_id", "agent", "step", "tokens_in", "tokens_out", "action", "observation", "outcome"}
