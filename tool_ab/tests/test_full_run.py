"""remote/full_run.py: the commands each arm runs on Modal."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "full_run", Path(__file__).resolve().parents[1] / "remote" / "full_run.py")
full_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full_run)


def test_each_arm_runs_retries_scores_and_converts_into_the_volume():
    root = Path("/vol/full")
    run, retry, score, convert = full_run.arm_commands("search", root, "0:100", 6)
    assert run[1].endswith("run_arm.py")
    assert run[run.index("--slice") + 1] == "0:100" and run[run.index("-w") + 1] == "6"
    assert run[run.index("-o") + 1] == "/vol/full/runs/search"
    assert retry == [*run, "--retry-errors"]
    assert score[score.index("--preds") + 1] == "/vol/full/runs/search/preds.json"
    assert score[score.index("--out") + 1] == "/vol/full/eval"
    assert convert[convert.index("--report") + 1] == "/vol/full/eval/search.json"
    assert convert[convert.index("--out") + 1] == "/vol/full"


@pytest.mark.parametrize(("arms", "expected"), [
    (["baseline", "search", "compressed"], ["qwen-search", "qwen-compressed"]),
    (["baseline"], None),
    (["search"], None),
])
def test_analysis_needs_the_baseline_and_another_arm(arms, expected):
    command = full_run.analyze_command(Path("/vol/x"), arms)
    if expected is None:
        assert command is None
    else:
        start = command.index("--arms") + 1
        assert command[start:start + len(expected)] == expected
        assert command[command.index("--out") + 1] == "/vol/x/results"
