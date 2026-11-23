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


@pytest.mark.filterwarnings("ignore:The run_arm_job function is executing locally")
def test_an_arm_still_converts_when_some_tasks_could_not_be_graded(monkeypatch, tmp_path):
    monkeypatch.setattr(full_run, "VOLUME_ROOT", tmp_path)
    ran = []

    def fake_run(command, log_path):
        ran.append(command[1].rsplit("/", 1)[-1])
        if ran[-1] != "evaluate.py":
            return 0
        report = tmp_path / "x" / "eval" / "search.json"  # written, with some tasks under error_ids
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}")
        return 1

    monkeypatch.setattr(full_run, "run_and_commit", fake_run)
    monkeypatch.setattr(full_run, "point_at_model_server", lambda: None)
    result = full_run.run_arm_job.local("search", "x", "0:1", 1)
    assert ran == ["run_arm.py", "run_arm.py", "evaluate.py", "convert.py"]
    assert result.startswith("search: done") and "not graded" in result


@pytest.mark.filterwarnings("ignore:The run_arm_job function is executing locally")
def test_an_arm_stops_when_running_the_agent_fails(monkeypatch):
    monkeypatch.setattr(full_run, "run_and_commit", lambda command, log_path: 1)
    monkeypatch.setattr(full_run, "point_at_model_server", lambda: None)
    assert "FAILED" in full_run.run_arm_job.local("search", "x", "0:1", 1)


@pytest.mark.filterwarnings("ignore:The run_arm_job function is executing locally")
def test_a_stale_report_is_never_used_when_evaluation_crashes(monkeypatch, tmp_path):
    monkeypatch.setattr(full_run, "VOLUME_ROOT", tmp_path)
    stale = tmp_path / "x" / "eval" / "search.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    ran = []

    def fake_run(command, log_path):
        ran.append(command[1].rsplit("/", 1)[-1])
        return 1 if ran[-1] == "evaluate.py" else 0  # crashes before writing a report

    monkeypatch.setattr(full_run, "run_and_commit", fake_run)
    monkeypatch.setattr(full_run, "point_at_model_server", lambda: None)
    result = full_run.run_arm_job.local("search", "x", "0:1", 1)
    assert "FAILED" in result and "convert.py" not in ran and not stale.exists()
