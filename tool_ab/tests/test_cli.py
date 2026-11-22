"""Command-line entry points: run_arm (runner mocked), convert, analyze, tasks."""

import json
import re
import sys

import analyze
import convert
import pytest
import run_arm
import tasks
from shortstop_plugin.agent import StoppingAgent


def test_slice_and_read_tasks(tmp_path):
    path = tmp_path / "tasks.txt"
    path.write_text("# header\nt1\n\nt2\nt3\n")
    assert run_arm.read_tasks(path) == ["t1", "t2", "t3"]
    assert run_arm.slice_tasks(["a", "b", "c"], "0:2") == ["a", "b"]
    assert run_arm.slice_tasks(["a", "b", "c"], "1:") == ["b", "c"]
    assert run_arm.slice_tasks(["a", "b"], "") == ["a", "b"]
    with pytest.raises(ValueError):
        run_arm.config_specs("nonexistent")


def test_run_arm_main_swaps_in_the_plugin_and_passes_the_task_filter(tmp_path, monkeypatch):
    from minisweagent import environments
    from minisweagent.run.benchmarks import swebench
    from shortstop_plugin.environment import BashModalEnvironment

    monkeypatch.setitem(environments._ENVIRONMENT_MAPPING, "swerex_modal",
                        environments._ENVIRONMENT_MAPPING["swerex_modal"])

    calls = {}
    monkeypatch.setattr(swebench, "main", lambda **kw: calls.update(kw))
    monkeypatch.setattr(swebench, "ProgressTrackingAgent", swebench.ProgressTrackingAgent)
    task_file = tmp_path / "tasks.txt"
    task_file.write_text("a__a-1\nb__b-2\nc__c-3\n")
    run_arm.main(["--arm", "search", "--tasks", str(task_file), "--slice", "0:2",
                  "-o", str(tmp_path / "out"), "-c", "agent.step_limit=5"])

    assert issubclass(swebench.ProgressTrackingAgent, StoppingAgent)
    assert environments.get_environment_class("swerex_modal") is BashModalEnvironment
    assert re.match(calls["filter_spec"], "b__b-2") and not re.match(calls["filter_spec"], "c__c-3")
    assert calls["config_spec"][-2].endswith("search.yaml")
    assert calls["config_spec"][-1] == "agent.step_limit=5"
    assert calls["subset"] == "verified" and calls["split"] == "test"


def test_run_arm_main_rejects_an_empty_selection(tmp_path):
    task_file = tmp_path / "tasks.txt"
    task_file.write_text("a__a-1\n")
    with pytest.raises(SystemExit):
        run_arm.main(["--arm", "baseline", "--tasks", str(task_file), "--slice", "5:9"])


def test_progress_agent_reports_each_step(tmp_path):
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel, make_output

    statuses = []

    class Progress:
        def update_instance_status(self, instance_id, status):
            statuses.append((instance_id, status))

    class StoppableEnvironment(LocalEnvironment):
        stopped = 0

        def stop(self):
            self.stopped += 1

    model = DeterministicModel(outputs=[make_output("s", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}])])
    env = StoppableEnvironment(cwd=str(tmp_path))
    agent = run_arm.make_progress_agent()(
        model, env, progress_manager=Progress(), instance_id="t1",
        system_template="s", instance_template="{{task}}",
    )
    agent.run("x")
    assert statuses == [("t1", "Step   1")]
    assert env.stopped == 1  # the batch runner never stops environments itself


def test_progress_agent_stops_the_environment_even_when_the_run_fails(tmp_path):
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel

    class Progress:
        def update_instance_status(self, instance_id, status):
            raise ConnectionError("model server dropped")

    class StoppableEnvironment(LocalEnvironment):
        stopped = 0

        def stop(self):
            self.stopped += 1

    env = StoppableEnvironment(cwd=str(tmp_path))
    agent = run_arm.make_progress_agent()(
        DeterministicModel(outputs=[]), env, progress_manager=Progress(), instance_id="t1",
        system_template="s", instance_template="{{task}}",
    )
    with pytest.raises(ConnectionError):
        agent.run("x")
    assert env.stopped == 1


def write_arm_runs(runs_dir, outcomes_by_task):
    for task_id in outcomes_by_task:
        path = runs_dir / task_id / f"{task_id}.traj.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"instance_id": task_id, "info": {"exit_status": "Submitted"}, "messages": [
            {"role": "assistant", "content": "x", "extra": {"actions": [{"command": "ls"}]}},
        ]}))


def test_convert_and_analyze_mains(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    resolved = {"baseline": ["t1"], "search": ["t1", "t2"]}
    for arm, ids in resolved.items():
        write_arm_runs(tmp_path / "runs" / arm, ["t1", "t2", "t3"])
        report = tmp_path / f"{arm}.json"
        report.write_text(json.dumps({"resolved_ids": ids}))
        monkeypatch.setattr(sys, "argv", ["convert.py", "--arm", arm, "--report", str(report),
                                          "--runs", str(tmp_path / "runs" / arm), "--out", str(data)])
        convert.main()
    assert "qwen-search: 3 runs, 2 resolved" in capsys.readouterr().out

    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--data", str(data), "--out", str(out),
                                      "--arms", "qwen-search", "qwen-missing", "--n-boot", "100"])
    analyze.main()
    printed = capsys.readouterr().out
    assert "skip qwen-missing" in printed
    assert "| qwen-search | 3 | 33.3% | 66.7%" in printed
    assert (out / "full_run_verdicts.csv").read_text().startswith("arm,n_tasks")


def test_analyze_main_with_nothing_to_compare(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "qwen-baseline.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--data", str(data), "--arms", "qwen-x"])
    with pytest.raises(SystemExit):
        analyze.main()


def test_compare_requires_shared_tasks():
    with pytest.raises(ValueError):
        analyze.compare({"a": {"outcome": 1, "tokens": 1}}, {"b": {"outcome": 1, "tokens": 1}})


def test_task_order_is_fixed_by_the_seed():
    ids = [f"t{i}" for i in range(50)]
    assert tasks.ordered_tasks(ids, 0) == tasks.ordered_tasks(list(reversed(ids)), 0)
    assert tasks.ordered_tasks(ids, 0) != tasks.ordered_tasks(ids, 1)
    assert sorted(tasks.ordered_tasks(ids, 0)) == sorted(ids)


def test_retry_errors_removes_errored_tasks_from_preds(tmp_path):
    out = tmp_path / "out"
    preds = {}
    for task, status in [("good", "Submitted"), ("bad", "APIConnectionError")]:
        path = out / task / f"{task}.traj.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"instance_id": task, "info": {"exit_status": status}, "messages": []}))
        preds[task] = {"instance_id": task, "model_patch": ""}
    (out / "preds.json").write_text(json.dumps(preds))
    assert run_arm.forget_errored_runs(out) == ["bad"]
    assert set(json.loads((out / "preds.json").read_text())) == {"good"}


def test_retry_errors_also_catches_runs_whose_sandbox_never_started(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # The runner records an empty patch but saves no trajectory when the environment fails.
    (out / "preds.json").write_text(json.dumps({"lost": {"instance_id": "lost", "model_patch": ""}}))
    assert run_arm.forget_errored_runs(out) == ["lost"]
    assert json.loads((out / "preds.json").read_text()) == {}
