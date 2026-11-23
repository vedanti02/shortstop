import json

from convert import convert_arm, load_resolved, write_arm

from shortstop import load_trajectories


def write_traj(runs_dir, instance_id, n_steps, shortstop=None):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(n_steps):
        messages += [
            {"role": "assistant", "content": f"think {i}", "extra": {"actions": [{"command": "ls"}]}},
            {"role": "tool", "content": f"out {i}"},
        ]
    info = {"exit_status": "Submitted"}
    if shortstop is not None:
        info["shortstop"] = shortstop
    path = runs_dir / instance_id / f"{instance_id}.traj.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"info": info, "messages": messages, "instance_id": instance_id}))


def test_load_resolved_reads_both_report_styles(tmp_path):
    harness = tmp_path / "harness.json"
    harness.write_text(json.dumps({"resolved_ids": ["a", "b"], "unresolved_ids": ["c"]}))
    per_task = tmp_path / "per_task.json"
    per_task.write_text(json.dumps({"a": {"resolved": True}, "c": {"resolved": False}}))
    assert load_resolved(harness) == {"a", "b"}
    assert load_resolved(per_task) == {"a"}


def test_convert_arm_writes_shared_format_and_stop_records(tmp_path):
    runs = tmp_path / "runs"
    write_traj(runs, "t1", 3, shortstop={"steps_run": 3, "continue_probs": [1, 1, 1],
                                          "predictions": [0.5] * 3, "stopped_early": False})
    write_traj(runs, "t2", 2)
    records, stops, _ = convert_arm(runs, resolved={"t1"}, agent="qwen-baseline")
    assert {(r["task_id"], r["outcome"]) for r in records} == {("t1", 1), ("t2", 0)}
    assert stops["t1"]["continue_probs"] == [1, 1, 1]
    assert "t2" not in stops

    out = tmp_path / "out"
    jsonl = write_arm(records, stops, out, "qwen-baseline")
    runs_loaded = load_trajectories(jsonl)
    assert [s.step for s in runs_loaded[("t1", "qwen-baseline")]] == [1, 2, 3]
    assert json.loads((out / "qwen-baseline.stops.json").read_text())["t1"]["steps_run"] == 3


def test_run_without_model_calls_is_kept_as_one_empty_step(tmp_path):
    runs = tmp_path / "runs"
    write_traj(runs, "t0", 0)
    records, _, _ = convert_arm(runs, resolved=set(), agent="a")
    assert len(records) == 1 and records[0]["step"] == 1 and records[0]["outcome"] == 0


def test_early_stopped_run_has_unknown_outcome(tmp_path):
    runs = tmp_path / "runs"
    write_traj(runs, "t3", 1, shortstop={"steps_run": 1, "continue_probs": [1, 0.2],
                                          "predictions": [0.5, 0.4], "stopped_early": True})
    records, stops, _ = convert_arm(runs, resolved={"t3"}, agent="a")
    assert {r["outcome"] for r in records} == {-1}
    assert stops["t3"]["stopped_early"] is True


def test_infrastructure_errors_are_excluded_not_scored_as_failures(tmp_path):
    runs = tmp_path / "runs"
    write_traj(runs, "ok", 1)
    write_traj(runs, "broken", 1)
    write_traj(runs, "overflow", 1)
    for task, status in [("broken", "DeploymentStartupError"), ("overflow", "ContextWindowExceededError")]:
        path = runs / task / f"{task}.traj.json"
        data = json.loads(path.read_text())
        data["info"]["exit_status"] = status
        path.write_text(json.dumps(data))
    records, _, errors = convert_arm(runs, resolved={"ok"}, agent="a")
    assert {r["task_id"] for r in records} == {"ok", "overflow"}  # context overflow is the agent's failure
    assert errors == {"broken": "DeploymentStartupError"}


def test_tasks_in_preds_without_a_trajectory_are_errors(tmp_path):
    runs = tmp_path / "runs"
    write_traj(runs, "ok", 1)
    (runs / "preds.json").write_text(json.dumps({"ok": {}, "lost": {}}))
    records, _, errors = convert_arm(runs, resolved=set(), agent="a")
    assert {r["task_id"] for r in records} == {"ok"}
    assert errors == {"lost": "NoTrajectory"}


def test_tasks_the_report_did_not_score_are_left_out_not_counted_as_failures(tmp_path):
    from convert import load_report

    runs = tmp_path / "runs"
    for task in ["ok", "graded_fail", "grading_error", "missing"]:
        write_traj(runs, task, 1)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"resolved_ids": ["ok"], "unresolved_ids": ["graded_fail"],
                                  "error_ids": ["grading_error"]}))
    resolved, scored = load_report(report)
    records, _, errors = convert_arm(runs, resolved, "a", scored)
    assert {(r["task_id"], r["outcome"]) for r in records} == {("ok", 1), ("graded_fail", 0)}
    assert errors == {"grading_error": "NotScored", "missing": "NotScored"}


def test_reports_that_cannot_say_what_they_scored():
    import tempfile
    from pathlib import Path

    from convert import load_report

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.json"
        path.write_text(json.dumps({"resolved_ids": ["a"]}))
        assert load_report(path) == ({"a"}, None)
        path.write_text(json.dumps({"a": {"resolved": True}, "b": {"resolved": False}, "c": {}}))
        assert load_report(path) == ({"a"}, {"a", "b"})


def test_runs_that_lost_the_sandbox_connection_are_errors(tmp_path):
    runs = tmp_path / "runs"
    write_traj(runs, "dead", 2)
    path = runs / "dead" / "dead.traj.json"
    data = json.loads(path.read_text())
    data["messages"][-1]["extra"] = {"exception_type": "ClientOSError"}
    path.write_text(json.dumps(data))
    records, _, errors = convert_arm(runs, resolved=set(), agent="a")
    assert records == [] and errors == {"dead": "SandboxConnectionError"}


def test_step_diffs_are_written_beside_the_trajectories(tmp_path):
    from convert import collect_diffs

    runs = tmp_path / "runs"
    write_traj(runs, "t1", 1)
    write_traj(runs, "t2", 1)
    path = runs / "t1" / "t1.traj.json"
    data = json.loads(path.read_text())
    data["info"]["step_diffs"] = [{"step": 1, "diff": "+x", "returncode": 0}]
    path.write_text(json.dumps(data))
    diffs = collect_diffs(runs, "qwen-a", {"t1", "t2"})
    assert diffs == [{"task_id": "t1", "agent": "qwen-a", "step": 1, "diff": "+x", "returncode": 0}]
    write_arm([], {}, tmp_path / "out", "qwen-a", diffs=diffs)
    assert (tmp_path / "out" / "qwen-a.diffs.jsonl").read_text().count("\n") == 1
    write_arm([], {}, tmp_path / "out2", "qwen-a")
    assert not (tmp_path / "out2" / "qwen-a.diffs.jsonl").exists()
