"""predictor/live_monitor.py: record checks, estimates, the live status and the final report."""

import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor"))
import live_monitor as lm
from plugin_adapter import LiveSession
from test_plugin_adapter import PluginLoop, runs_for
from test_runtime import tiny_runtime

from shortstop import rr_estimate


def write_traj(folder: Path, task: str, record: dict, exit_status: str, tokens: int = 1000) -> None:
    (folder / task).mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": "x"},
                {"role": "assistant", "content": "a", "usage": {"prompt_tokens": tokens, "completion_tokens": 10}},
                {"role": "user", "content": "failed", "extra": {"response": {"usage": {"prompt_tokens": 5,
                                                                                       "completion_tokens": 1}}}}]
    traj = {"instance_id": task, "info": {"exit_status": exit_status, "shortstop": record}, "messages": messages}
    (folder / task / f"{task}.traj.json").write_text(json.dumps(traj))


def live_batch(tmp_path: Path, n: int = 6):
    """A batch through the adapter: n - 1 runs ended (trajectories written), one still running."""
    runs = runs_for(n)
    log = tmp_path / "a.decisions.jsonl"
    session = LiveSession(tiny_runtime(), list(runs), log=log, arm="a")
    folder, records = tmp_path / "runs", {}
    for i, (task, steps) in enumerate(runs.items()):
        loop = PluginLoop(session.predict, session.continue_probability, task, 0)
        if i == n - 1:
            loop.u = -1.0
            loop.run(steps[:2])  # two steps in, no trajectory yet
            continue
        records[task] = loop.run(steps)
        write_traj(folder, task, records[task], "EarlyStopped" if records[task]["stopped_early"] else "Submitted")
    grades = {"resolved_ids": [t for t, s in runs.items() if s[0].outcome == 1],
              "unresolved_ids": [t for t, s in runs.items() if s[0].outcome == 0]}
    (tmp_path / "eval.json").write_text(json.dumps(grades))
    return log, folder, records, runs


def test_check_record_catches_each_problem():
    good = {"predictions": [0.6, 0.5], "continue_probs": [0.8, 0.9], "stopped_early": False, "s": 2}
    assert lm.check_record(good, 1) == []
    cases = {"predictions but": {**good, "predictions": [0.6]},
             "a prediction outside": {**good, "predictions": [0.6, 1.2]},
             "a continue probability outside": {**good, "continue_probs": [0.8, 0.0]},
             "below the floor": {**good, "continue_probs": [0.2, 0.2]},
             "s = 1": {**good, "s": 1},
             "no draw": {**good, "predictions": [], "continue_probs": [], "s": 0}}
    for message, record in cases.items():
        assert any(message in p for p in lm.check_record(record, 1)), message
    assert any("outcome -1" in p for p in lm.check_record(good, -1))
    assert any("outcome 1" in p for p in lm.check_record({**good, "stopped_early": True, "s": 1}, 1))


def test_record_estimate_for_each_way_a_run_ends():
    f, p = [0.6, 0.5, 0.7], [0.8, 0.5, 0.9]
    stopped = {"predictions": f, "continue_probs": p, "stopped_early": True}  # the third draw said stop
    assert lm.record_estimate(stopped, None) == rr_estimate(f, p, 2)
    # three steps run, then the outcome; a third step that got the go-ahead but never completed
    # leaves the same record, and the outcome stands in for its prediction in the same way
    ended = {"predictions": f, "continue_probs": p, "stopped_early": False}
    assert lm.record_estimate(ended, 1) == rr_estimate([*f, 1.0], p, 3)
    with pytest.raises(ValueError, match="outcome"):
        lm.record_estimate(ended, None)


def test_status_of_a_clean_batch(tmp_path):
    log, folder, records, runs = live_batch(tmp_path)
    s = lm.status(log, folder, grades=tmp_path / "eval.json")
    assert (s["planned"], s["started"], s["ended"], s["running"]) == (6, 6, 5, 1)
    assert [a for a in s["alerts"] if a["level"] != "info"] == []
    expected = [lm.record_estimate(r, None if r["stopped_early"] else runs[t][0].outcome) for t, r in records.items()]
    assert s["scored"] == 5 and abs(s["score"] - np.mean(expected)) < 1e-12 and s["awaiting_grades"] == 0
    assert s["tokens_spent"] >= 5 * 1016  # the failed call's tokens count too
    assert lm.exit_code(s["alerts"], strict=True) == 0
    assert "running score" in lm.render_status(s)
    ungraded = lm.status(log, folder)
    assert ungraded["awaiting_grades"] == sum(not r["stopped_early"] for r in records.values())


def test_status_raises_each_alarm(tmp_path):
    log, folder, records, _ = live_batch(tmp_path)
    task = next(t for t, r in records.items() if r["predictions"][0] < 0.99)
    tampered = {**records[task], "predictions": [records[task]["predictions"][0] + 1e-6, *records[task]["predictions"][1:]]}
    write_traj(folder, task, tampered, "EarlyStopped" if tampered["stopped_early"] else "Submitted")
    write_traj(folder, "outsider", records[task], "SandboxConnectionError")
    with log.open("a") as fh:
        fh.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "session": "x", "event": "error",
                             "where": "predictor", "task_id": "t1", "k": 2, "error": "ValueError('boom')"}) + "\n")
        fh.write('{"time": "half a li')  # the adapter is still writing: skipped
    s = lm.status(log, folder, latency_ms=0.0, now=datetime.now(timezone.utc) + timedelta(hours=2))
    checks = {(a["level"], a["check"]) for a in s["alerts"]}
    for expected in [("critical", "errors"), ("critical", "records vs log"), ("warning", "stalled"),
                     ("warning", "latency"), ("warning", "endings"), ("warning", "tasks")]:
        assert expected in checks, expected
    assert lm.exit_code(s["alerts"], strict=False) == 2


def test_status_flags_predictions_that_do_not_fit_the_agent(tmp_path):
    """A predictor that expects 90% solved for an agent that solves none, on 25 runs."""
    log, folder = tmp_path / "log.jsonl", tmp_path / "runs"
    tasks = [f"t{i}" for i in range(25)]
    t = datetime.now(timezone.utc).isoformat()
    lines = [{"time": t, "session": "s", "event": "session", "arm": "a", "tasks": 25, "design": "standard", "shift": 0.0,
              "kappa": 0.4, "floor": 0.05, "cost_scale": 1.0, "success_models": 1, "mean_f0": 0.9,
              "predicted_full_tokens": 1e6, "task_ids": tasks}]
    for task in tasks:
        lines.append({"time": t, "session": "s", "event": "run_start", "task_id": task, "attempt": 1, "f0": 0.9})
        lines.append({"time": t, "session": "s", "event": "decision", "task_id": task, "attempt": 1, "k": 1, "f": 0.9,
                      "p": 1.0, "reach": 1.0, "at_floor": False, "tokens_so_far": 0, "predict_ms": 1, "total_ms": 1})
        write_traj(folder, task, {"predictions": [0.9], "continue_probs": [1.0], "stopped_early": False, "s": 1}, "Submitted")
    log.write_text("".join(json.dumps(x) + "\n" for x in lines))
    grades = tmp_path / "eval.json"
    grades.write_text(json.dumps({"resolved_ids": [], "unresolved_ids": tasks}))
    s = lm.status(log, folder, grades=grades)
    assert ("warning", "calibration") in {(a["level"], a["check"]) for a in s["alerts"]}
    assert lm.exit_code(s["alerts"], strict=True) == 1 and lm.exit_code(s["alerts"], strict=False) == 0


def converted_arm(folder: Path, agent: str, runs: dict, records: dict) -> None:
    """Write what tool_ab/convert.py writes: one line per step, the stop records, the errors."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, steps in runs.items():
        rec = records[task] if records else None
        n = (len(rec["continue_probs"]) - rec["stopped_early"]) if rec else len(steps)
        outcome = -1 if rec and rec["stopped_early"] else steps[0].outcome
        rows += [{"task_id": task, "agent": agent, "step": i + 1, "tokens_in": 100, "tokens_out": 10, "action": "",
                  "observation": "", "outcome": outcome} for i in range(max(min(n, len(steps)), 1))]
    (folder / f"{agent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if records:
        (folder / f"{agent}.stops.json").write_text(json.dumps(records))
    (folder / f"{agent}.errors.json").write_text("{}")


def two_arms(tmp_path: Path, seed_b: int = 0):
    base = runs_for(12)
    other = {t: [replace(x, outcome=1 - x.outcome) for x in s] if i % 3 == 0 else s
             for i, (t, s) in enumerate(base.items())}
    session = LiveSession(tiny_runtime(), list(base))
    rec_a = {t: PluginLoop(session.predict, session.continue_probability, t, 0).run(s) for t, s in base.items()}
    rec_b = {t: {**PluginLoop(session.predict, session.continue_probability, t, 0).run(s), "seed": seed_b}
             for t, s in other.items()}
    data, truth = tmp_path / "data", tmp_path / "truth"
    converted_arm(data, "qwen-a", base, rec_a)
    converted_arm(data, "qwen-b", other, rec_b)
    converted_arm(truth, "qwen-a", base, {})
    converted_arm(truth, "qwen-b", other, {})
    return data, truth, base, other, rec_a, rec_b


def test_report_scores_each_arm_and_the_difference(tmp_path):
    data, truth, base, other, rec_a, rec_b = two_arms(tmp_path)
    r = lm.report(data, ["a", "b"], baseline="a", truth=truth)
    est = {name: {t: lm.record_estimate(rec[t], None if rec[t]["stopped_early"] else runs[t][0].outcome) for t in runs}
           for name, runs, rec in [("a", base, rec_a), ("b", other, rec_b)]}
    for name in ("a", "b"):
        assert abs(r["arms"][name]["score"] - np.mean(list(est[name].values()))) < 1e-12
        assert r["arms"][name]["tasks"] == 12
    d = [est["b"][t] - est["a"][t] for t in base]
    assert abs(r["paired"]["b"]["difference"] - np.mean(d)) < 1e-12
    assert r["arms"]["a"]["true_rate"] == np.mean([s[0].outcome for s in base.values()])
    assert abs(r["paired"]["b"]["true_difference"] - np.mean([other[t][0].outcome - base[t][0].outcome for t in base])) < 1e-12
    assert 0 < r["arms"]["a"]["tokens_share"] <= 1
    assert [a["level"] for a in r["alerts"]] == ["info"]
    assert "| b |" in lm.render_report(r)


def test_report_checks_records_and_seeds(tmp_path, capsys):
    data, _, _, _, rec_a, _ = two_arms(tmp_path, seed_b=1)
    broken = dict(rec_a)
    task = next(iter(broken))
    broken[task] = {**broken[task], "s": broken[task]["s"] + 1}
    (data / "qwen-a.stops.json").write_text(json.dumps(broken))
    code = lm.main(["report", "--data", str(data), "--arms", "a", "b", "--baseline", "a", "--json"])
    out = json.loads(capsys.readouterr().out)
    checks = {(a["level"], a["check"]) for a in out["alerts"]}
    assert ("critical", "a records") in checks and ("warning", "seeds") in checks
    assert code == 2 and out["arms"]["a"]["tasks"] == 11


def test_status_command_prints_and_exits(tmp_path, capsys):
    log, folder, _, _ = live_batch(tmp_path)
    assert lm.main(["status", "--log", str(log), "--runs", str(folder)]) == 0
    assert "planned" in capsys.readouterr().out
    assert lm.main(["status", "--log", str(tmp_path / "missing.jsonl")]) == 2
