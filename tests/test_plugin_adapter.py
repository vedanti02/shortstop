"""The adapter between Rishi's plug-in and the runtime (predictor/plugin_adapter.py).

PluginLoop below is the decision loop of StoppingAgent in tool_ab/shortstop_plugin/agent.py
(PR #3 at 1e75aa1): the same test for a task_id parameter, one uniform per task drawn from
random.Random(f"{seed}:{task_id}"), and a step taken while that uniform stays below the product
of the continue probabilities. predictor/plugin_rehearsal.py runs the real plug-in instead.
"""

import inspect
import json
import random
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "predictor"))
import plugin_adapter
from live_monitor import check_record, record_estimate
from plugin_adapter import LiveSession, read_outcomes, slice_tasks
from test_runtime import fake_run, save_runtime, tiny_runtime


def accepts_task_id(fn) -> bool:
    """tool_ab/shortstop_plugin/agent.py: _accepts_task_id."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "task_id" or p.kind is p.VAR_KEYWORD for p in params)


class PluginLoop:
    """StoppingAgent's draws for one run, with the steps taken from a logged run."""

    def __init__(self, predictor, policy, task_id: str, seed: int):
        self.predictor, self.policy, self.task_id = predictor, policy, task_id
        self.u, self.reach = random.Random(f"{seed}:{task_id}").random(), 1.0
        self.predictions, self.continue_probs, self.stopped_early = [], [], False

    def call(self, fn, *args) -> float:
        return float(fn(*args, **({"task_id": self.task_id} if accepts_task_id(fn) else {})))

    def decide(self, prefix) -> bool:
        f = self.call(self.predictor, prefix)
        assert 0 <= f <= 1
        p = self.call(self.policy, prefix, (*self.predictions, f))
        assert 0 < p <= 1
        self.predictions.append(f)
        self.continue_probs.append(p)
        self.reach *= p
        return self.u < self.reach

    def run(self, steps, crash: bool = False) -> dict:
        """crash: the last step gets its draw but never completes (a context overflow)."""
        taken = steps[:-1] if crash else steps
        for k in range(len(taken) + int(crash)):
            if not self.decide(steps[:k]):
                self.stopped_early = True
                break
        return {"s": len(self.continue_probs) - int(self.stopped_early), "stopped_early": self.stopped_early,
                "predictions": list(self.predictions), "continue_probs": list(self.continue_probs), "seed": 0}


def runs_for(n_tasks: int = 8, n_steps=None) -> dict:
    return {f"t{i}": fake_run(f"t{i}", n_steps or 3 + i % 4, i % 2) for i in range(n_tasks)}


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def fresh_sessions():
    yield
    for session in plugin_adapter._SESSIONS.values():
        session.close()
    plugin_adapter._SESSIONS.clear()


def test_adapter_answers_exactly_as_the_runtime():
    rt = tiny_runtime()
    runs = runs_for()
    session = LiveSession(rt, list(runs))
    plan = rt.plan(list(runs), design="standard")
    for task, steps in runs.items():
        run = plan.start(task)
        run.u = -1.0  # never stops, so every step is compared
        preds, reach = [], 1.0
        for k in range(len(steps)):
            f = session.predict(steps[:k], task_id=task)
            preds.append(f)
            p = session.continue_probability(steps[:k], preds, task_id=task)
            assert run.should_continue()
            assert f == run.f[-1]
            reach *= p
            assert abs(reach - run.reach[-1]) < 1e-12
            assert session._runs[task].reach[-1] == run.reach[-1]
            run.observe(steps[k])


def test_plugin_passes_the_task_id_to_both():
    session = LiveSession(tiny_runtime(), ["t0"])
    assert accepts_task_id(session.predict) and accepts_task_id(session.continue_probability)


def test_estimates_from_plugin_records_are_unbiased():
    """Over the plug-in's own draws, including runs whose last step never completes."""
    rt = tiny_runtime(kappa=1.5)
    runs = runs_for(8, n_steps=4)
    crash = {"t0", "t3", "t5", "t6"}
    session = LiveSession(rt, list(runs))  # one session: each replay is a new attempt
    truth = np.array([steps[0].outcome for steps in runs.values()])
    est = []
    for seed in range(400):
        row = []
        for task, steps in runs.items():
            record = PluginLoop(session.predict, session.continue_probability, task, seed).run(steps, task in crash)
            assert check_record(record, None if not record["stopped_early"] else -1) == []
            row.append(record_estimate(record, None if record["stopped_early"] else steps[0].outcome))
        est.append(row)
    est = np.array(est)
    sd = est.std(axis=0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - truth) < 5 * sd + 1e-9)
    assert est.std(axis=0).max() > 0  # the draws really vary


def test_records_give_the_runtimes_estimate_for_the_same_uniforms():
    rt = tiny_runtime()
    runs = runs_for(10)
    session = LiveSession(rt, list(runs))
    for seed in range(5):
        plan = rt.plan(list(runs), design="standard")
        for task, steps in runs.items():
            loop = PluginLoop(session.predict, session.continue_probability, task, seed)
            record = loop.run(steps)
            run = plan.start(task)
            run.u = loop.u
            for step in steps:
                if not run.should_continue():
                    break
                run.observe(step)
            else:
                run.finish(steps[0].outcome)
            assert record["s"] == len(run.reach)
            mine = record_estimate(record, None if record["stopped_early"] else steps[0].outcome)
            assert abs(mine - run.estimate()) < 1e-9


def test_decision_log_has_one_line_per_decision(tmp_path):
    rt = tiny_runtime()
    runs = runs_for(3)
    log = tmp_path / "live" / "a.decisions.jsonl"
    session = LiveSession(rt, list(runs), log=log, arm="a")
    records = {t: PluginLoop(session.predict, session.continue_probability, t, 0).run(s) for t, s in runs.items()}
    lines = events(log)  # readable while the session is open: each line is flushed
    assert lines[0]["event"] == "session" and lines[0]["tasks"] == 3 and lines[0]["arm"] == "a"
    assert {e["session"] for e in lines} == {session.id}
    decisions = [e for e in lines if e["event"] == "decision"]
    assert len(decisions) == sum(len(r["continue_probs"]) for r in records.values())
    assert [e["event"] for e in lines].count("run_start") == 3
    for task, record in records.items():
        mine = [e for e in decisions if e["task_id"] == task]
        assert [e["f"] for e in mine] == record["predictions"]
        assert [e["p"] for e in mine] == record["continue_probs"]
        assert [e["k"] for e in mine] == list(range(1, len(mine) + 1))
    assert all(e["total_ms"] >= e["predict_ms"] >= 0 and 0.05 <= e["reach"] <= 1 for e in decisions)


def test_a_rerun_starts_a_new_attempt(tmp_path):
    rt = tiny_runtime()
    steps = fake_run("t0", 5, 1)
    log = tmp_path / "log.jsonl"
    session = LiveSession(rt, ["t0"], log=log)
    first = PluginLoop(session.predict, session.continue_probability, "t0", 3)
    first.u = -1.0
    first.run(steps[:3])  # stopped by an infrastructure error after 3 steps, then rerun
    second = PluginLoop(session.predict, session.continue_probability, "t0", 3).run(steps)
    starts = [e for e in events(log) if e["event"] == "run_start"]
    assert [e["attempt"] for e in starts] == [1, 2]
    assert session._runs["t0"].attempt == 2
    assert len(session._runs["t0"].reach) == len(second["continue_probs"])


def test_misuse_fails_loudly_and_is_logged(tmp_path):
    rt = tiny_runtime()
    steps = fake_run("t0", 3, 0)
    log = tmp_path / "log.jsonl"
    session = LiveSession(rt, ["t0"], log=log)
    with pytest.raises(ValueError, match="task ID"):
        session.predict([])
    with pytest.raises(KeyError, match="not in this batch"):
        session.predict([], task_id="other")
    with pytest.raises(RuntimeError, match="before the predictor"):
        session.continue_probability(steps[:1], [0.5, 0.5], task_id="t0")
    f = session.predict([], task_id="t0")
    with pytest.raises(RuntimeError, match="do not end with"):
        session.continue_probability([], [f + 0.1], task_id="t0")
    errors = [e for e in events(log) if e["event"] == "error"]
    assert {e["where"] for e in errors} == {"predictor", "stop_policy"}


def test_settings_are_checked(tmp_path):
    folder = save_runtime(tiny_runtime(), tmp_path / "rt", [])
    base = {"runtime": str(folder), "tasks": ["t0", "t1", "t2"], "log": str(tmp_path / "x.jsonl")}
    with pytest.raises(TypeError, match="unknown setting"):
        LiveSession.from_config({**base, "p": 0.5})
    with pytest.raises(ValueError, match="only design 'standard'"):
        LiveSession.from_config({**base, "design": "balanced"})
    with pytest.raises(TypeError, match="exactly one"):
        LiveSession.from_config({**base, "tasks_file": "x.txt"})
    with pytest.raises(ValueError, match="each listed once"):
        LiveSession(tiny_runtime(), ["t0", "t0"])
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("# header\nt0\nt1\n\nt2\nt3\n")
    s = LiveSession.from_config({**{k: v for k, v in base.items() if k != "tasks"},
                                 "tasks_file": str(tasks_file), "slice": "1:3"})
    assert s.summary["task_ids"] == ["t1", "t2"] and slice_tasks(["a", "b", "c"], ":2") == ["a", "b"]


def test_calibration_from_earlier_runs(tmp_path):
    folder = save_runtime(tiny_runtime(), tmp_path / "rt", [])
    as_json = tmp_path / "earlier.json"
    as_json.write_text(json.dumps({"t0": 1, "t1": 0, "t2": 0, "t3": 0}))
    as_jsonl = tmp_path / "earlier.jsonl"
    rows = [{"task_id": t, "step": 1, "outcome": y} for t, y in [("t0", 1), ("t1", 0), ("t2", 0), ("t3", 0), ("t4", -1)]]
    as_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert read_outcomes(as_json) == read_outcomes(as_jsonl) == {"t0": 1, "t1": 0, "t2": 0, "t3": 0}
    expected = tiny_runtime().calibrate({"t0": 1, "t1": 0, "t2": 0, "t3": 0}).shift
    for path in (as_json, as_jsonl):
        s = LiveSession.from_config({"runtime": str(folder), "tasks": ["t0", "t5"], "calibrate_from": str(path),
                                     "log": str(tmp_path / f"{path.suffix}.jsonl")})
        assert abs(s.rt.shift - expected) < 1e-12 and s.summary["calibrated_on"] == 4


def test_one_session_per_settings_shared_across_threads(tmp_path):
    folder = save_runtime(tiny_runtime(), tmp_path / "rt", [])
    runs = runs_for(16)
    settings = {"runtime": str(folder), "tasks": list(runs), "log": str(tmp_path / "live.jsonl")}
    predictor, policy = plugin_adapter.predictor(**settings), plugin_adapter.policy(**settings)
    assert predictor.__self__ is policy.__self__  # one session: one plan, one log
    sequential = {t: PluginLoop(predictor, policy, t, 1).run(s) for t, s in runs.items()}
    threaded, failures = {}, []

    def work(task):
        try:
            threaded[task] = PluginLoop(predictor, policy, task, 1).run(runs[task])
        except Exception as e:  # noqa: BLE001 -- reported below
            failures.append(e)

    threads = [threading.Thread(target=work, args=(t,)) for t in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures and threaded == sequential


def test_the_adapter_loads_without_offline_only_packages():
    """The plug-in's environment has mini-swe-agent and the runtime's packages, not scikit-learn,
    which only the offline training scripts use."""
    code = ("import sys; sys.modules['sklearn'] = None; sys.path.insert(0, 'predictor'); "
            "import plugin_adapter, live_monitor")
    subprocess.run([sys.executable, "-I", "-c", code], cwd=REPO, check=True)


def test_a_policy_without_settings_uses_the_predictors_session(tmp_path):
    folder = save_runtime(tiny_runtime(), tmp_path / "rt", [])
    runs = runs_for(4)
    lone = plugin_adapter.policy()  # built first, as the plug-in's preload does
    with pytest.raises(RuntimeError, match="has 0"):
        lone([], [0.5], task_id="t0")
    settings = {"runtime": str(folder), "tasks": list(runs), "log": str(tmp_path / "a.jsonl")}
    predictor = plugin_adapter.predictor(**settings)
    shared = plugin_adapter.policy(**settings)
    for task, steps in runs.items():
        assert PluginLoop(predictor, lone, task, 2).run(steps) == PluginLoop(predictor, shared, task, 2).run(steps)
    plugin_adapter.predictor(**{**settings, "log": str(tmp_path / "b.jsonl")})  # a second session
    with pytest.raises(RuntimeError, match="has 2"):
        PluginLoop(predictor, lone, "t0", 2).run(runs["t0"])
