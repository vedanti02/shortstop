import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).parent.parent / "coverage"))
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor"))
import runtime
from experiments import interval_stats
from runtime import DESIGNS, Runtime
from step_features import RunFeatures, run_rows
from train_lgbm import GROUPS

from shortstop import Step

COLS = GROUPS["gold"]


def tiny_runtime(kappa=0.6) -> Runtime:
    """A Runtime with small models fitted on random rows, enough to exercise the interface."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((400, len(COLS) + 1)), columns=[*COLS, "task_cost"])
    params = {"verbose": -1, "num_threads": 1, "min_data_in_leaf": 20}
    success = lgb.train({**params, "objective": "binary"}, lgb.Dataset(x[COLS], (x["k"] > 0.5).astype(int)), 5)
    cost = lgb.train({**params, "objective": "regression"}, lgb.Dataset(x[[*COLS, "task_cost"]], 5 + x["k"]), 5)
    meta = {"success_features": COLS, "cost_features": [*COLS, "task_cost"], "task_rate": {},
            "default_task_rate": 0.5, "task_cost": {}, "default_task_cost": 0.0, "kappa": kappa,
            "floor": 0.05, "block": 4}
    return Runtime(success, cost, meta, gold={})


def fake_run(task: str, n: int, outcome: int) -> list[Step]:
    return [Step(task, "a", i + 1, 100 * (i + 1), 20, f"```bash\npytest t{i}.py\n```",
                 "<returncode>0</returncode>\n1 passed", outcome) for i in range(n)]


def replay(plan, runs, crash=()):
    """Feed logged runs as a live harness would. A task in `crash` gets the go-ahead for its last
    step, which never completes (a context overflow), and the harness then calls finish()."""
    for task, steps in runs.items():
        run = plan.start(task)
        taken = steps[:-1] if task in crash else steps
        stopped = False
        for step in taken:
            if not run.should_continue():
                stopped = True
                break
            run.observe(step)
        if not stopped and (task not in crash or run.should_continue()):
            run.finish(steps[0].outcome)
    return plan


def save_runtime(rt: Runtime, folder: Path, gold_meta: list[str]) -> Path:
    """Write rt the way export.py does, with the given gold patch tables in meta.json."""
    folder.mkdir(parents=True)
    rt.success_models[0].save_model(str(folder / "success.txt"))
    rt.cost_model.save_model(str(folder / "cost.txt"))
    meta = {**rt.meta, "success_models": ["success.txt"], "gold_meta": gold_meta}
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


def test_incremental_features_match_offline_rows():
    steps = [vars(s) for s in fake_run("t", 6, 1)]
    offline = run_rows(steps, None)
    state = RunFeatures(None)
    for k, step in enumerate(steps):
        live = state.row()
        assert all(live[c] == offline[k][c] or (np.isnan(live[c]) and np.isnan(offline[k][c])) for c in live)
        state.update(step)


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_live_loop_policy_and_estimate(design):
    rt = tiny_runtime()
    runs = {f"t{i}": fake_run(f"t{i}", 3 + i % 4, i % 2) for i in range(12)}
    plan = replay(rt.plan(list(runs), seed=1, design=design), runs)
    for run in plan.runs.values():
        assert all(0.05 <= p <= 1 for p in run.reach)
        assert all(a >= b for a, b in zip(run.reach, run.reach[1:]))  # never increases
        # f_0 + sum_k (f_k - f_{k-1}) / P_k, with P_k = P(reach step k), written out directly
        direct = run.f[0] + sum((run.f[k] - run.f[k - 1]) / run.reach[k - 1] for k in range(1, len(run.reach) + 1))
        assert abs(run.estimate() - direct) < 1e-12
    res = plan.result()
    assert res["design"] == design
    assert res["interval"][0] <= res["score"] <= res["interval"][1]
    assert res["tasks"] == 12


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_live_loop_is_unbiased(design):
    """Also for runs whose last step never completes, which the harness ends with finish()."""
    rt = tiny_runtime(kappa=1.5)
    runs = {f"t{i}": fake_run(f"t{i}", 4, i % 2) for i in range(8)}
    crash = {"t0", "t3", "t5", "t6"}  # two failures and two successes
    truth = np.array([steps[0].outcome for steps in runs.values()])
    est = []
    for seed in range(600):
        plan = replay(rt.plan(list(runs), seed=seed, design=design), runs, crash)
        est.append([plan.runs[t].estimate() for t in runs])
    est = np.array(est)
    sd = est.std(axis=0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - truth) < 5 * sd + 1e-9)


def test_standard_design_reports_vedantis_interval():
    """The default design's interval is coverage/experiments.py's mean +/- 1.96 sd / sqrt(n)."""
    rt = tiny_runtime()
    runs = {f"t{i}": fake_run(f"t{i}", 3 + i % 4, i % 2) for i in range(12)}
    plan = replay(rt.plan(list(runs), seed=2), runs)
    res = plan.result()
    est = np.array([plan.runs[t].estimate() for t in runs])
    assert res["design"] == "standard"
    assert abs(res["score"] - est.mean()) < 1e-12
    assert abs(res["half_width"] - interval_stats(est[None, :], 0.5)["ci_half_width"]) < 1e-12


def test_unknown_design_is_rejected():
    with pytest.raises(ValueError):
        tiny_runtime().plan(["t0"], design="stratified")


def test_a_step_that_never_completes_reveals_the_outcome():
    """A context overflow on step 3: the outcome is appended as f_3, not written over f_2."""
    rt = tiny_runtime()
    plan = rt.plan(["t0"], seed=0)
    plan.u["t0"] = 0.0  # this run never stops
    run = plan.start("t0")
    for step in fake_run("t0", 2, 1):
        assert run.should_continue()
        run.observe(step)
    predictions = list(run.f)
    assert run.should_continue()  # the go-ahead for step 3, which never completes
    run.finish(1, tokens=500)
    assert run.f == [*predictions, 1.0]
    direct = run.f[0] + sum((run.f[k] - run.f[k - 1]) / run.reach[k - 1] for k in range(1, 4))
    assert abs(run.estimate() - direct) < 1e-12
    assert run.tokens == 120 + 220 + 500


def test_live_loop_refuses_misuse():
    rt = tiny_runtime()
    steps = fake_run("t0", 3, 1)
    plan = rt.plan(["t0", "t1", "t2"], seed=0)
    plan.u.update({"t0": 0.0, "t1": 0.0, "t2": 1.0})  # t0 and t1 never stop, t2 stops at once
    run = plan.start("t0")
    with pytest.raises(RuntimeError):
        run.finish(1)  # no step decided yet
    with pytest.raises(RuntimeError):
        run.observe(steps[0])  # a step taken without asking
    assert run.should_continue() and run.should_continue()  # asking again before observing ...
    assert len(run.reach) == 1  # ... does not draw again
    run.observe(steps[0])
    with pytest.raises(ValueError):
        run.finish(-1)  # "not evaluated" is not an outcome
    run.finish(1)
    with pytest.raises(RuntimeError):
        run.finish(1)
    with pytest.raises(RuntimeError):
        run.observe(steps[1])
    stopped = plan.start("t2")
    assert not stopped.should_continue()
    with pytest.raises(RuntimeError):
        stopped.finish(0)  # the outcome of a stopped run must not be used
    plan.start("t1").should_continue()  # started, then neither stopped nor finished
    with pytest.raises(RuntimeError, match="t1"):
        plan.result()


def test_calibration_moves_every_prediction_by_one_logit_shift():
    rt = tiny_runtime()
    earlier = {f"e{i}": int(i < 6) for i in range(20)}  # 30% solved
    cal = rt.calibrate(earlier)
    assert abs(np.mean([cal.predict(t, cal.tracker(t))[0] for t in earlier]) - 0.3) < 1e-8
    state = rt.tracker("x")
    for step in fake_run("x", 3, 1):
        state.update(vars(step))
    assert abs(logit(cal.predict("x", state)[0]) - logit(rt.predict("x", state)[0]) - cal.shift) < 1e-9
    assert rt.shift == 0.0 and abs(cal.calibrate(earlier).shift - cal.shift) < 1e-9  # not compounded
    runs = {f"t{i}": fake_run(f"t{i}", 3, i % 2) for i in range(6)}
    plan = replay(cal.plan(list(runs), seed=0), runs)
    assert plan.step0["t0"][0] == cal.predict("t0", cal.tracker("t0"))[0]
    assert plan.result()["shift"] == cal.shift


def test_calibration_edge_cases():
    rt = tiny_runtime()
    none_solved = rt.calibrate({f"e{i}": 0 for i in range(20)})
    assert abs(none_solved.predict("e0", none_solved.tracker("e0"))[0] - 0.5 / 20) < 1e-8  # not 0
    for bad in [{}, {"e0": -1}]:
        with pytest.raises(ValueError):
            rt.calibrate(bad)


def test_missing_gold_table_fails_loudly(tmp_path):
    folder = save_runtime(tiny_runtime(), tmp_path / "models", ["data/meta/not_there.parquet"])
    with pytest.raises(FileNotFoundError, match="not_there"):
        Runtime.load(folder)


def test_gold_table_is_read_from_the_repo_root(tmp_path, monkeypatch):
    """A relative path in meta.json is read from the repo root, whatever the working directory."""
    monkeypatch.setattr(runtime, "REPO", tmp_path)
    (tmp_path / "data" / "meta").mkdir(parents=True)
    pd.DataFrame({"instance_id": ["t0"], "repo": ["o/r"], "patch": ["--- a/x.py\n+++ b/x.py\n+y = 1\n"],
                  "test_patch": [""], "FAIL_TO_PASS": ["[]"]}).to_parquet(tmp_path / "data/meta/gold.parquet")
    folder = save_runtime(tiny_runtime(), tmp_path / "models", ["data/meta/gold.parquet"])
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert set(Runtime.load(folder).gold) == {"t0"}


def test_seed_average_predicts_the_mean():
    """A Runtime given several success models (export.py runtime --seeds N) averages them."""
    rt = tiny_runtime()
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.random((400, len(COLS))), columns=COLS)
    other = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 1, "min_data_in_leaf": 20},
                      lgb.Dataset(x, (x["k"] > 0.3).astype(int)), 5)
    both = Runtime([rt.success_models[0], other], rt.cost_model, rt.meta, gold={})
    state = rt.tracker("t0")
    for step in fake_run("t0", 3, 1):
        state.update(vars(step))
    f_first = rt.predict("t0", state)[0]
    f_other = Runtime(other, rt.cost_model, rt.meta, gold={}).predict("t0", state)[0]
    assert abs(both.predict("t0", state)[0] - (f_first + f_other) / 2) < 1e-12
    plan = replay(both.plan(["t0", "t1"], seed=0), {t: fake_run(t, 3, 1) for t in ["t0", "t1"]})
    assert plan.result()["tasks"] == 2
