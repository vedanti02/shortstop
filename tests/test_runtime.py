import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor"))
from runtime import Runtime
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


def replay(plan, runs):
    for task, steps in runs.items():
        run, i = plan.start(task), 0
        while run.should_continue():
            run.observe(steps[i])
            i += 1
            if i == len(steps):
                run.finish(steps[0].outcome)
                break
    return plan


def test_incremental_features_match_offline_rows():
    steps = [vars(s) for s in fake_run("t", 6, 1)]
    offline = run_rows(steps, None)
    state = RunFeatures(None)
    for k, step in enumerate(steps):
        live = state.row()
        assert all(live[c] == offline[k][c] or (np.isnan(live[c]) and np.isnan(offline[k][c])) for c in live)
        state.update(step)


def test_live_loop_policy_and_estimate():
    rt = tiny_runtime()
    runs = {f"t{i}": fake_run(f"t{i}", 3 + i % 4, i % 2) for i in range(12)}
    plan = replay(rt.plan(list(runs), seed=1), runs)
    for run in plan.runs.values():
        assert all(0.05 <= p <= 1 for p in run.reach)
        assert all(a >= b for a, b in zip(run.reach, run.reach[1:]))  # never increases
        # f_0 + sum_k (f_k - f_{k-1}) / P_k, with P_k = P(reach step k), written out directly
        direct = run.f[0] + sum((run.f[k] - run.f[k - 1]) / run.reach[k - 1] for k in range(1, len(run.reach) + 1))
        assert abs(run.estimate() - direct) < 1e-12
    res = plan.result()
    assert res["interval"][0] <= res["score"] <= res["interval"][1]
    assert res["tasks"] == 12


def test_live_loop_is_unbiased():
    rt = tiny_runtime(kappa=1.5)
    runs = {f"t{i}": fake_run(f"t{i}", 4, i % 2) for i in range(8)}
    truth = np.array([steps[0].outcome for steps in runs.values()])
    est = []
    for seed in range(600):
        plan = replay(rt.plan(list(runs), seed=seed), runs)
        est.append([plan.runs[t].estimate() for t in runs])
    est = np.array(est)
    sd = est.std(axis=0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - truth) < 5 * sd + 1e-9)
