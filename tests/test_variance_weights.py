import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "coverage"))
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor"))
import stopping
from experiments import pad_runs
from variance_weights import tracking_weights


def random_predictions(n_runs=8, seed=0):
    """Prediction rows (agent, task_id, k, f, next_cost, outcome) for runs of random length."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_runs):
        T, y = int(rng.integers(1, 12)), int(rng.integers(0, 2))
        for k in range(T):
            rows.append({"agent": "a", "task_id": f"t{i}", "k": k, "f": rng.uniform(0.01, 0.99),
                         "next_cost": rng.uniform(1, 5), "outcome": y})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed)  # order must not matter


@pytest.mark.parametrize("kappa", [0.3, 1.0, 3.0])
def test_weighted_squared_error_equals_exact_variance(kappa):
    p = random_predictions()
    w = tracking_weights(p, kappa)
    sq = (p.loc[w.index, "outcome"] - p.loc[w.index, "f"]) ** 2
    per_run = (w * sq).groupby(p.loc[w.index, "task_id"]).sum().sort_index()

    F, cost = pad_runs(p)
    variance, _ = stopping.exact_moments(F, cost, stopping.reach_tracking(F, kappa, 0.05))
    assert per_run.to_numpy() == pytest.approx(variance)
