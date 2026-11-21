"""Premise check for a learned stopping policy: what is per-task, cost-aware allocation worth?

    python predictor/allocation_check.py

Vedanti's policies set the reach probability from the prediction (and a coarse state), never
from how expensive the particular run is. By Theorem 1 the optimal reach probability grows
with the expected error and shrinks with the expected cost, so a policy that knows which runs
are expensive should spend less on them. Before training a joint predictor and policy, this
measures how much that could give, with one-shot policies (each run is finished with
probability P_i, else scored by its step-0 prediction):

    uniform           P_i = q
    uncertainty       P_i = kappa * sqrt(f0 (1 - f0))
    + predicted cost  P_i = kappa * sqrt(f0 (1 - f0) / c_hat_i)
    + true cost       P_i = kappa * sqrt(f0 (1 - f0) / c_i)          (bound)

c_hat is the run's cost relative to the agent's average, predicted from how expensive the
task was for the other agents (never from the scored agent). Every P is clipped to
[0.05, 1]; kappa (or q) is tuned on the other agents. Compute is exact (no simulation), the
geometric mean over the 37 held-out agents; predictions are gold LightGBM, held out.

Output: predictor/results/allocation_check.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
import optimal_policy as op
from experiments import pad_runs

FLOOR = 0.05
KAPPAS = np.geomspace(0.01, 3.0, 40)


def geo(x) -> float:
    return float(np.exp(np.mean(np.log(x))))


def main() -> None:
    p = pd.read_pickle("data/predictor/predictions_gold_loao.pkl")
    run_cost = p.groupby(["agent", "task_id"])["next_cost"].sum().unstack("agent")  # task x agent
    log_cost = np.log(run_cost.clip(lower=1))
    agents = sorted(run_cost.columns)

    data = {}
    for a in agents:
        g = p[p["agent"] == a]
        F, cost = pad_runs(g)
        tasks = sorted(g["task_id"].unique())
        c = cost.sum(axis=1)
        c_rel = c / c.mean()
        others = log_cost.drop(columns=a)
        task_profile = others.sub(others.mean(axis=0), axis=1).mean(axis=1)  # task's relative log cost
        c_hat = np.exp(task_profile.reindex(tasks).to_numpy())
        data[a] = (F, cost, c_rel, c_hat / c_hat.mean())

    def reach_for(a, rule, k):
        F, cost, c_rel, c_hat = data[a]
        f0 = np.clip(F[:, 0], 1e-4, 1 - 1e-4)
        u = np.sqrt(f0 * (1 - f0))
        c_rel, c_hat = np.maximum(c_rel, 1e-6), np.maximum(c_hat, 1e-6)  # a free run: always finish it
        target = {"uniform": np.full(len(F), k), "uncertainty": k * u,
                  "+ predicted cost": k * u / np.sqrt(c_hat), "+ true cost": k * u / np.sqrt(c_rel),
                  "true cost only": k / np.sqrt(c_rel)}[rule]
        return np.broadcast_to(np.clip(target, FLOOR, 1.0)[:, None], cost.shape).copy()

    rules = ["uniform", "uncertainty", "+ predicted cost", "+ true cost", "true cost only"]
    table = {r: {a: [op.relative_compute(data[a][0], data[a][1], reach_for(a, r, k)) for k in KAPPAS]
                 for a in agents} for r in rules}
    rows = []
    for a in agents:
        train = [b for b in agents if b != a]
        for r in rules:
            i = int(np.argmin([geo([table[r][b][j] for b in train]) for j in range(len(KAPPAS))]))
            rows.append({"agent": a, "rule": r, "kappa": KAPPAS[i], "relative_compute": table[r][a][i]})
    d = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / "allocation_check.csv"
    d.to_csv(out, index=False)
    ref = d[d["rule"] == "uniform"].set_index("agent")["relative_compute"]
    summary = d.groupby("rule", sort=False)["relative_compute"].agg(["median", geo, "max"])
    summary["vs uniform"] = [geo(d[d["rule"] == r].set_index("agent")["relative_compute"] / ref) for r in summary.index]
    cv = np.mean([np.std(v[2]) for v in data.values()])
    corr = np.mean([np.corrcoef(np.log(v[2][v[2] > 0]), np.log(v[3][v[2] > 0]))[0, 1] for v in data.values()])
    print(f"run cost varies across tasks: mean coefficient of variation {cv:.2f}; "
          f"correlation of predicted and true log cost {corr:.2f}")
    print(summary.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
