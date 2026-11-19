"""Stopping policies fitted to minimize compute for a fixed interval width.

    python coverage/optimal_policy.py

The quantity minimized is variance x expected cost per task, relative to full runs: the
compute needed to reach a given interval width. Every policy is fitted on the other
agents and scored on the held-out one. Four policies are compared:

    one-shot           finish the run with probability q                    (q tuned)
    tracking           reach = kappa * sqrt(f (1 - f))                      (kappa tuned)
    step schedule      reach depends only on the step number; the optimum of
                       (V0 + sum_k a_k / P_k) * (sum_k c_k P_k) with P non-increasing,
                       where a_k is the mean drop in squared error at step k and c_k the
                       mean cost of step k
    learned tracking   reach = kappa * sqrt(E[(Y - f)^2 | state] / E[remaining cost | state])
                       with both expectations measured on the training agents, so it
                       does not assume the predictor is calibrated            (kappa tuned)

Output: coverage/results/policy_<split>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import stopping
from experiments import interval_stats, pad_runs, sample_estimates
from scipy.optimize import minimize

P_MIN = 0.05
STEP_EDGES = np.array([0, 1, 2, 3, 5, 7, 10, 14, 19, 26, 35, 47, 63, 84, 112, 150, 200])
F_EDGES = np.linspace(0, 1, 11)[1:-1]
KAPPAS = [0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
QS = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]


def relative_compute(F, cost, reach) -> float:
    """Compute for a fixed interval width, relative to full runs."""
    var, expected_cost = stopping.exact_moments(F, cost, reach)
    y = F[:, -1]
    return (y.var() + var.mean()) * expected_cost.mean() / (y.var() * cost.sum(axis=1).mean())


def step_bins(T: int) -> np.ndarray:
    """Bin index of steps 1..T (as positions 0..T-1)."""
    return np.searchsorted(STEP_EDGES, np.arange(T), side="right") - 1


# --- step schedule -------------------------------------------------------------------------


def schedule_stats(F, cost):
    """Per-bin sums behind a_k and c_k, plus the constant term, for one agent."""
    sq = (F[:, -1:] - F) ** 2
    bins = step_bins(cost.shape[1])
    drop = np.bincount(bins, (sq[:, :-1] - sq[:, 1:]).sum(axis=0), len(STEP_EDGES))
    spend = np.bincount(bins, cost.sum(axis=0), len(STEP_EDGES))
    return {"drop": drop, "spend": spend, "v0": (F[:, -1].var() - sq[:, 0].mean()) * len(F),
            "n": len(F)}


def schedule_from_theta(theta: np.ndarray) -> np.ndarray:
    """Non-increasing reach probabilities in [P_MIN, 1], one per step bin."""
    return P_MIN + (1 - P_MIN) * np.exp(-np.cumsum(np.logaddexp(0, theta)))


def fit_schedule(stats: list[dict]) -> np.ndarray:
    n = sum(s["n"] for s in stats)
    drop = sum(s["drop"] for s in stats) / n
    spend = sum(s["spend"] for s in stats) / n
    v0 = sum(s["v0"] for s in stats) / n

    def objective(theta):
        reach = schedule_from_theta(theta)
        return np.log(v0 + (drop / reach).sum()) + np.log((spend * reach).sum())

    best = min((minimize(objective, np.full(len(STEP_EDGES), start), method="L-BFGS-B")
                for start in (-4.0, -2.0, 0.0)), key=lambda r: r.fun)
    return schedule_from_theta(best.x)


def reach_schedule(F, schedule: np.ndarray) -> np.ndarray:
    return np.broadcast_to(schedule[step_bins(F.shape[1] - 1)], F[:, :-1].shape).copy()


# --- learned tracking ----------------------------------------------------------------------


def state_stats(F, cost, lengths):
    """Sums of squared error and remaining cost by (step bin, prediction bin)."""
    T = cost.shape[1]
    valid = np.arange(T)[None, :] < lengths[:, None]
    sq = ((F[:, -1:] - F) ** 2)[:, :-1]
    remaining = np.cumsum(cost[:, ::-1], axis=1)[:, ::-1]
    cell = (step_bins(T)[None, :] * 10 + np.digitize(F[:, :-1], F_EDGES))[valid]
    size = len(STEP_EDGES) * 10
    return {"sq": np.bincount(cell, sq[valid], size), "rem": np.bincount(cell, remaining[valid], size),
            "count": np.bincount(cell, minlength=size).astype(float)}


def fit_state_table(stats: list[dict]) -> np.ndarray:
    """sqrt(mean squared error / mean remaining cost) per state, scaled to 1 at the start."""
    sq = sum(s["sq"] for s in stats)
    rem = sum(s["rem"] for s in stats)
    count = sum(s["count"] for s in stats)
    ratio = np.sqrt((sq + 1e-9) / (rem + 1e-9))
    start = count[:10] > 0  # states at step 1
    scale = np.average(ratio[:10][start], weights=count[:10][start])
    return np.where(count > 0, ratio / scale, 1.0)


def reach_learned(F, table: np.ndarray, kappa: float) -> np.ndarray:
    cell = step_bins(F.shape[1] - 1)[None, :] * 10 + np.digitize(F[:, :-1], F_EDGES)
    return np.minimum.accumulate(np.clip(kappa * table[cell], P_MIN, 1.0), axis=1)


# --- leave-one-agent-out comparison --------------------------------------------------------


def geo_mean(values) -> float:
    return float(np.exp(np.mean(np.log(values))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--split", choices=["loao", "temporal"], default="loao")
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()

    predictions = pd.read_pickle(args.data / f"predictions_{args.split}.pkl")
    runs, lengths = {}, {}
    for agent, rows in predictions.groupby("agent", sort=True):
        runs[agent] = pad_runs(rows)
        lengths[agent] = rows.groupby("task_id", sort=True).size().to_numpy()
    agents = list(runs)
    sched = {a: schedule_stats(*runs[a]) for a in agents}
    state = {a: state_stats(*runs[a], lengths[a]) for a in agents}

    # Policies with no fitted table score the same whichever agent is held out.
    one_shot = {a: [relative_compute(*runs[a], stopping.reach_one_shot(runs[a][0], q))
                    for q in QS] for a in agents}
    tracking = {a: [relative_compute(*runs[a], stopping.reach_tracking(runs[a][0], k, P_MIN))
                    for k in KAPPAS] for a in agents}

    rng = np.random.default_rng(0)
    rows = []
    for held_out in agents:
        train = [a for a in agents if a != held_out]
        F, cost = runs[held_out]

        def tuned(scores: dict) -> int:
            """Index of the setting with the best geometric-mean score on the training agents."""
            return int(np.argmin([geo_mean([scores[a][i] for a in train])
                                  for i in range(len(scores[train[0]]))]))

        q = QS[tuned(one_shot)]
        kappa = KAPPAS[tuned(tracking)]
        schedule = fit_schedule([sched[a] for a in train])
        table = fit_state_table([state[a] for a in train])
        learned_scores = {a: [relative_compute(*runs[a], reach_learned(runs[a][0], table, k))
                              for k in KAPPAS] for a in train}
        learned_kappa = KAPPAS[tuned(learned_scores)]

        policies = {
            "one-shot": stopping.reach_one_shot(F, q),
            "tracking": stopping.reach_tracking(F, kappa, P_MIN),
            "step schedule": reach_schedule(F, schedule),
            "learned tracking": reach_learned(F, table, learned_kappa),
        }
        settings = {"one-shot": q, "tracking": kappa, "learned tracking": learned_kappa}
        idx = rng.integers(0, len(F), (args.replicates, len(F)))
        for name, reach in policies.items():
            _, expected_cost = stopping.exact_moments(F, cost, reach)
            stats = interval_stats(sample_estimates(F, reach, idx, rng), F[:, -1].mean())
            rows.append({"agent": held_out, "policy": name, "setting": settings.get(name),
                         "true_rate": F[:, -1].mean(), "coverage": stats["coverage"],
                         "cost_fraction": expected_cost.mean() / cost.sum(axis=1).mean(),
                         "relative_compute": relative_compute(F, cost, reach)})
        print(f"done {held_out}", flush=True)

    results = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / f"policy_{args.split}.csv"
    results.to_csv(out, index=False)
    summary = results.groupby("policy", sort=False).agg(
        median=("relative_compute", "median"), geo_mean=("relative_compute", geo_mean),
        worst=("relative_compute", "max"),
        worse_than_full=("relative_compute", lambda r: int((r > 1).sum())),
        cost_fraction=("cost_fraction", "mean"), coverage=("coverage", "mean"),
        worst_coverage=("coverage", "min"))
    print(summary.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
