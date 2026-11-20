"""Stopping policies fitted to minimize compute for a fixed interval width.

    python coverage/optimal_policy.py

The quantity minimized is variance x expected cost per task, relative to full runs: the
compute needed to reach a given interval width. Every policy is fitted on the other
agents and scored on the held-out one. Five policies are compared:

    one-shot           finish the run with probability q                    (q tuned)
    tracking           reach = kappa * sqrt(f (1 - f))                      (kappa tuned)
    step schedule      reach depends only on the step number; the optimum of
                       (V0 + sum_k a_k / P_k) * (sum_k c_k P_k) with P non-increasing,
                       where a_k is the mean drop in squared error at step k and c_k the
                       mean cost of step k
    learned tracking   reach = kappa * sqrt(E[(Y - f)^2 | state] / E[remaining cost | state])
                       with both expectations measured on the training agents, so it
                       does not assume the predictor is calibrated            (kappa tuned)
    index              reach = kappa * sqrt(index of the state), where the index is the
                       best ratio, over every rule for when to stop looking ahead, of the
                       expected drop in squared error to the expected cost. This is the
                       optimal policy when the state is Markov (see THEORY.md); learned
                       tracking is the special case that always looks ahead to the end
                                                                              (kappa tuned)

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
INDEX_KAPPAS = list(np.geomspace(0.02, 3.0, 14).round(4))
MU_GRID = np.geomspace(1e-4, 1e4, 193)  # candidate index values, in units of 1 / mean run cost
SHRINK = 20.0  # runs' worth of pull toward the step average, for sparsely visited states


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


# --- index policy ---------------------------------------------------------------------------


def index_stats(F, cost, lengths, T_max: int) -> dict:
    """Sums by (step, prediction bin) for runs still going, and transitions to the next bin."""
    T = cost.shape[1]
    alive = np.arange(T)[None, :] < lengths[:, None]
    sq = (F[:, -1:] - F) ** 2
    bins = np.digitize(F[:, :-1], F_EDGES)
    step = np.broadcast_to(np.arange(T), bins.shape)
    cell = (step * 10 + bins)[alive]
    both = alive[:, :-1] & alive[:, 1:]
    pair = (step[:, :-1] * 100 + bins[:, :-1] * 10 + bins[:, 1:])[both]
    return {"drop": np.bincount(cell, (sq[:, :-1] - sq[:, 1:])[alive], T_max * 10).reshape(T_max, 10),
            "spend": np.bincount(cell, cost[alive], T_max * 10).reshape(T_max, 10),
            "count": np.bincount(cell, minlength=T_max * 10).reshape(T_max, 10).astype(float),
            "trans": np.bincount(pair, minlength=T_max * 100).reshape(T_max, 10, 10).astype(float)}


def fit_index_table(stats: list[dict], shrink: float = SHRINK) -> np.ndarray:
    """Index of each (step, prediction bin) state, by backward induction.

    For a price mu per unit of cost, M[state] is the best expected value of
    (drop in squared error - mu * cost) over the next steps, stopping the look-ahead when
    it turns negative. The index is the price at which M is zero.
    """
    drop, spend, count, trans = (sum(s[key] for s in stats) for key in ("drop", "spend", "count", "trans"))
    spend = spend / (spend.sum() / count[0].sum())  # cost in units of the mean run cost
    index = np.zeros(count.shape)
    M = np.zeros((10, len(MU_GRID)))
    for k in reversed(range(len(count))):
        total = drop[k][:, None] - spend[k][:, None] * MU_GRID + trans[k] @ np.maximum(M, 0)
        step_mean = total.sum(axis=0) / max(count[k].sum(), 1.0)
        M = (total + shrink * step_mean) / np.maximum(count[k][:, None] + shrink, 1e-12)
        # M decreases in mu and is piecewise linear: interpolate its zero crossing.
        last = (M > 0).sum(axis=1) - 1
        lo, hi = np.clip(last, 0, len(MU_GRID) - 2), np.clip(last, 0, len(MU_GRID) - 2) + 1
        rows = np.arange(10)
        m_lo, m_hi = M[rows, lo], M[rows, hi]
        crossing = MU_GRID[lo] + m_lo * (MU_GRID[hi] - MU_GRID[lo]) / np.maximum(m_lo - m_hi, 1e-300)
        index[k] = np.where(last < 0, 0.0, np.where(last >= len(MU_GRID) - 1, MU_GRID[-1], crossing))
    return index


def reach_index(F, index: np.ndarray, kappa: float, p_min: float = P_MIN) -> np.ndarray:
    T = F.shape[1] - 1
    target = kappa * np.sqrt(index[np.arange(T)[None, :], np.digitize(F[:, :-1], F_EDGES)])
    return np.minimum.accumulate(np.clip(target, p_min, 1.0), axis=1)


# --- leave-one-agent-out comparison --------------------------------------------------------


def geo_mean(values) -> float:
    return float(np.exp(np.mean(np.log(values))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--split", choices=["loao", "temporal", "lofo"], default="loao")
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
    T_max = max(runs[a][1].shape[1] for a in agents)
    index = {a: index_stats(*runs[a], lengths[a], T_max) for a in agents}

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

        def tuned(scores: dict, train=train) -> int:
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
        index_table = fit_index_table([index[a] for a in train])
        index_scores = {a: [relative_compute(*runs[a], reach_index(runs[a][0], index_table, k))
                            for k in INDEX_KAPPAS] for a in train}
        index_kappa = INDEX_KAPPAS[tuned(index_scores)]

        policies = {
            "one-shot": stopping.reach_one_shot(F, q),
            "tracking": stopping.reach_tracking(F, kappa, P_MIN),
            "step schedule": reach_schedule(F, schedule),
            "learned tracking": reach_learned(F, table, learned_kappa),
            "index": reach_index(F, index_table, index_kappa),
        }
        settings = {"one-shot": q, "tracking": kappa, "learned tracking": learned_kappa,
                    "index": index_kappa}
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
