"""Simulated evaluations with the ability-aware predictor, theta estimated from earlier batches.

    python predictor/ability_eval.py --floor 0.05 --prior 100

Same simulation as coverage/adaptive.py (tasks resampled with replacement, evaluated in
batches of 25, tracking policy, interval mean +/- 1.96 sd / sqrt(n)); only the predictions
differ. Before each batch, theta is the running estimate of the agent's score from the
batches already evaluated, damped toward the training agents' mean rate by `prior` tasks'
worth of weight, and the batch is scored with the predictor at that theta. Theta depends
only on earlier batches, so every task's estimate is unbiased given the past
(coverage/THEORY.md, Theorem 2b).

Modes:
    online   theta from earlier batches (the method)
    oracle   theta = the agent's true rate (a bound; uses the answer)
    average  theta = the training agents' mean rate throughout (no adaptation)

Options: --rule eb replaces the damping with a precision-weighted estimate shrunk toward the
training agents (prior variance = variance of their rates); --explore N runs the first N
tasks at --explore-floor (1.0 = to the end) before stopping gets aggressive.

Predictions come from predictor/train_ability.py's grid over theta, interpolated on the
logit scale (typical difference from the direct prediction 0.003). Output:
predictor/results/ability_<features>_f<floor>_<m<prior> | eb>[_x<N>at<floor>].csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
import stopping
from experiments import Z

RESULTS = Path(__file__).parent / "results"


def pad_grid(rows: pd.DataFrame, preds: np.ndarray):
    """(L, cost, y): L[i, k, g] is the logit of the prediction after k steps at GRID[g]."""
    rows = rows.assign(_i=np.arange(len(rows)))
    runs = [g.sort_values("k") for _, g in rows.groupby("task_id", sort=True)]
    T, G = max(len(g) for g in runs), preds.shape[1]
    L = np.empty((len(runs), T + 1, G), dtype=np.float32)
    cost = np.zeros((len(runs), T))
    y = np.empty(len(runs))
    big = 30.0  # logit standing in for a known outcome (sigmoid(30) = 1 - 1e-13)
    for i, g in enumerate(runs):
        L[i, : len(g)] = logit(np.clip(preds[g["_i"].to_numpy()], 1e-6, 1 - 1e-6))
        y[i] = g["outcome"].iloc[0]
        L[i, len(g):] = big if y[i] == 1 else -big
        cost[i, : len(g)] = g["next_cost"].to_numpy()
    return L, cost, y


def batch_predictions(L, tasks, theta, grid):
    """Predictions (R, B, T+1) for the batch's tasks, each replicate at its own theta."""
    pos = np.clip(np.interp(theta, grid, np.arange(len(grid))), 0, len(grid) - 1 - 1e-9)
    lo = pos.astype(int)[:, None, None]
    w = (pos - lo[:, 0, 0])[:, None, None]
    t3, ks = tasks[:, :, None], np.arange(L.shape[1])[None, None, :]
    lower, upper = L[t3, ks, lo], L[t3, ks, lo + 1]  # (R, B, T+1), only two grid columns
    F = expit((1 - w) * lower + w * upper)
    F[..., -1] = np.round(F[..., -1])  # the padded end is the true outcome
    return F


def eb_theta(estimates: np.ndarray, groups: list[slice], mean_theta: float, tau2: float):
    """Precision-weighted estimate of the agent's rate, shrunk toward the training agents.

    Each group of tasks evaluated under one policy contributes its mean weighted by
    n / sample variance; the prior is N(mean_theta, tau2), with tau2 the variance of the
    training agents' rates. With one group this is damping by var / tau2 tasks.
    """
    num = np.full(len(estimates), mean_theta / tau2)
    den = np.full(len(estimates), 1.0 / tau2)
    for g in groups:
        part = estimates[:, g]
        if part.shape[1] >= 2:
            precision = part.shape[1] / np.maximum(part.var(axis=1, ddof=1), 1e-3)
            num += precision * part.mean(axis=1)
            den += precision
    return num / den


def simulate(L, cost, y, grid, mode, replicates, batch, kappa, floor, prior, mean_theta, rng,
             explore=0, explore_floor=1.0, rule="damped", tau2=0.02):
    """explore: the first `explore` tasks use reach floor `explore_floor` (1.0 = run to the
    end), so theta is learned from cheap-to-trust estimates before stopping gets aggressive.
    rule: how theta is computed from earlier batches, "damped" (prior tasks' worth of
    weight on mean_theta) or "eb" (eb_theta)."""
    n, T = cost.shape
    paid_table = np.concatenate([np.zeros((n, 1)), np.cumsum(cost, axis=1)], axis=1)
    idx = rng.integers(0, n, (replicates, n))
    estimates = np.empty((replicates, n))
    paid = np.empty((replicates, n))
    theta = np.full(replicates, y.mean() if mode == "oracle" else mean_theta)
    for start in range(0, n, batch):
        tasks = idx[:, start:start + batch]
        F = batch_predictions(L, tasks, theta, grid).reshape(-1, T + 1)
        reach = stopping.reach_tracking(F, kappa, explore_floor if start < explore else floor)
        steps = stopping.sample_stop_steps(reach, rng.random(len(F)))
        rows = np.arange(len(F))
        estimates[:, start:start + batch] = stopping.estimate_table(F, reach)[rows, steps].reshape(tasks.shape)
        paid[:, start:start + batch] = paid_table[tasks.ravel(), steps].reshape(tasks.shape)
        if mode == "online":
            seen = start + batch
            if rule == "eb":
                cut = min(explore, seen)
                groups = [slice(0, cut), slice(cut, seen)]
                theta = eb_theta(estimates, groups, mean_theta, tau2)
            else:
                theta = (estimates[:, :seen].sum(axis=1) + prior * mean_theta) / (seen + prior)
            theta = np.clip(theta, grid[0], grid[-1])

    mean = estimates.mean(axis=1)
    half = Z * estimates.std(axis=1, ddof=1) / np.sqrt(n)
    full_variance, full_cost = y.var() / n, cost.sum(axis=1).mean()
    return {"coverage": float((np.abs(mean - y.mean()) <= half).mean()),
            "bias": float(mean.mean() - y.mean()),
            "miss_below": float((mean + half < y.mean()).mean()),
            "miss_above": float((mean - half > y.mean()).mean()),
            "variance_ratio": float(((half / Z) ** 2).mean() / mean.var()),
            "cost_fraction": float(paid.mean() / full_cost),
            "relative_compute": float(mean.var() * paid.mean() / (full_variance * full_cost)),
            "final_theta": float(np.median(theta))}


def run_agent(job):
    agent, rows, grid, preds, mean_theta, tau2, modes, cfg, seed = job
    L, cost, y = pad_grid(rows, preds)
    out = []
    for mode in modes:
        rng = np.random.default_rng(seed)
        stats = simulate(L, cost, y, grid, mode, cfg["replicates"], cfg["batch"], cfg["kappa"],
                         cfg["floor"], cfg["prior"], mean_theta, rng, cfg["explore"],
                         cfg["explore_floor"], cfg["rule"], tau2)
        out.append({"agent": agent, "method": mode, "true_rate": y.mean(), **stats})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", default="gold")
    parser.add_argument("--modes", default="average,online,oracle")
    parser.add_argument("--replicates", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--prior", type=float, default=100.0)
    parser.add_argument("--rule", choices=["damped", "eb"], default="damped")
    parser.add_argument("--explore", type=int, default=0, help="tasks run first at --explore-floor")
    parser.add_argument("--explore-floor", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(f"data/predictor/ability_grid_{args.features}.pkl", "rb") as f:
        grids = pickle.load(f)
    outcomes = pd.read_pickle("data/predictor/predictions_gold_loao.pkl").query("k == 0")
    rates = outcomes.groupby("agent")["outcome"].mean()
    cfg = {k: getattr(args, k) for k in ["replicates", "batch", "kappa", "floor", "prior", "rule",
                                         "explore", "explore_floor"]}
    modes = args.modes.split(",")
    jobs = []
    for i, (agent, (rows, grid, preds)) in enumerate(sorted(grids.items())):
        others = rates.drop(agent)  # training agents only
        jobs.append((agent, rows, grid, preds, float(others.mean()), float(others.var()), modes, cfg, i))
    with Pool(args.workers) as pool:
        results = pd.DataFrame([r for rows in pool.imap(run_agent, jobs) for r in rows])

    RESULTS.mkdir(exist_ok=True)
    tag = f"m{args.prior:g}" if args.rule == "damped" else "eb"
    if args.explore:
        tag += f"_x{args.explore}at{args.explore_floor:g}"
    out = RESULTS / f"ability_{args.features}_f{args.floor:g}_{tag}.csv"
    results.to_csv(out, index=False)
    summary = results.groupby("method", sort=False).agg(
        median=("relative_compute", "median"),
        geo_mean=("relative_compute", lambda r: float(np.exp(np.log(r).mean()))),
        worst=("relative_compute", "max"), coverage=("coverage", "mean"),
        worst_coverage=("coverage", "min"), miss_below=("miss_below", "mean"),
        miss_above=("miss_above", "mean"), max_abs_bias=("bias", lambda b: b.abs().max()))
    print(summary.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
