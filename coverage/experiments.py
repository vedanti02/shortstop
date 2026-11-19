"""Coverage and cost of early-stopping methods, simulated offline on logged runs.

    python coverage/experiments.py                   # uses predictions_loao.pkl
    python coverage/experiments.py --split temporal

For each agent, the target is its resolve rate on all tasks. Each replicate resamples the
tasks with replacement, applies a stopping method, and builds a 95% interval as
mean +/- 1.96 * sd / sqrt(n). Coverage is how often that interval contains the target.
Output: coverage/results/results_<split>.csv, one row per agent and method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import stopping

Z = 1.96
THRESHOLDS = [(0.02, 0.98), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]

# name -> (kind, function of F). Deterministic methods return (label, cost paid);
# randomized ones return reach probabilities.
METHODS = {"full runs": ("randomized", stopping.reach_full)}
for lo, hi in THRESHOLDS:
    METHODS[f"deterministic {lo}/{hi}"] = (
        "deterministic", lambda F, cost, lo=lo, hi=hi: stopping.deterministic_stop(F, cost, lo, hi))
    METHODS[f"randomized threshold {lo}/{hi} q=0.2"] = (
        "randomized", lambda F, lo=lo, hi=hi: stopping.reach_threshold(F, lo, hi, 0.2))
for q in [0.2, 0.5]:
    METHODS[f"one-shot q={q}"] = ("randomized", lambda F, q=q: stopping.reach_one_shot(F, q))
for kappa in [0.5, 1.0, 1.5, 2.0, 3.0]:
    METHODS[f"tracking kappa={kappa}"] = (
        "randomized", lambda F, kappa=kappa: stopping.reach_tracking(F, kappa, 0.05))
# A lower floor allows weights up to 100, a stress test for the normal interval.
METHODS["tracking kappa=0.5 floor=0.01"] = (
    "randomized", lambda F: stopping.reach_tracking(F, 0.5, 0.01))


def pad_runs(agent_rows: pd.DataFrame):
    """Padded (F, cost) for one agent's prediction rows. See stopping.py for the layout."""
    runs = [g.sort_values("k") for _, g in agent_rows.groupby("task_id", sort=True)]
    T = max(len(g) for g in runs)
    F = np.empty((len(runs), T + 1))
    cost = np.zeros((len(runs), T))
    for i, g in enumerate(runs):
        F[i, : len(g)] = g["f"].to_numpy()
        F[i, len(g):] = g["outcome"].iloc[0]
        cost[i, : len(g)] = g["next_cost"].to_numpy()
    return F, cost


def interval_stats(estimates: np.ndarray, target: float) -> dict:
    """Coverage and width from per-task estimates of shape (replicates, n)."""
    mean = estimates.mean(axis=1)
    half = Z * estimates.std(axis=1, ddof=1) / np.sqrt(estimates.shape[1])
    return {"coverage": float((np.abs(mean - target) <= half).mean()),
            "ci_half_width": float(half.mean()),
            "bias": float(mean.mean() - target),
            "rmse": float(np.sqrt(((mean - target) ** 2).mean()))}


def sample_estimates(F: np.ndarray, reach: np.ndarray, idx: np.ndarray, rng) -> np.ndarray:
    """Randomized-truncation estimates for the resampled tasks idx (replicates, n)."""
    table = stopping.estimate_table(F, reach)
    estimates = np.empty(idx.shape)
    for start in range(0, len(idx), 50):  # chunked to bound memory
        rows_idx = idx[start:start + 50]
        steps = stopping.sample_stop_steps(reach[rows_idx], rng.random(rows_idx.shape))
        estimates[start:start + 50] = table[rows_idx, steps]
    return estimates


def evaluate_agent(F: np.ndarray, cost: np.ndarray, replicates: int, rng,
                   n_tasks: int | None = None, methods: dict | None = None) -> list[dict]:
    y = F[:, -1]
    target = y.mean()
    full_cost = cost.sum(axis=1).mean()
    idx = rng.integers(0, len(F), (replicates, n_tasks or len(F)))
    rows = []
    for name, (kind, fn) in (methods or METHODS).items():
        if kind == "deterministic":
            label, paid = fn(F, cost)
            stats = interval_stats(label[idx], target)
            mean_cost, variance = paid.mean(), label.var()
        else:
            reach = fn(F)
            stats = interval_stats(sample_estimates(F, reach, idx, rng), target)
            run_var, run_cost = stopping.exact_moments(F, cost, reach)
            mean_cost, variance = run_cost.mean(), y.var() + run_var.mean()
        rows.append({"method": name, "kind": kind, "true_rate": target, **stats,
                     "cost_fraction": mean_cost / full_cost,
                     # compute needed for a fixed interval width, relative to full runs
                     "relative_compute": (variance * mean_cost) / (y.var() * full_cost)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--split", choices=["loao", "temporal"], default="loao")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--tasks", type=int, help="tasks per replicate (default: all)")
    args = parser.parse_args()

    predictions = pd.read_pickle(args.data / f"predictions_{args.split}.pkl")
    rng = np.random.default_rng(0)
    rows = []
    for agent, agent_rows in predictions.groupby("agent", sort=True):
        F, cost = pad_runs(agent_rows)
        rows += [{"agent": agent, **r} for r in evaluate_agent(F, cost, args.replicates, rng, args.tasks)]
        print(f"done {agent}", flush=True)

    results = pd.DataFrame(rows)
    suffix = f"_n{args.tasks}" if args.tasks else ""
    out = Path(__file__).parent / "results" / f"results_{args.split}{suffix}.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)

    summary = results.groupby("method", sort=False).agg(
        coverage=("coverage", "mean"), worst_coverage=("coverage", "min"),
        mean_abs_bias=("bias", lambda b: b.abs().mean()), max_abs_bias=("bias", lambda b: b.abs().max()),
        cost_fraction=("cost_fraction", "mean"),
        median_relative_compute=("relative_compute", "median"),
        worst_relative_compute=("relative_compute", "max"))
    print(summary.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
