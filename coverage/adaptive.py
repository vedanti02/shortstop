"""Correct the predictor during an evaluation, using the agent's own results so far.

    python coverage/adaptive.py

A predictor trained on other agents expects a new agent to be average, which makes the
estimator expensive for agents that are far from average. Here tasks are evaluated in
batches. Before each batch, every prediction is shifted on the logit scale by one number,
delta, chosen so that the shifted step-0 predictions on the tasks seen so far average to
the current unbiased estimate of the agent's score.

Delta for a batch depends only on earlier batches, so each task's estimate is still
unbiased given everything before it, and the running mean is unbiased. The interval is the
usual mean +/- 1.96 sd / sqrt(n); whether it still covers is checked here by simulation.

Output: coverage/results/adaptive_<split>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import stopping
from experiments import Z, pad_runs

KAPPA, P_MIN = 0.25, 0.05


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def solve_delta(logit0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per row, the shift that makes mean(sigmoid(logit0 + delta)) equal target."""
    lo, hi = np.full(len(target), -8.0), np.full(len(target), 8.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        too_high = sigmoid(logit0 + mid[:, None]).mean(axis=1) > target
        hi, lo = np.where(too_high, mid, hi), np.where(too_high, lo, mid)
    return (lo + hi) / 2


def simulate(F, cost, replicates: int, batch: int, adapt: bool, rng,
             kappa: float = KAPPA, p_min: float = P_MIN) -> dict:
    n, T = cost.shape
    y = F[:, -1]
    logits = np.log(np.clip(F[:, :-1], 1e-4, 1 - 1e-4) / (1 - np.clip(F[:, :-1], 1e-4, 1 - 1e-4)))
    paid_table = np.concatenate([np.zeros((n, 1)), np.cumsum(cost, axis=1)], axis=1)

    idx = rng.integers(0, n, (replicates, n))  # tasks in the order they are evaluated
    estimates = np.empty((replicates, n))
    paid = np.empty((replicates, n))
    delta = np.zeros(replicates)
    for start in range(0, n, batch):
        tasks = idx[:, start:start + batch]
        shifted = sigmoid(logits[tasks] + delta[:, None, None])
        Fb = np.concatenate([shifted, y[tasks][..., None]], axis=2).reshape(-1, T + 1)
        reach = stopping.reach_tracking(Fb, kappa, p_min)
        steps = stopping.sample_stop_steps(reach, rng.random(len(Fb)))
        rows = np.arange(len(Fb))
        estimates[:, start:start + batch] = stopping.estimate_table(Fb, reach)[rows, steps].reshape(tasks.shape)
        paid[:, start:start + batch] = paid_table[tasks.ravel(), steps].reshape(tasks.shape)
        if adapt:
            seen = idx[:, :start + batch]
            target = np.clip(estimates[:, :start + batch].mean(axis=1), 0.01, 0.99)
            delta = solve_delta(logits[seen, 0], target)

    mean = estimates.mean(axis=1)
    half = Z * estimates.std(axis=1, ddof=1) / np.sqrt(n)
    full_variance, full_cost = y.var() / n, cost.sum(axis=1).mean()
    return {"coverage": float((np.abs(mean - y.mean()) <= half).mean()),
            "bias": float(mean.mean() - y.mean()),
            "cost_fraction": float(paid.mean() / full_cost),
            "relative_compute": float(mean.var() * paid.mean() / (full_variance * full_cost)),
            "final_delta": float(np.median(delta))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--split", choices=["loao", "temporal"], default="loao")
    parser.add_argument("--replicates", type=int, default=800)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--floor", type=float, default=P_MIN, help="lowest reach probability")
    args = parser.parse_args()

    predictions = pd.read_pickle(args.data / f"predictions_{args.split}.pkl")
    rng = np.random.default_rng(0)
    rows = []
    for agent, agent_rows in predictions.groupby("agent", sort=True):
        F, cost = pad_runs(agent_rows)
        for name, adapt in [("static", False), ("adaptive", True)]:
            stats = simulate(F, cost, args.replicates, args.batch, adapt, rng, args.kappa, args.floor)
            rows.append({"agent": agent, "method": name, "true_rate": F[:, -1].mean(), **stats})
        print(f"done {agent}: static {rows[-2]['relative_compute']:.2f}, "
              f"adaptive {rows[-1]['relative_compute']:.2f}", flush=True)

    results = pd.DataFrame(rows)
    suffix = "" if (args.kappa, args.floor) == (KAPPA, P_MIN) else f"_k{args.kappa}_f{args.floor}"
    out = Path(__file__).parent / "results" / f"adaptive_{args.split}{suffix}.csv"
    results.to_csv(out, index=False)
    summary = results.groupby("method", sort=False).agg(
        median=("relative_compute", "median"), worst=("relative_compute", "max"),
        worse_than_full=("relative_compute", lambda r: int((r > 1).sum())),
        cost_fraction=("cost_fraction", "mean"), coverage=("coverage", "mean"),
        worst_coverage=("coverage", "min"), max_abs_bias=("bias", lambda b: b.abs().max()))
    print(summary.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
