"""Vedanti's in-evaluation correction (coverage/adaptive.py) run on any set of predictions.

    python predictor/adaptive_eval.py data/predictor/predictions_gold_loao.pkl --name gold \
        --floor 0.2 --prior 50

Calls coverage/adaptive.simulate unchanged: tasks are evaluated in batches, the tracking
policy decides when to stop, and with "adaptive" every prediction is shifted on the logit
scale before each batch so the step-0 predictions on tasks seen so far match the running
unbiased estimate of the agent's score. "static" never shifts. Agents run in parallel, each
with its own random seed, so numbers differ slightly from coverage/adaptive.py, which
shares one generator across agents.

Output: predictor/results/adaptive_<name>_f<floor>_m<prior>.csv, one row per agent and method.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
import adaptive
from experiments import pad_runs

RESULTS = Path(__file__).parent / "results"


def geo_mean(values) -> float:
    return float(np.exp(np.mean(np.log(values))))


def run_agent(job: tuple) -> list[dict]:
    agent, F, cost, methods, cfg, seed = job
    rows = []
    for name in methods:
        rng = np.random.default_rng(seed)  # same task order for every method of this agent
        stats = adaptive.simulate(F, cost, cfg["replicates"], cfg["batch"], name == "adaptive", rng,
                                  cfg["kappa"], cfg["floor"], cfg["prior"])
        rows.append({"agent": agent, "method": name, "true_rate": F[:, -1].mean(), **stats})
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("method", sort=False).agg(
        median=("relative_compute", "median"), geo_mean=("relative_compute", geo_mean),
        worst=("relative_compute", "max"),
        worse_than_full=("relative_compute", lambda r: int((r > 1).sum())),
        cost_fraction=("cost_fraction", "mean"), coverage=("coverage", "mean"),
        worst_coverage=("coverage", "min"), max_abs_bias=("bias", lambda b: b.abs().max()),
        variance_ratio=("variance_ratio", "mean"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--methods", default="static,adaptive")
    parser.add_argument("--replicates", type=int, default=800)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--kappa", type=float, default=adaptive.KAPPA)
    parser.add_argument("--floor", type=float, default=adaptive.P_MIN)
    parser.add_argument("--prior", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    cfg = {k: getattr(args, k) for k in ["replicates", "batch", "kappa", "floor", "prior"]}
    methods = args.methods.split(",")
    predictions = pd.read_pickle(args.predictions)
    jobs = [(agent, *pad_runs(rows), methods, cfg, i)
            for i, (agent, rows) in enumerate(predictions.groupby("agent", sort=True))]
    with Pool(args.workers) as pool:
        results = pd.DataFrame([r for rows in pool.imap(run_agent, jobs) for r in rows])

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"adaptive_{args.name}_f{args.floor:g}_m{args.prior:g}.csv"
    results.to_csv(out, index=False)
    print(summarize(results).round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
