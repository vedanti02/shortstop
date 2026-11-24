"""Offline estimate of mid-run test probes: run the benchmark's tests right after the decisive edit.

    python predictor/probe_estimate.py --accuracy 0.9

Agents make their last edit to the files the reference patch touches about a quarter of the
way through a run's tokens (median 25 to 27%, predictor/README.md), then spend the rest
testing and polishing. A live evaluation could run the benchmark's hidden tests on the code
at that point, at no LLM-token cost. This estimates what that would be worth on the logged
runs, with coverage/midrun_signal.py's convention: from the step after a run's last
gold-file edit, the prediction is replaced by `accuracy` or 1 - `accuracy` according to a
signal that equals the final outcome with probability `accuracy`. Runs with no detected
gold-file edit get no signal. The accuracy stands in for two things this cannot check
offline: later edits elsewhere changing the result, and edits the command-text detection
misses. Before the signal the prediction is gold LightGBM (held out).

Output: data/predictor/predictions_gold_loao_probe_a<accuracy>.pkl; score it with evaluate.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--accuracy", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    p = pd.read_pickle("data/predictor/predictions_gold_loao.pkl")
    feats = pd.read_pickle("data/swebench_verified/features_rich.pkl")[["agent", "task_id", "k", "gold_edited_frac"]]
    p = p.merge(feats, on=["agent", "task_id", "k"], how="left").sort_values(["agent", "task_id", "k"])
    run = p.groupby(["agent", "task_id"], sort=False)
    final = run["gold_edited_frac"].transform("max")
    reached = (p["gold_edited_frac"] >= final) & (final > 0)
    # first row at which the last gold-file edit has happened: the probe result is known from here
    p["probe_from"] = p["k"].where(reached).groupby([p["agent"], p["task_id"]]).transform("min")

    rng = np.random.default_rng(args.seed)
    runs = p[["agent", "task_id", "outcome"]].drop_duplicates(["agent", "task_id"])
    correct = rng.random(len(runs)) < args.accuracy
    runs["signal"] = np.where(correct, runs["outcome"], 1 - runs["outcome"])
    p = p.merge(runs[["agent", "task_id", "signal"]], on=["agent", "task_id"])
    informed = np.where(p["signal"] == 1, args.accuracy, 1 - args.accuracy)
    use = p["probe_from"].notna() & (p["k"] >= p["probe_from"])
    p["f"] = np.where(use, informed, p["f"])

    with_probe = p.groupby(["agent", "task_id"])["probe_from"].first().notna().mean()
    print(f"runs with a detected gold-file edit (probe available): {with_probe:.1%}; "
          f"rows replaced: {use.mean():.1%}")
    out = Path(f"data/predictor/predictions_gold_loao_probe_a{args.accuracy:g}.pkl")
    p[["agent", "task_id", "k", "next_cost", "outcome", "f"]].to_pickle(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
