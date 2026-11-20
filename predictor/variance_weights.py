"""Row weights that turn the estimator's variance into a weighted training loss.

    python predictor/variance_weights.py data/predictor/predictions_gold_loao.pkl --kappa tuned

For one run, Var(Y_hat | run) = sum_k w_k (Y - f_{k-1})^2 with w_k = 1/P_k - 1/P_{k-1}
(coverage/THEORY.md, Lemma 2). The prediction after j steps (feature row j) is therefore
weighted by w_{j+1}: how much the policy cuts the reach probability right after it. A
predictor trained with these weights spends its accuracy where the policy stops runs.

The reach probabilities come from the tracking policy applied to an existing set of
predictions, with kappa either given or tuned to minimize the geometric-mean relative
compute over all agents.

Output: data/predictor/weights_<input name>.pkl, a Series aligned with features_rich.pkl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
import optimal_policy as op
import stopping
from experiments import pad_runs

FLOOR = 0.01  # every row keeps this share of the mean weight, so no row is ignored outright


def tune_kappa(predictions: pd.DataFrame) -> float:
    runs = [pad_runs(rows) for _, rows in predictions.groupby("agent", sort=True)]
    scores = [op.geo_mean([op.relative_compute(F, c, stopping.reach_tracking(F, k, op.P_MIN))
                           for F, c in runs]) for k in op.KAPPAS]
    return op.KAPPAS[int(np.argmin(scores))]


def tracking_weights(predictions: pd.DataFrame, kappa: float, p_min: float = op.P_MIN) -> pd.Series:
    p = predictions.sort_values(["agent", "task_id", "k"])
    f = p["f"].to_numpy()
    target = np.clip(kappa * np.sqrt(f * (1 - f)), p_min, 1.0)  # target for P_{k+1}
    runs = [p["agent"], p["task_id"]]
    reach = pd.Series(target, index=p.index).groupby(runs).cummin()
    u = 1.0 / reach
    u_prev = u.groupby(runs).shift(1).fillna(1.0)
    return (u - u_prev).clip(lower=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--kappa", default="tuned")
    parser.add_argument("--features", type=Path, default=Path("data/swebench_verified/features_rich.pkl"))
    args = parser.parse_args()

    predictions = pd.read_pickle(args.predictions)
    kappa = tune_kappa(predictions) if args.kappa == "tuned" else float(args.kappa)
    w = tracking_weights(predictions, kappa)
    w = w + FLOOR * w.mean()

    keys = predictions.loc[w.index, ["agent", "task_id", "k"]].assign(w=w.to_numpy())
    rows = pd.read_pickle(args.features)[["agent", "task_id", "k"]].reset_index()
    merged = rows.merge(keys, on=["agent", "task_id", "k"], how="left")
    weights = pd.Series(merged["w"].to_numpy(), index=merged["index"]).fillna(FLOOR * w.mean())

    by_k = keys.assign(step=np.minimum(keys["k"], 10)).groupby("step")["w"].sum()
    print(f"kappa {kappa}; share of total weight by step (10 = 10 or later):")
    print((by_k / by_k.sum()).round(3).to_string())
    out = args.predictions.parent / f"weights_{args.predictions.stem}.pkl"
    weights.to_pickle(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
