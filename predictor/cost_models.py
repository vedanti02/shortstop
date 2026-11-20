"""Which model of remaining cost serves the stopping policy best?

    python predictor/cost_models.py

The policy in cost_aware.py needs the expected tokens still to be spent, E[r | step features].
cost_aware.py regresses log(1 + r) with squared error and exponentiates, which estimates a
geometric mean and so understates runs whose remaining cost is uncertain. Alternatives, each
a LightGBM model per held-out agent trained on the other agents (same features as
cost_aware.py):

    log-L2            log(1 + r), squared error (cost_aware.py's model)
    log-L2 smeared    the same, times the training mean of exp(residual) (Duan's smearing),
                      a standard correction from log scale to the mean
    tweedie           r itself, Tweedie loss (variance power 1.5): estimates E[r] directly
    gamma             1 + r, gamma loss with log link: estimates E[r] directly

Each is scored by the policy it feeds, kappa * sqrt(f (1 - f) / r_hat) with a running minimum
and a 0.05 floor, kappa tuned on the other agents, exactly as in cost_aware.py.

Output: predictor/results/cost_models.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_aware import MAX_TRAIN_ROWS, REG, geo, policies, prepare, task_cost
from train_lgbm import GROUPS, add_task_rate, crossfit_task_rate

OBJECTIVES = {"log-L2": ("regression", "log_remaining"), "tweedie": ("tweedie", "remaining"),
              "gamma": ("gamma", "remaining_plus1")}


def fit(train: pd.DataFrame, cols: list[str], objective: str, target: str, seed: int = 0) -> lgb.Booster:
    agents = sorted(train["agent"].unique())
    valid = set(np.random.default_rng(seed).choice(agents, size=4, replace=False))
    is_valid = train["agent"].isin(valid).to_numpy()
    params = {**REG, "objective": objective, "seed": seed}
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5
    dtr = lgb.Dataset(train.loc[~is_valid, cols], train.loc[~is_valid, target])
    dva = lgb.Dataset(train.loc[is_valid, cols], train.loc[is_valid, target])
    return lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", type=Path, default=Path("data/swebench_verified/features_rich.pkl"))
    parser.add_argument("--predictions", type=Path, default=Path("data/predictor/predictions_gold_loao.pkl"))
    parser.add_argument("--name", default="cost_models")
    args = parser.parse_args()

    d, outcomes, rel_log_cost = prepare(args.features, args.predictions)
    d["remaining_plus1"] = d["remaining"] + 1.0
    cols = GROUPS["gold"] + ["task_cost"]
    agents = sorted(outcomes.columns)
    preds = {name: [] for name in [*OBJECTIVES, "log-L2 smeared"]}
    keep = ["agent", "task_id", "k", "f", "next_cost", "outcome", "remaining"]
    for a in agents:
        others = [b for b in agents if b != a]
        train = d[d["agent"].isin(others)]
        train = train.sample(min(len(train), MAX_TRAIN_ROWS), random_state=0).copy()
        train["task_rate"] = crossfit_task_rate(train, outcomes, others)
        train["task_cost"] = task_cost(train, rel_log_cost, others, crossfit=True)
        test = d[d["agent"] == a].copy()
        test["task_rate"] = add_task_rate(test, outcomes, others, leave_own_out=False)
        test["task_cost"] = task_cost(test, rel_log_cost, others, crossfit=False)
        for name, (objective, target) in OBJECTIVES.items():
            m = fit(train, cols, objective, target)
            raw = m.predict(test[cols], num_iteration=m.best_iteration)
            if name == "log-L2":
                r_hat = np.expm1(raw)
                resid = train[target] - m.predict(train[cols], num_iteration=m.best_iteration)
                smear = float(np.mean(np.exp(resid)))
                preds["log-L2 smeared"].append(test[keep].assign(r_hat=np.clip(np.exp(raw) * smear - 1, 1.0, None),
                                                                 e_hat=1e-4))
            elif name == "gamma":
                r_hat = raw - 1.0
            else:
                r_hat = raw
            preds[name].append(test[keep].assign(r_hat=np.clip(r_hat, 1.0, None), e_hat=1e-4))
        print(f"done {a}", flush=True)

    rows = []
    for name, parts in preds.items():
        inputs = pd.concat(parts, ignore_index=True)
        corr = inputs.groupby("agent").apply(
            lambda g: np.corrcoef(g["remaining"], g["r_hat"])[0, 1], include_groups=False).median()
        res = policies(inputs)
        res = res[res["rule"].isin(["tracking", "+ predicted cost"])].assign(model=name, corr_raw_scale=corr)
        rows.append(res)
    res = pd.concat(rows, ignore_index=True)
    out = Path(__file__).parent / "results" / f"{args.name}.csv"
    res.to_csv(out, index=False)
    s = res[res["rule"] == "+ predicted cost"].groupby("model", sort=False).agg(
        corr=("corr_raw_scale", "first"), median=("relative_compute", "median"),
        geo_mean=("relative_compute", geo), worst=("relative_compute", "max"),
        coverage=("coverage", "mean"), worst_coverage=("coverage", "min"))
    print(s.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
