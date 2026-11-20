"""Ability-aware step predictor: how the prediction should change with the agent's strength.

    python predictor/train_ability.py              # leave one agent out, gold features

A predictor trained on other agents expects every new agent to be average. Vedanti's
in-evaluation correction (coverage/adaptive.py) fixes the level by shifting every logit by
one number, but a uniform shift also makes the predictor confidently wrong on the tasks
whose outcome does not move with the agent's strength, which skews the estimate and costs
coverage at a low reach floor (predictor/README.md, "Agent calibration").

Here the agent's resolve rate, theta, is an input. The model is

    logit f(h, theta) = [logit(theta) - logit(mean training theta)] + trees(h, theta)

so the uniform shift is the starting point (and carries the prediction for agents stronger
or weaker than any training agent), and the trees learn where the shift should be smaller
or larger. For a training run, theta is the agent's rate on the other half of its tasks,
so it never contains the run's own outcome. During an evaluation, theta is the running
estimate of the agent's score from earlier batches, which keeps the estimate unbiased
(coverage/THEORY.md, Theorem 2b).

Outputs, for each held-out agent, predictions on a grid of theta values:
    data/predictor/ability_grid_<features>.pkl   {agent: (rows, grid, preds[rows, grid])}
    data/predictor/predictions_ability_<features>_oracle.pkl   theta = the agent's true rate
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from train_lgbm import GROUPS, MAX_ROUNDS, N_VALID_AGENTS, PARAMS, add_task_rate, crossfit_task_rate

GRID = np.linspace(0.02, 0.98, 49)
KEYS = ["agent", "task_id", "k", "next_cost", "outcome"]


def crossfit_ability(rows: pd.DataFrame, outcomes: pd.DataFrame, seed: int) -> pd.Series:
    """Each training run gets its agent's resolve rate on the other half of the agent's tasks."""
    tasks = np.array(sorted(outcomes.index))
    half = pd.Series(np.random.default_rng(seed + 1).permutation(len(tasks)) % 2, index=tasks)
    theta = pd.Series(np.nan, index=rows.index)
    row_half = rows["task_id"].map(half).to_numpy()
    for h in (0, 1):
        rate_other = outcomes.loc[half.index[half != h]].mean()  # per agent, on the other half
        mask = row_half == h
        theta[mask] = rows.loc[mask, "agent"].map(rate_other).to_numpy()
    return theta.clip(0.01, 0.99)


def offset(theta, mean_theta: float) -> np.ndarray:
    return logit(np.clip(theta, 0.01, 0.99)) - logit(mean_theta)


def fit(train: pd.DataFrame, cols: list[str], mean_theta: float, seed: int) -> lgb.Booster:
    agents = sorted(train["agent"].unique())
    valid_agents = set(np.random.default_rng(seed).choice(agents, size=N_VALID_AGENTS, replace=False))
    is_valid = train["agent"].isin(valid_agents).to_numpy()
    init = offset(train["theta"].to_numpy(), mean_theta)
    params = {**PARAMS, "seed": seed,
              "monotone_constraints": [1 if c == "theta" else 0 for c in cols]}
    dtrain = lgb.Dataset(train.loc[~is_valid, cols], train.loc[~is_valid, "outcome"], init_score=init[~is_valid])
    dvalid = lgb.Dataset(train.loc[is_valid, cols], train.loc[is_valid, "outcome"], init_score=init[is_valid])
    return lgb.train(params, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def predict(model: lgb.Booster, test: pd.DataFrame, cols: list[str], theta: float,
            mean_theta: float) -> np.ndarray:
    x = test[[c for c in cols if c != "theta"]].copy()
    x["theta"] = theta
    raw = model.predict(x[cols], num_iteration=model.best_iteration, raw_score=True)
    return expit(raw + offset(theta, mean_theta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--out", type=Path, default=Path("data/predictor"))
    parser.add_argument("--features", choices=list(GROUPS), default="gold")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features = pd.read_pickle(args.data / "features_rich.pkl")
    outcomes = features[features["k"] == 0].pivot(index="task_id", columns="agent", values="outcome")
    outcomes = outcomes.dropna()
    features = features[features["task_id"].isin(outcomes.index)]
    agents = sorted(outcomes.columns)
    cols = GROUPS[args.features] + ["theta"]

    grids, oracle = {}, []
    for agent in agents:
        others = [a for a in agents if a != agent]
        train = features[features["agent"].isin(others)].copy()
        train["task_rate"] = crossfit_task_rate(train, outcomes, others, args.seed)
        train["theta"] = crossfit_ability(train, outcomes[others], args.seed)
        mean_theta = float(outcomes[others].mean().mean())
        model = fit(train, cols, mean_theta, args.seed)

        test = features[features["agent"] == agent].copy()
        test["task_rate"] = add_task_rate(test, outcomes, others, leave_own_out=False)
        preds = np.column_stack([predict(model, test, cols, t, mean_theta) for t in GRID])
        grids[agent] = (test[KEYS].reset_index(drop=True), GRID, preds.astype(np.float32))
        true_rate = float(outcomes[agent].mean())
        part = test[KEYS].copy()
        part["f"] = predict(model, test, cols, true_rate, mean_theta)
        oracle.append(part)
        first = part[part["k"] == 0]
        print(f"{agent}: true {true_rate:.3f}, step-0 mean at true theta {first['f'].mean():.3f}, "
              f"step-0 Brier {((first['f'] - first['outcome']) ** 2).mean():.3f}, "
              f"{model.best_iteration} trees", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"ability_grid_{args.features}.pkl", "wb") as f:
        pickle.dump(grids, f)
    out = args.out / f"predictions_ability_{args.features}_oracle.pkl"
    pd.concat(oracle, ignore_index=True).to_pickle(out)
    print(f"wrote {out} and the theta grid")


if __name__ == "__main__":
    main()
