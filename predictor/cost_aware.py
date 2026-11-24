"""Learned inputs for the stopping policy: remaining cost and expected error, trained per held-out agent.

    python predictor/cost_aware.py                      # SWE-bench Verified, gold LightGBM
    python predictor/cost_aware.py --features data/rebench_lb/features_rich.pkl \
        --predictions data/predictor/rebench_lb/predictions_gold_loao.pkl --name cost_aware_rebench

By Theorem 1 (coverage/THEORY.md) the optimal reach probability grows with the expected squared
error of the current prediction and shrinks with the expected cost still to come. Vedanti's
tracking rule uses f (1 - f) for the first and ignores the second; learned tracking estimates
both on a coarse state (step bin, prediction decile). Run costs vary a lot (coefficient of
variation 1.3 across a single agent's tasks), and a rule told each run's true remaining cost
cuts compute from 52.6% to 45.1% (geometric mean over held-out agents).

Two LightGBM regressions, trained on the other agents only, supply those inputs from the
features available at each step:

    remaining cost   log(1 + tokens still to be spent in the run)
    expected error   (Y - f)^2 of the held-out LightGBM prediction f

plus, as a feature, how expensive the task was for the other agents (cross-fitted for the
training rows). The predicted remaining cost is divided by its mean at step 0 over the
agent's runs, so only relative costs matter (in a live evaluation, the mean over tasks seen).

Policies (running minimum, reach floor 0.05, kappa tuned on the other agents):
    tracking                    kappa * sqrt(f (1 - f))
    + learned error             kappa * sqrt(e_hat)
    + predicted cost            kappa * sqrt(f (1 - f) / r_hat)
    + learned error and cost    kappa * sqrt(e_hat / r_hat)
    + true remaining cost       kappa * sqrt(f (1 - f) / r)                (bound)

Output: predictor/results/cost_aware.csv and data/predictor/cost_aware_inputs.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimal_policy as op
from experiments import interval_stats, sample_estimates
from train_lgbm import GROUPS, PARAMS, add_task_rate, crossfit_task_rate

FLOOR = 0.05
KAPPAS = np.geomspace(0.01, 3.0, 40)
MAX_TRAIN_ROWS = 400_000
REG = {**{k: v for k, v in PARAMS.items() if k != "objective"}, "objective": "regression"}


def geo(x) -> float:
    return float(np.exp(np.mean(np.log(x))))


def prepare(features: Path, predictions: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = pd.read_pickle(features)
    pred = pd.read_pickle(predictions)[["agent", "task_id", "k", "f"]]
    d = feats.merge(pred, on=["agent", "task_id", "k"]).sort_values(["agent", "task_id", "k"])
    d["task"] = d["task_id"].str.split("@").str[0]  # repeated runs share a task
    run = d.groupby(["agent", "task_id"], sort=False)["next_cost"]
    d["remaining"] = run.transform(lambda s: s[::-1].cumsum()[::-1])  # cost of steps k+1..T
    d["log_remaining"] = np.log1p(d["remaining"])
    d["sq_err"] = (d["outcome"] - d["f"]) ** 2
    d = d.reset_index(drop=True)
    outcomes = d[d["k"] == 0].pivot_table(index="task", columns="agent", values="outcome", aggfunc="mean")
    run_cost = d.groupby(["agent", "task_id", "task"])["next_cost"].sum().clip(lower=1)
    rel_log_cost = np.log(run_cost).groupby(["task", "agent"]).mean().unstack("agent")
    rel_log_cost = rel_log_cost - rel_log_cost.mean()  # each agent's tasks relative to its average
    return d, outcomes, rel_log_cost


def task_cost(rows: pd.DataFrame, rel_log_cost: pd.DataFrame, agents: list[str], crossfit: bool,
              seed: int = 0) -> pd.Series:
    """How expensive the task was for other agents (relative log cost); cross-fitted for training."""
    if not crossfit:
        return rows["task"].map(rel_log_cost[agents].mean(axis=1))
    perm = np.random.default_rng(seed).permutation(agents)
    halves = [list(perm[::2]), list(perm[1::2])]
    out = pd.Series(np.nan, index=rows.index)
    for mine, other in [(halves[0], halves[1]), (halves[1], halves[0])]:
        m = rows["agent"].isin(mine)
        out[m] = rows.loc[m, "task"].map(rel_log_cost[other].mean(axis=1))
    return out


def fit_reg(train: pd.DataFrame, cols: list[str], target: str, seed: int = 0) -> lgb.Booster:
    agents = sorted(train["agent"].unique())
    valid = set(np.random.default_rng(seed).choice(agents, size=4, replace=False))
    is_valid = train["agent"].isin(valid).to_numpy()
    dtr = lgb.Dataset(train.loc[~is_valid, cols], train.loc[~is_valid, target])
    dva = lgb.Dataset(train.loc[is_valid, cols], train.loc[is_valid, target])
    return lgb.train({**REG, "seed": seed}, dtr, 2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def learn_inputs(d, outcomes, rel_log_cost) -> pd.DataFrame:
    agents = sorted(outcomes.columns)
    cost_cols = GROUPS["gold"] + ["task_cost"]
    err_cols = GROUPS["gold"] + ["task_cost", "f"]
    parts = []
    for a in agents:
        others = [b for b in agents if b != a]
        train = d[d["agent"].isin(others)]
        if len(train) > MAX_TRAIN_ROWS:
            train = train.sample(MAX_TRAIN_ROWS, random_state=0)
        train = train.copy()
        train["task_rate"] = crossfit_task_rate(train, outcomes, others)
        train["task_cost"] = task_cost(train, rel_log_cost, others, crossfit=True)
        test = d[d["agent"] == a].copy()
        test["task_rate"] = add_task_rate(test, outcomes, others, leave_own_out=False)
        test["task_cost"] = task_cost(test, rel_log_cost, others, crossfit=False)
        m_cost = fit_reg(train, cost_cols, "log_remaining")
        m_err = fit_reg(train, err_cols, "sq_err")
        test["r_hat"] = np.expm1(m_cost.predict(test[cost_cols], num_iteration=m_cost.best_iteration)).clip(min=1.0)
        test["e_hat"] = m_err.predict(test[err_cols], num_iteration=m_err.best_iteration).clip(1e-4, 1.0)
        parts.append(test[["agent", "task_id", "k", "f", "next_cost", "outcome", "remaining", "r_hat", "e_hat"]])
        print(f"{a}: remaining-cost corr {np.corrcoef(np.log1p(test['remaining']), np.log(test['r_hat']))[0, 1]:.2f}, "
              f"trees {m_cost.best_iteration}/{m_err.best_iteration}", flush=True)
    return pd.concat(parts, ignore_index=True)


def pad(g: pd.DataFrame, col: str, fill: float) -> np.ndarray:
    runs = [x.sort_values("k")[col].to_numpy() for _, x in g.groupby("task_id", sort=True)]
    T = max(len(r) for r in runs)
    out = np.full((len(runs), T), fill)
    for i, r in enumerate(runs):
        out[i, : len(r)] = r
    return out


def policies(inputs: pd.DataFrame) -> pd.DataFrame:
    from experiments import pad_runs
    data = {}
    for a, g in inputs.groupby("agent", sort=True):
        F, cost = pad_runs(g)
        r_hat = pad(g, "r_hat", 1.0)
        rem_true = np.cumsum(cost[:, ::-1], axis=1)[:, ::-1]
        scale_hat = r_hat[:, 0].mean()
        data[a] = {"F": F, "cost": cost, "e_hat": pad(g, "e_hat", 1e-4),
                   "r_hat": r_hat / scale_hat, "r_true": rem_true / cost.sum(axis=1).mean()}

    def reach(a, rule, k):
        x = data[a]
        f = np.clip(x["F"][:, :-1], 1e-4, 1 - 1e-4)
        u2 = f * (1 - f)
        t = {"tracking": u2, "+ learned error": x["e_hat"], "+ predicted cost": u2 / x["r_hat"],
             "+ learned error and cost": x["e_hat"] / x["r_hat"],
             "+ true remaining cost": u2 / np.maximum(x["r_true"], 1e-6)}[rule]
        t = np.where(x["cost"] > 0, k * np.sqrt(t), 1.0)
        return np.minimum.accumulate(np.clip(t, FLOOR, 1.0), axis=1)

    rules = ["tracking", "+ learned error", "+ predicted cost", "+ learned error and cost", "+ true remaining cost"]
    agents = sorted(data)
    table = {r: {a: [op.relative_compute(data[a]["F"], data[a]["cost"], reach(a, r, k)) for k in KAPPAS]
                 for a in agents} for r in rules}
    rng = np.random.default_rng(0)
    rows = []
    for a in agents:
        train = [b for b in agents if b != a]
        F = data[a]["F"]
        idx = rng.integers(0, len(F), (500, len(F)))
        for r in rules:
            i = int(np.argmin([geo([table[r][b][j] for b in train]) for j in range(len(KAPPAS))]))
            rch = reach(a, r, KAPPAS[i])
            stats = interval_stats(sample_estimates(F, rch, idx, rng), F[:, -1].mean())
            rows.append({"agent": a, "rule": r, "kappa": KAPPAS[i], "relative_compute": table[r][a][i],
                         "coverage": stats["coverage"], "bias": stats["bias"]})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", type=Path, default=Path("data/swebench_verified/features_rich.pkl"))
    parser.add_argument("--predictions", type=Path, default=Path("data/predictor/predictions_gold_loao.pkl"))
    parser.add_argument("--name", default="cost_aware")
    args = parser.parse_args()

    d, outcomes, rel_log_cost = prepare(args.features, args.predictions)
    inputs = learn_inputs(d, outcomes, rel_log_cost)
    inputs.to_pickle(f"data/predictor/{args.name}_inputs.pkl")
    res = policies(inputs)
    out = Path(__file__).parent / "results" / f"{args.name}.csv"
    res.to_csv(out, index=False)
    ref = res[res["rule"] == "tracking"].set_index("agent")["relative_compute"]
    s = res.groupby("rule", sort=False).agg(median=("relative_compute", "median"), geo_mean=("relative_compute", geo),
                                            worst=("relative_compute", "max"), coverage=("coverage", "mean"),
                                            worst_coverage=("coverage", "min"))
    s["vs tracking"] = [geo(res[res["rule"] == r].set_index("agent")["relative_compute"] / ref) for r in s.index]
    print(s.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
