"""LightGBM step predictor, scored on agents it never saw.

    python predictor/train_lgbm.py --features base            # Vedanti's inputs, LightGBM model
    python predictor/train_lgbm.py --features process         # + how the run is going
    python predictor/train_lgbm.py --features gold            # + reference-solution checks
    python predictor/train_lgbm.py --features gold --split insample   # optimistic bound

Splits: loao trains on every other agent, lofo leaves out the agent's whole vendor, temporal
trains only on agents submitted earlier (agents with fewer than 5 earlier ones are skipped),
insample trains on all agents including the scored one (only to measure how much is lost
to never having seen the agent). Like the stand-in, task_rate is how often the training
agents solved the task. For training rows it is cross-fitted (measured on the other half of
the training agents) so it never depends on the row's own outcome; see crossfit_task_rate.

An optional --weights file gives one weight per (agent, task_id, k) row, used to train a
predictor for a particular stopping policy (see variance_weights.py).

--seeds 0,1,2 trains one model per seed and averages their predictions: less training noise
(see "Does more training data help?" in predictor/README.md). Use it with --tag, e.g. --tag ens3.

Output: data/predictor/predictions_<features>_<split>[_<tag>].pkl with Vedanti's columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
from standin_predictor import vendor

BASE = ["k", "tokens_so_far", "test_runs", "edits", "errors", "nonzero_exits", "tests_passed",
        "last_nonzero_exit", "task_rate"]
PROCESS = ["context_tokens", "mean_out_tokens", "format_errors", "timeouts", "repro_created",
           "repro_runs", "repeats", "consecutive_repeat", "git_diffs", "last_test_status",
           "n_passed_last", "n_failed_last", "steps_since_test", "steps_since_edit",
           "edits_since_test", "edited_files_n", "obs_chars_last"]
GOLD = ["gold_viewed_frac", "gold_found_frac", "gold_edited_frac", "nongold_edited_n",
        "diff_gold_frac", "diff_nongold_n", "f2p_file_runs", "f2p_name_mentions",
        "f2p_last_status", "gold_lines", "gold_files_n", "f2p_n"]
GROUPS = {"base": BASE, "process": BASE + PROCESS, "gold": BASE + PROCESS + GOLD}

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 12}
MAX_ROUNDS = 2000
N_VALID_AGENTS = 4
MIN_TEMPORAL_AGENTS = 5  # as in coverage/standin_predictor.py


def task_of(rows: pd.DataFrame) -> pd.Series:
    """The task a row belongs to: task_id without the "@<run>" suffix of repeated runs."""
    return rows["task"] if "task" in rows else rows["task_id"]


def add_task_rate(rows: pd.DataFrame, outcomes: pd.DataFrame, train_agents: list[str],
                  leave_own_out: bool) -> pd.Series:
    """Leave-one-out solve rate. Leaks the row's own outcome once the model can identify the
    task (rate is lower exactly when the run passed), so training uses crossfit_task_rate.
    With repeated runs, outcomes holds each agent's mean over its runs of the task."""
    solved = outcomes[train_agents]
    total, n = solved.sum(axis=1), len(train_agents)
    if leave_own_out:
        return (task_of(rows).map(total) - rows["outcome"]) / (n - 1)
    return task_of(rows).map(total / n)


def crossfit_task_rate(rows: pd.DataFrame, outcomes: pd.DataFrame, train_agents: list[str],
                       seed: int = 0) -> pd.Series:
    """Solve rate for training rows that never depends on the row's own outcome.

    Training agents are split in two halves; rows of one half get the rate measured on the
    other half. With a leave-one-out rate, task-level columns (patch size, number of tests)
    let the model identify the task and read the run's own outcome off the rate: on the
    37-agent split this raised the step-0 Brier from 0.129 to 0.152.
    """
    perm = np.random.default_rng(seed).permutation(train_agents)
    halves = [list(perm[::2]), list(perm[1::2])]
    rate = pd.Series(np.nan, index=rows.index)
    for mine, other in [(halves[0], halves[1]), (halves[1], halves[0])]:
        is_mine = rows["agent"].isin(mine)
        rate[is_mine] = task_of(rows.loc[is_mine]).map(outcomes[other].mean(axis=1))
    return rate


def fit(train: pd.DataFrame, cols: list[str], weight: np.ndarray | None, seed: int,
        never_valid: tuple[str, ...] = ()):
    """Fit with early stopping on a few held-back training agents, then return the model."""
    agents = sorted(set(train["agent"].unique()) - set(never_valid))
    rng = np.random.default_rng(seed)
    valid_agents = set(rng.choice(agents, size=min(N_VALID_AGENTS, len(agents) // 4), replace=False))
    is_valid = train["agent"].isin(valid_agents).to_numpy()
    w = np.ones(len(train)) if weight is None else weight
    dtrain = lgb.Dataset(train.loc[~is_valid, cols], train.loc[~is_valid, "outcome"], weight=w[~is_valid])
    dvalid = lgb.Dataset(train.loc[is_valid, cols], train.loc[is_valid, "outcome"], weight=w[is_valid])
    model = lgb.train({**PARAMS, "seed": seed}, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def predict_agent(features, outcomes, agent, train_agents, cols, weights, seed=0) -> np.ndarray:
    train = features[features["agent"].isin(train_agents)].copy()
    train["task_rate"] = crossfit_task_rate(train, outcomes, train_agents, seed)
    w = None if weights is None else weights.loc[train.index].to_numpy()
    model = fit(train, cols, w, seed)
    if agent in train_agents:  # insample bound: score the rows exactly as they were trained
        test = train[train["agent"] == agent]
    else:
        test = features[features["agent"] == agent].copy()
        test["task_rate"] = add_task_rate(test, outcomes, train_agents, leave_own_out=False)
    return model.predict(test[cols], num_iteration=model.best_iteration), model.best_iteration


def predict_agent_seen(features, outcomes, agent, others, cols, seed=0, with_id=False):
    """The agent is seen on half of its tasks and scored on the other half (two folds).

    Measures what knowing the agent is worth without letting the model memorize the runs it
    is scored on. task_rate never uses the agent's own outcomes. with_id adds the agent's
    name as a categorical feature, the most direct way to know the agent.
    """
    tasks = np.array(sorted(features.loc[features["agent"] == agent, "task_id"].unique()))
    fold = pd.Series(np.random.default_rng(seed).permutation(len(tasks)) % 2, index=tasks)
    rest = features[features["agent"].isin(others)].copy()
    rest["task_rate"] = crossfit_task_rate(rest, outcomes, others, seed)
    mine = features[features["agent"] == agent].copy()
    mine["task_rate"] = add_task_rate(mine, outcomes, others, leave_own_out=False)
    use = cols + (["agent_id"] if with_id else [])
    categories = sorted([*others, agent])
    f = np.full(len(mine), np.nan)
    in_b_all = mine["task_id"].map(fold).to_numpy()
    for b in (0, 1):
        train = pd.concat([rest, mine[in_b_all != b]])
        test = mine[in_b_all == b].copy()
        if with_id:
            train["agent_id"] = pd.Categorical(train["agent"], categories=categories)
            test["agent_id"] = pd.Categorical(test["agent"], categories=categories)
        model = fit(train, use, None, seed, never_valid=(agent,))
        f[in_b_all == b] = model.predict(test[use], num_iteration=model.best_iteration)
    return f, model.best_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--out", type=Path, default=Path("data/predictor"))
    parser.add_argument("--features", choices=list(GROUPS), default="gold")
    parser.add_argument("--split", choices=["loao", "lofo", "temporal", "insample", "agentseen",
                                            "agentseen_id"], default="loao")
    parser.add_argument("--weights", type=Path, help="pickled Series of row weights")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", help="several seeds, comma-separated (e.g. 0,1,2): predictions are averaged")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    features = pd.read_pickle(args.data / "features_rich.pkl")
    features["task"] = features["task_id"].str.split("@").str[0]  # repeated runs share a task
    outcomes = features[features["k"] == 0].pivot_table(index="task", columns="agent",
                                                        values="outcome", aggfunc="mean")
    outcomes = outcomes.dropna()
    features = features[features["task"].isin(outcomes.index)]
    weights = pd.read_pickle(args.weights).reindex(features.index) if args.weights else None
    agents = sorted(outcomes.columns)
    cols = GROUPS[args.features]

    parts = []
    for agent in agents:
        if args.split == "loao":
            train_agents = [a for a in agents if a != agent]
        elif args.split == "lofo":
            train_agents = [a for a in agents if vendor(a) != vendor(agent)]
        elif args.split == "temporal":  # only agents submitted earlier (names start with the date)
            train_agents = [a for a in agents if a[:8] < agent[:8]]
            if len(train_agents) < MIN_TEMPORAL_AGENTS:
                print(f"skip {agent}: only {len(train_agents)} earlier agents")
                continue
        else:
            train_agents = agents
        part = features.loc[features["agent"] == agent, ["agent", "task_id", "k", "next_cost", "outcome"]].copy()
        if args.split.startswith("agentseen"):
            others = [a for a in agents if a != agent]
            runs = [predict_agent_seen(features, outcomes, agent, others, cols, s,
                                       with_id=args.split == "agentseen_id") for s in seeds]
        else:
            runs = [predict_agent(features, outcomes, agent, train_agents, cols, weights, s) for s in seeds]
        part["f"] = np.mean([f for f, _ in runs], axis=0)
        rounds = "/".join(str(r) for _, r in runs)
        parts.append(part)
        first = part[part["k"] == 0]
        print(f"{agent}: true {first['outcome'].mean():.3f}, step-0 mean {first['f'].mean():.3f}, "
              f"step-0 Brier {((first['f'] - first['outcome']) ** 2).mean():.3f}, {rounds} trees",
              flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"predictions_{args.features}_{args.split}" + (f"_{args.tag}" if args.tag else "")
    out = args.out / f"{name}.pkl"
    pd.concat(parts, ignore_index=True).to_pickle(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
