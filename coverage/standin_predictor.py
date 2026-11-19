"""Stand-in step predictor: logistic regression on step counts and test-run features.

    python coverage/standin_predictor.py                   # leave the evaluated agent out
    python coverage/standin_predictor.py --split temporal  # train only on earlier agents

Each agent is scored by a model that never saw it. One feature, task_rate, is how often the
training agents solved the task, which is known before a new agent is run.
Output: data/swebench_verified/predictions_<split>.pkl with a column f, the predicted
chance of success after k steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOGGED = ["k", "tokens_so_far", "test_runs", "edits", "errors", "nonzero_exits", "tests_passed"]
MAX_TRAIN_ROWS = 300_000
MIN_TRAIN_AGENTS = 5


def design(rows: pd.DataFrame, task_rate: pd.Series) -> np.ndarray:
    cols = [np.log1p(rows[c].to_numpy(dtype=float)) for c in LOGGED]
    cols.append(rows["last_nonzero_exit"].to_numpy(dtype=float))
    cols.append(task_rate.to_numpy(dtype=float))
    return np.column_stack(cols)


def predict_agent(features: pd.DataFrame, outcomes: pd.DataFrame, agent: str,
                  train_agents: list[str], seed: int = 0) -> np.ndarray:
    """Predictions for one held-out agent from a model trained on train_agents."""
    solved = outcomes[train_agents]  # tasks x agents, 0/1
    total, n = solved.sum(axis=1), len(train_agents)

    train = features[features["agent"].isin(train_agents)]
    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(MAX_TRAIN_ROWS, random_state=seed)
    # For a training run, leave its own outcome out of the task's solve rate.
    train_rate = (train["task_id"].map(total) - train["outcome"]) / (n - 1)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(design(train, train_rate), train["outcome"])

    test = features[features["agent"] == agent]
    return model.predict_proba(design(test, test["task_id"].map(total / n)))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--split", choices=["loao", "temporal"], default="loao")
    args = parser.parse_args()

    features = pd.read_pickle(args.data / "features.pkl")
    outcomes = features[features["k"] == 0].pivot(index="task_id", columns="agent",
                                                  values="outcome")
    outcomes = outcomes.dropna()  # keep tasks every agent has
    features = features[features["task_id"].isin(outcomes.index)]
    agents = sorted(outcomes.columns)  # names start with the submission date

    parts = []
    for agent in agents:
        if args.split == "loao":
            train_agents = [a for a in agents if a != agent]
        else:
            train_agents = [a for a in agents if a[:8] < agent[:8]]
        if len(train_agents) < MIN_TRAIN_AGENTS:
            print(f"skip {agent}: only {len(train_agents)} training agents")
            continue
        part = features[features["agent"] == agent].copy()
        part["f"] = predict_agent(features, outcomes, agent, train_agents)
        parts.append(part[["agent", "task_id", "k", "f", "next_cost", "outcome"]])
        first = part[part["k"] == 0]
        brier0 = ((first["f"] - first["outcome"]) ** 2).mean()
        print(f"{agent}: true {first['outcome'].mean():.3f}, mean step-0 prediction "
              f"{first['f'].mean():.3f}, step-0 Brier {brier0:.3f}", flush=True)

    out = args.data / f"predictions_{args.split}.pkl"
    pd.concat(parts, ignore_index=True).to_pickle(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
