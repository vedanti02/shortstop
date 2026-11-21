"""Does adding another dataset's runs to training help on held-out Verified agents?

    python predictor/extra_data_check.py --source openhands --rows 250000 --tag oh250k
    python predictor/extra_data_check.py --source rebench --rows 250000 --tag rb250k
    python predictor/evaluate.py data/predictor/predictions_gold_loao_oh250k.pkl --name lgbm_gold_loao_oh250k
    python predictor/compare.py lgbm_gold_loao_oh250k --ref lgbm_gold_loao

The same procedure as `train_lgbm.py --features gold --split loao`: the seed, the validation
agents and the cross-fitted task_rate are identical. The one change is that each held-out
agent's model also trains on a random sample of whole runs from another dataset:

    openhands   Nebius and CoderForge: 316,536 runs, all from one model. Their tasks have no
                other agents, so task_rate (how often other agents solved the task) is left
                missing; LightGBM learns which way missing values go.
    rebench     The SWE-rebench leaderboard: 18,468 runs of 36 configurations on 110 tasks.
                task_rate is cross-fitted across its configurations, exactly as for Verified,
                using all of its runs.

- The sample is drawn once (seed 0), a whole run at a time, until it has --rows step rows. A
  smaller sample is the start of a larger one.
- The added agents are never used for early stopping, so the validation agents are the same
  Verified agents as in the baseline.

Output: data/predictor/predictions_gold_loao_<tag>.pkl, in the format of train_lgbm.py. The
sample is cached as train_sample_<rows>.pkl in the dataset's data folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgbm import GROUPS, add_task_rate, crossfit_task_rate, fit

SOURCES = {"openhands": [Path("data/openhands/features_rich.pkl"), Path("data/openhands/coderforge_features.pkl")],
           "rebench": [Path("data/rebench_lb/features_rich.pkl")]}


def with_task_rate(d: pd.DataFrame, source: str, seed: int) -> pd.DataFrame:
    """task_rate for every row of a full dataset: missing for openhands, cross-fitted for rebench."""
    if source == "openhands":
        d["task_rate"] = np.nan
        return d
    d["task"] = d["task_id"].str.split("@").str[0]  # 5 runs per task share it
    outcomes = d[d["k"] == 0].pivot_table(index="task", columns="agent", values="outcome", aggfunc="mean")
    d["task_rate"] = crossfit_task_rate(d, outcomes, sorted(outcomes.columns), seed)
    return d


def extra_sample(source: str, rows: int, seed: int = 0) -> pd.DataFrame:
    """Whole runs, in a fixed random order, until the sample has `rows` step rows."""
    cache = SOURCES[source][0].parent / f"train_sample_{rows}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = [with_task_rate(pd.read_pickle(p), source, seed) for p in SOURCES[source]]
    runs = [d.groupby(["agent", "task_id"], sort=True).ngroup().to_numpy() for d in frames]
    sizes = [np.bincount(r) for r in runs]
    offsets = np.cumsum([0] + [len(s) for s in sizes[:-1]])
    all_sizes = np.concatenate(sizes)
    order = np.random.default_rng(seed).permutation(len(all_sizes))
    n_runs = int(np.searchsorted(np.cumsum(all_sizes[order]), rows)) + 1
    chosen = np.zeros(len(all_sizes), dtype=bool)
    chosen[order[:n_runs]] = True
    sample = pd.concat([d[chosen[off + r]] for d, r, off in zip(frames, runs, offsets)],  # per file:
                       ignore_index=True)                                              # no full copy
    sample.to_pickle(cache)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=list(SOURCES), default="openhands")
    parser.add_argument("--rows", type=int, required=True, help="step rows to add")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features = pd.read_pickle("data/swebench_verified/features_rich.pkl")
    features["task"] = features["task_id"].str.split("@").str[0]
    outcomes = features[features["k"] == 0].pivot_table(index="task", columns="agent",
                                                        values="outcome", aggfunc="mean").dropna()
    features = features[features["task"].isin(outcomes.index)]
    agents = sorted(outcomes.columns)
    cols = GROUPS["gold"]
    keep = ["agent", "outcome", *cols]

    sample = extra_sample(args.source, args.rows)
    n_runs = sample.groupby(["agent", "task_id"]).ngroups
    extra = sample[keep]
    extra_agents = tuple(sorted(extra["agent"].unique()))
    print(f"{args.source} sample: {n_runs:,} runs, {len(extra):,} rows, {len(extra_agents)} agents, "
          f"task_rate missing for {extra['task_rate'].isna().mean():.0%}", flush=True)

    parts = []
    for agent in agents:
        train_agents = [a for a in agents if a != agent]
        train = features[features["agent"].isin(train_agents)].copy()
        train["task_rate"] = crossfit_task_rate(train, outcomes, train_agents, args.seed)
        model = fit(pd.concat([train[keep], extra], ignore_index=True), cols, None, args.seed,
                    never_valid=extra_agents)
        test = features[features["agent"] == agent].copy()
        test["task_rate"] = add_task_rate(test, outcomes, train_agents, leave_own_out=False)
        part = test[["agent", "task_id", "k", "next_cost", "outcome"]].copy()
        part["f"] = model.predict(test[cols], num_iteration=model.best_iteration)
        parts.append(part)
        first = part[part["k"] == 0]
        print(f"{agent}: true {first['outcome'].mean():.3f}, step-0 mean {first['f'].mean():.3f}, "
              f"step-0 Brier {((first['f'] - first['outcome']) ** 2).mean():.3f}, "
              f"{model.best_iteration} trees", flush=True)

    out = Path("data/predictor") / f"predictions_gold_loao_{args.tag}.pkl"
    pd.concat(parts, ignore_index=True).to_pickle(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
