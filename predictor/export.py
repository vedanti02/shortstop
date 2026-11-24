"""Export the predictor for the rest of the project.

    python predictor/export.py coverage --split loao        # predictions for coverage/*.py
    python predictor/export.py runtime                      # models for a live harness
    python predictor/export.py runtime --exclude <agent>    # the same without one agent (demo)
    python predictor/export.py runtime --seeds 3            # 3 success models, averaged -> all_ens3/

coverage
    Copies data/predictor/predictions_gold_<split>.pkl (made by train_lgbm.py, gold features,
    the scored agent never seen) to data/predictor/for_coverage/predictions_<split>.pkl, the file
    name Vedanti's scripts read with --data. Columns are hers: agent, task_id, k, f, next_cost,
    outcome. Run, for example:
        python coverage/experiments.py --data data/predictor/for_coverage --split loao
    (her scripts write to coverage/results/, so this overwrites the stand-in's results there).

runtime
    Trains the two models a harness needs on the SWE-bench Verified agents (all, or all but one):
        success   gold LightGBM: P(success | steps so far)
        cost      LightGBM on log(1 + tokens still to be spent)
    Both use cross-fitted task features in training (predictor/README.md), the full tables at
    run time. Saved to data/predictor/runtime/<name>/ with meta.json: feature lists, the task
    solve-rate and task-cost tables, and the policy settings (kappa tuned on held-out agents in
    cost_aware.py, the 0.05 floor, strata of 50). Load with predictor/runtime.py.

    --seeds N trains N success models (seeds 0 to N-1, each with its own cross-fit halves) and
    the runtime averages their predictions; they go to <name>_ens<N>/, next to the single-seed
    models, and load the same way. The remaining-cost model stays a single seed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_aware import fit_reg, task_cost
from train_lgbm import GROUPS, crossfit_task_rate, fit

FEATURES = Path("data/swebench_verified/features_rich.pkl")
FLOOR, BLOCK = 0.05, 50


def export_coverage(split: str) -> None:
    src = Path(f"data/predictor/predictions_gold_{split}.pkl")
    if not src.exists():
        raise SystemExit(f"{src} is missing: run python predictor/train_lgbm.py --features gold --split {split}")
    dst = Path("data/predictor/for_coverage") / f"predictions_{split}.pkl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"wrote {dst}; run: python coverage/experiments.py --data {dst.parent} --split {split}")


def export_runtime(exclude: str | None, seeds: int = 1, seed: int = 0) -> None:
    d = pd.read_pickle(FEATURES).sort_values(["agent", "task_id", "k"]).reset_index(drop=True)
    d["task"] = d["task_id"]
    outcomes = d[d["k"] == 0].pivot(index="task_id", columns="agent", values="outcome").dropna()
    agents = [a for a in sorted(outcomes.columns) if a != exclude]
    d = d[d["agent"].isin(agents) & d["task_id"].isin(outcomes.index)].copy()
    run = d.groupby(["agent", "task_id"], sort=False)["next_cost"]
    d["log_remaining"] = np.log1p(run.transform(lambda s: s[::-1].cumsum()[::-1]))
    run_cost = d.groupby(["agent", "task_id"])["next_cost"].sum().unstack("agent").clip(lower=1)
    rel_log_cost = np.log(run_cost)
    rel_log_cost = rel_log_cost - rel_log_cost.mean()

    d["task_rate"] = crossfit_task_rate(d, outcomes, agents, seed)
    d["task_cost"] = task_cost(d, rel_log_cost, agents, crossfit=True, seed=seed)
    success_cols = GROUPS["gold"]
    cost_cols = GROUPS["gold"] + ["task_cost"]
    train = d.sample(min(len(d), 400_000), random_state=seed)
    cost = fit_reg(train, cost_cols, "log_remaining", seed)
    successes = []
    for s in range(seed, seed + seeds):  # each seed with its own cross-fit halves, as in train_lgbm.py
        d["task_rate"] = crossfit_task_rate(d, outcomes, agents, s)
        successes.append(fit(d, success_cols, None, s))

    tuned = pd.read_csv(Path(__file__).parent / "results" / "cost_aware.csv")
    tuned = tuned[tuned["rule"] == "+ predicted cost"]
    kappa = float(tuned.loc[tuned["agent"] == exclude, "kappa"].iloc[0]) if exclude else \
        float(tuned["kappa"].mode().iloc[0])  # tuned without the excluded agent / the usual value

    name = ("all" if exclude is None else f"without_{exclude}") + (f"_ens{seeds}" if seeds > 1 else "")
    out = Path("data/predictor/runtime") / name
    out.mkdir(parents=True, exist_ok=True)
    files = ["success.txt" if i == 0 else f"success_seed{seed + i}.txt" for i in range(seeds)]
    for file, model in zip(files, successes):
        model.save_model(str(out / file), num_iteration=model.best_iteration)
    cost.save_model(str(out / "cost.txt"), num_iteration=cost.best_iteration)
    task_rate = outcomes[agents].mean(axis=1)
    meta = {"success_features": success_cols, "cost_features": cost_cols, "success_models": files,
            "task_rate": task_rate.round(6).to_dict(), "default_task_rate": float(task_rate.mean()),
            "task_cost": rel_log_cost[agents].mean(axis=1).round(6).to_dict(), "default_task_cost": 0.0,
            "kappa": kappa, "floor": FLOOR, "block": BLOCK, "training_agents": agents,
            "gold_meta": ["data/meta/swebench_verified.parquet"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {len(agents)} training agents, kappa {kappa}, "
          f"trees {'/'.join(str(m.best_iteration) for m in successes)} (success) / "
          f"{cost.best_iteration} (cost)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["coverage", "runtime"])
    parser.add_argument("--split", default="loao", help="coverage: loao, lofo or temporal")
    parser.add_argument("--exclude", help="runtime: an agent to leave out of training")
    parser.add_argument("--seeds", type=int, default=1, help="runtime: success models to train and average")
    args = parser.parse_args()
    if args.what == "coverage":
        export_coverage(args.split)
    else:
        export_runtime(args.exclude, args.seeds)


if __name__ == "__main__":
    main()
