"""Score a set of step predictions with every stopping policy, on held-out agents.

    python predictor/evaluate.py data/swebench_verified/predictions_loao.pkl --name standin

The input has Vedanti's prediction columns (agent, task_id, k, f, next_cost, outcome), one
row per step prefix. For each agent, every policy is tuned on the other agents and scored
on that agent, exactly as in coverage/optimal_policy.py. Results go to predictor/results/
so nothing under coverage/ is overwritten.

Outputs:
    <name>_policy.csv       one row per agent and policy: relative compute, coverage, cost
    <name>_signal.csv       one row per agent: Brier score and AUC at points through the run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
import optimal_policy as op
import stopping
from experiments import interval_stats, pad_runs, sample_estimates

RESULTS = Path(__file__).parent / "results"
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75]  # how far through the run the prediction is read


def load_runs(predictions: pd.DataFrame):
    runs, lengths = {}, {}
    for agent, rows in predictions.groupby("agent", sort=True):
        runs[agent] = pad_runs(rows)
        lengths[agent] = rows.groupby("task_id", sort=True).size().to_numpy()
    return runs, lengths


def compare_policies(runs: dict, lengths: dict, replicates: int, seed: int = 0) -> pd.DataFrame:
    """Leave-one-agent-out comparison of the five policies in coverage/optimal_policy.py."""
    agents = list(runs)
    sched = {a: op.schedule_stats(*runs[a]) for a in agents}
    state = {a: op.state_stats(*runs[a], lengths[a]) for a in agents}
    T_max = max(runs[a][1].shape[1] for a in agents)
    index = {a: op.index_stats(*runs[a], lengths[a], T_max) for a in agents}
    one_shot = {a: [op.relative_compute(*runs[a], stopping.reach_one_shot(runs[a][0], q))
                    for q in op.QS] for a in agents}
    tracking = {a: [op.relative_compute(*runs[a], stopping.reach_tracking(runs[a][0], k, op.P_MIN))
                    for k in op.KAPPAS] for a in agents}

    rng = np.random.default_rng(seed)
    rows = []
    for held_out in agents:
        train = [a for a in agents if a != held_out]
        F, cost = runs[held_out]

        def tuned(scores: dict, train=train) -> int:
            return int(np.argmin([op.geo_mean([scores[a][i] for a in train])
                                  for i in range(len(scores[train[0]]))]))

        q = op.QS[tuned(one_shot)]
        kappa = op.KAPPAS[tuned(tracking)]
        schedule = op.fit_schedule([sched[a] for a in train])
        table = op.fit_state_table([state[a] for a in train])
        learned = {a: [op.relative_compute(*runs[a], op.reach_learned(runs[a][0], table, k))
                       for k in op.KAPPAS] for a in train}
        learned_kappa = op.KAPPAS[tuned(learned)]
        index_table = op.fit_index_table([index[a] for a in train])
        index_scores = {a: [op.relative_compute(*runs[a], op.reach_index(runs[a][0], index_table, k))
                            for k in op.INDEX_KAPPAS] for a in train}
        index_kappa = op.INDEX_KAPPAS[tuned(index_scores)]

        policies = {
            "one-shot": (stopping.reach_one_shot(F, q), q),
            "tracking": (stopping.reach_tracking(F, kappa, op.P_MIN), kappa),
            "step schedule": (op.reach_schedule(F, schedule), None),
            "learned tracking": (op.reach_learned(F, table, learned_kappa), learned_kappa),
            "index": (op.reach_index(F, index_table, index_kappa), index_kappa),
        }
        idx = rng.integers(0, len(F), (replicates, len(F)))
        for name, (reach, setting) in policies.items():
            _, expected_cost = stopping.exact_moments(F, cost, reach)
            stats = interval_stats(sample_estimates(F, reach, idx, rng), F[:, -1].mean())
            rows.append({"agent": held_out, "policy": name, "setting": setting,
                         "true_rate": F[:, -1].mean(), "coverage": stats["coverage"],
                         "bias": stats["bias"],
                         "cost_fraction": expected_cost.mean() / cost.sum(axis=1).mean(),
                         "relative_compute": op.relative_compute(F, cost, reach)})
    return pd.DataFrame(rows)


def signal_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Brier score and AUC of the prediction read at fixed fractions of each run.

    This is the "does information arrive early" diagnostic: if the prediction a quarter of
    the way in is no better than the step-0 prediction, stopping inside a run cannot beat
    one-shot by much, whatever the policy.
    """
    p = predictions.copy()
    p["T"] = p.groupby(["agent", "task_id"])["k"].transform("max") + 1
    rows = []
    for agent, g in p.groupby("agent", sort=True):
        row = {"agent": agent}
        for frac in FRACTIONS:
            target = np.floor(frac * g["T"]).astype(int)
            pick = g[g["k"] == target]
            y, f = pick["outcome"].to_numpy(), pick["f"].to_numpy()
            row[f"brier_{frac}"] = float(((f - y) ** 2).mean())
            row[f"auc_{frac}"] = float(roc_auc_score(y, f)) if 0 < y.mean() < 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    for frac in FRACTIONS[1:]:
        out[f"skill_{frac}"] = 1 - out[f"brier_{frac}"] / out["brier_0.0"]
    return out


def summarize(policy: pd.DataFrame, signal: pd.DataFrame) -> str:
    s = policy.groupby("policy", sort=False).agg(
        median=("relative_compute", "median"), geo_mean=("relative_compute", op.geo_mean),
        worst=("relative_compute", "max"),
        worse_than_full=("relative_compute", lambda r: int((r > 1).sum())),
        cost_fraction=("cost_fraction", "mean"), coverage=("coverage", "mean"),
        worst_coverage=("coverage", "min"), max_abs_bias=("bias", lambda b: b.abs().max()))
    wide = policy.pivot(index="agent", columns="policy", values="relative_compute")
    per_step = wide[["tracking", "step schedule", "learned tracking", "index"]].min(axis=1)
    gain = wide["one-shot"] - per_step
    lines = [s.round(3).to_string(), "",
             (f"best per-step policy vs one-shot: median gain {gain.median():.3f} "
              f"(positive = stopping inside the run helps), "
              f"better for {(gain > 0).sum()} of {len(gain)} agents"),
             "", "prediction quality through the run (mean over agents):",
             signal.drop(columns="agent").mean().round(3).to_string()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--name", required=True, help="prefix for the output files")
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()

    predictions = pd.read_pickle(args.predictions)
    runs, lengths = load_runs(predictions)
    policy = compare_policies(runs, lengths, args.replicates)
    signal = signal_table(predictions)
    RESULTS.mkdir(exist_ok=True)
    policy.to_csv(RESULTS / f"{args.name}_policy.csv", index=False)
    signal.to_csv(RESULTS / f"{args.name}_signal.csv", index=False)
    print(summarize(policy, signal))
    print(f"wrote {RESULTS / args.name}_*.csv")


if __name__ == "__main__":
    main()
