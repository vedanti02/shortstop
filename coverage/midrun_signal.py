"""Controlled check: does the index policy gain when the predictor learns mid-run?

    python coverage/midrun_signal.py

The stand-in predictor learns little until a run ends, and there the index policy and
learned tracking tie. Here the same logged runs and costs are kept, but the predictor is
given a signal partway through each run: from step ceil(when * length) on, it sees the
true outcome with probability `accuracy` (the wrong one otherwise) and predicts
accuracy or 1 - accuracy accordingly. Before that step it is the stand-in predictor.

Policies are fitted on the other agents and scored on the held-out one, as in
optimal_policy.py. Output: coverage/results/midrun_signal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optimal_policy as op
import pandas as pd
import stopping
from experiments import pad_runs

SETTINGS = [(None, None), (0.5, 0.9), (0.25, 0.9), (0.1, 0.9), (0.25, 0.75)]  # (when, accuracy)


def add_signal(F, lengths, when: float, accuracy: float, rng) -> np.ndarray:
    """Predictions with a noisy look at the outcome from step ceil(when * length) on."""
    y = F[:, -1]
    signal = np.where(rng.random(len(F)) < accuracy, y, 1 - y)
    informed = np.where(signal == 1, accuracy, 1 - accuracy)
    start = np.ceil(when * lengths).astype(int)
    steps = np.arange(F.shape[1])[None, :]
    replace = (steps >= start[:, None]) & (steps < lengths[:, None])
    return np.where(replace, informed[:, None], F)


def compare(runs: dict, lengths: dict) -> list[dict]:
    agents = list(runs)
    T_max = max(runs[a][1].shape[1] for a in agents)
    state = {a: op.state_stats(*runs[a], lengths[a]) for a in agents}
    index = {a: op.index_stats(*runs[a], lengths[a], T_max) for a in agents}
    one_shot = {a: [op.relative_compute(*runs[a], stopping.reach_one_shot(runs[a][0], q))
                    for q in op.QS] for a in agents}
    rows = []
    for held_out in agents:
        train = [a for a in agents if a != held_out]
        F, cost = runs[held_out]

        def best(score, settings, train=train):
            """Setting with the best geometric-mean score over the training agents."""
            return settings[int(np.argmin([op.geo_mean([score(a, x) for a in train])
                                           for x in settings]))]

        q = best(lambda a, q: one_shot[a][op.QS.index(q)], op.QS)
        table = op.fit_state_table([state[a] for a in train])
        index_table = op.fit_index_table([index[a] for a in train])
        k_learned = best(lambda a, k, table=table: op.relative_compute(
            *runs[a], op.reach_learned(runs[a][0], table, k)), op.INDEX_KAPPAS)
        k_index = best(lambda a, k, table=index_table: op.relative_compute(
            *runs[a], op.reach_index(runs[a][0], table, k)), op.INDEX_KAPPAS)
        for name, reach in [("one-shot", stopping.reach_one_shot(F, q)),
                            ("learned tracking", op.reach_learned(F, table, k_learned)),
                            ("index", op.reach_index(F, index_table, k_index))]:
            rows.append({"agent": held_out, "policy": name,
                         "relative_compute": op.relative_compute(F, cost, reach)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    args = parser.parse_args()

    predictions = pd.read_pickle(args.data / "predictions_loao.pkl")
    base, lengths = {}, {}
    for agent, rows in predictions.groupby("agent", sort=True):
        base[agent] = pad_runs(rows)
        lengths[agent] = rows.groupby("task_id", sort=True).size().to_numpy()

    results = []
    for when, accuracy in SETTINGS:
        rng = np.random.default_rng(0)
        runs = base if when is None else {
            a: (add_signal(F, lengths[a], when, accuracy, rng), cost) for a, (F, cost) in base.items()}
        label = "none" if when is None else f"at {when:g} of the run, accuracy {accuracy:g}"
        rows = compare(runs, lengths)
        results += [{"signal": label, **r} for r in rows]
        frame = pd.DataFrame(rows).pivot(index="agent", columns="policy", values="relative_compute")
        wins = int((frame["index"] < frame["learned tracking"]).sum())
        print(f"signal {label}: " + ", ".join(
            f"{p} median {frame[p].median():.3f} geo {op.geo_mean(frame[p]):.3f}"
            for p in ["one-shot", "learned tracking", "index"])
            + f"; index better for {wins} of {len(frame)} agents", flush=True)

    out = Path(__file__).parent / "results" / "midrun_signal.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
