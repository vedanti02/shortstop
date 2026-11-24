"""Held-out check on Rishi's tool A/B runs: Qwen3-Coder-30B-A3B, 4 arms x 100 SWE-bench Verified tasks.

    python -I predictor/tool_ab_check.py                 # models in data/predictor/runtime/all
    python -I predictor/tool_ab_check.py --ensemble 3    # the seed-averaged models (all_ens3)

The runs (data/tool_ab/qwen-<arm>.jsonl, written by tool_ab/convert.py) come from a model that is
not one of the 37 training agents, so the runtime models trained on all of them are held out for
it. Predictions are made as runtime.py makes them live (RunFeatures, one step at a time), batched
per model, and checked against Runtime.predict on a few runs, with and without a shift.

Calibration moves all of an arm's predictions by one logit shift, as Runtime.calibrate does:
    as trained          no shift
    cross-fit           the tasks are split into two random halves, and each half gets the shift
                        that the arm's runs on the other half give, so no run's own outcome is
                        used: what a harness can do with earlier full runs of the agent
    oracle              each arm's shift from its own outcomes on all tasks: a bound, not a method
    cross-fit baseline  for the comparisons: every arm gets the baseline arm's cross-fit shift,
                        so both sides of a comparison keep the same step-0 predictions

Four sections:
    1 signal    Brier score and AUC of the prediction at fixed fractions of each run
                (evaluate.py), as trained and cross-fit.
    2 designs   One arm at a time, as for the 37 agents in predictor/README.md: tracking and the
                learned-remaining-cost policy, independent and balanced stopping, through
                balanced.py's simulate() (tasks resampled), with kappa as in the runtime; as
                trained, cross-fit and oracle.
    3 controls  Where the cost comes from: the 37 training agents (held-out predictions) on the
                same 100 tasks, all of them and those solving under 45% of the tasks; Vedanti's
                in-batch correction (coverage/adaptive.py: tracking kappa 0.25, floor 0.05,
                batches of 25, damping 100) on each arm; and, on the 37 agents, a step-0
                prediction that does not know the task, which is all a predictor given only the
                steps so far can make (the plug-in as written).
    4 ab        Each tool change against the baseline, paired by task, under the standard design
                (mean +/- 1.96 sd / sqrt(n) of the per-task differences), with each task's
                stopping uniform shared by both arms or drawn separately. On the tasks both arms
                ran (all 100, or 99 when one run is an infrastructure error): how often the
                full-run verdict is reached, and the tokens spent. With tasks resampled
                (batches as large, and of 500 for the calibrated rows): the compute for
                the same precision, and the coverage. Larger kappas are also tried as trained;
                those are tuned on these runs, so optimistic.

Output: predictor/results/tool_ab_{signal,designs,controls,ab}[_ens<N>].csv
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive
import stopping
from balanced import Z, reach_for, simulate
from cost_aware import pad
from evaluate import signal_table
from experiments import pad_runs
from runtime import Runtime, as_dict, logit_shift, shift_probabilities

from shortstop import load_trajectories

ARMS = ["baseline", "descriptions", "search", "compressed"]
RESULTS = Path(__file__).parent / "results"
KAPPA_SCALES = [1.0, 1.5, 2.0, 3.0, 5.0]
CALIBRATIONS = ["as trained", "cross-fit", "oracle", "cross-fit baseline"]
SHOWN = {"tracking, independent": ("tracking", "independent", "naive"),
         "standard design": ("cost-aware", "independent", "naive"),
         "balanced design": ("cost-aware", "stratified B=50", "design + fpc")}
WEAK = 0.45  # training agents that solve fewer of the 100 tasks than this count as weak


def predict_runs(rt: Runtime, runs: dict) -> pd.DataFrame:
    """Rows k = 0..T-1 per run: the prediction after k steps and the cost of step k + 1."""
    rows = []
    for (task, agent), steps in sorted(runs.items()):
        state = rt.tracker(task)
        for step in steps:
            d = as_dict(step)
            rows.append({**state.row(), "agent": agent, "task_id": task,
                         "next_cost": d["tokens_in"] + d["tokens_out"], "outcome": d["outcome"]})
            state.update(d)
    x = pd.DataFrame(rows)
    task = x["task_id"].str.split("@").str[0]
    x["task_rate"] = task.map(rt.meta["task_rate"]).fillna(rt.meta["default_task_rate"])
    x["task_cost"] = task.map(rt.meta["task_cost"]).fillna(rt.meta["default_task_cost"])
    f = np.mean([m.predict(x[rt.meta["success_features"]]) for m in rt.success_models], axis=0)
    x["f"] = shift_probabilities(f)  # kept inside [1e-4, 1 - 1e-4], as Runtime.predict
    x["r_hat"] = np.maximum(np.expm1(rt.cost_model.predict(x[rt.meta["cost_features"]])), 1.0)
    return x[["agent", "task_id", "k", "f", "r_hat", "next_cost", "outcome"]]


def largest_gap_to_runtime(rt: Runtime, runs: dict, pred: pd.DataFrame, shift: float = 0.0,
                           n_runs: int = 3) -> float:
    """Largest difference between the batched predictions, moved by `shift`, and those of a
    Runtime with that shift, step by step."""
    live = Runtime(rt.success_models, rt.cost_model, rt.meta, rt.gold, shift)
    worst = 0.0
    for (task, agent), steps in sorted(runs.items())[:n_runs]:
        mine = pred[(pred["task_id"] == task) & (pred["agent"] == agent)].sort_values("k")
        state = live.tracker(task)
        for step, (f, r) in zip(steps, mine[["f", "r_hat"]].to_numpy()):
            f_live, r_live = live.predict(task, state)
            worst = max(worst, abs(shift_probabilities(f, shift) - f_live), abs(r - r_live) / r_live)
            state.update(as_dict(step))
    return worst


def geo(s) -> float:
    return float(np.exp(np.log(s).mean()))


def select(d: pd.DataFrame, policy: str, design: str, interval: str) -> pd.DataFrame:
    return d[(d["policy"] == policy) & (d["design"] == design) & (d["interval"] == interval)]


def verdict(mean: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.where(mean + half < 0, "worse", np.where(mean - half > 0, "better", "no difference"))


def run_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F, cost, r_rel) for one agent's rows, tasks in sorted order: what simulate() takes."""
    F, cost = pad_runs(g)
    r_hat = pad(g, "r_hat", 1.0)
    return F, cost, r_hat / r_hat[:, 0].mean()  # remaining cost relative to the batch at step 0


def arm_arrays(pred: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return run_arrays(pred[pred["agent"] == f"qwen-{arm}"])


def pair_arrays(pred: pd.DataFrame, arm: str) -> tuple[tuple, tuple]:
    """Baseline and arm arrays on the tasks both arms ran, in the same order. A task missing from
    one arm (an infrastructure error, listed in its errors.json) is left out of that comparison."""
    base, other = pred[pred["agent"] == "qwen-baseline"], pred[pred["agent"] == f"qwen-{arm}"]
    both = set(base["task_id"]) & set(other["task_id"])
    return run_arrays(base[base["task_id"].isin(both)]), run_arrays(other[other["task_id"].isin(both)])


def first_half(pred: pd.DataFrame) -> np.ndarray:
    """True for the rows of tasks in the first of two random halves, the same split in every arm."""
    tasks = np.array(sorted(pred["task_id"].unique()))
    half = set(np.random.default_rng(0).permutation(tasks)[: len(tasks) // 2])
    return pred["task_id"].isin(half).to_numpy()


def shift_on(pred: pd.DataFrame, rows: np.ndarray) -> float:
    """The logit shift that brings the mean step-0 prediction of these rows to their solve rate."""
    s = pred[rows & (pred["k"].to_numpy() == 0)]
    return logit_shift(s["f"], s["outcome"])


def calibrate(pred: pd.DataFrame, how: str) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Predictions after a logit shift per arm (see the module docstring), and the shifts used."""
    if how == "as trained":
        return pred, {}
    first, agent = first_half(pred), pred["agent"].to_numpy()
    out, shifts = pred.copy(), {}
    for a in np.unique(agent):
        rows = agent == a
        ref = agent == "qwen-baseline" if how == "cross-fit baseline" else rows
        if how == "oracle":
            parts = [(rows, shift_on(pred, ref))]
        else:  # each half of the tasks gets what the reference arm's runs on the other half give
            parts = [(rows & first, shift_on(pred, ref & ~first)), (rows & ~first, shift_on(pred, ref & first))]
        for m, d in parts:
            out.loc[m, "f"] = shift_probabilities(pred.loc[m, "f"].to_numpy(), d)
        shifts[a] = [d for _, d in parts]
    return out, shifts


def compare(base, arm, kappa: float, shared: bool, draws: int, replicates: int, seed: int,
            tasks: int | None = None) -> dict:
    """One tool change against the baseline under early stopping, standard design. On the same
    tasks only the stopping is random; then tasks are resampled with replacement, `tasks` per
    batch (default: as many as there are), for the compute and the coverage."""
    (Fb, cb, rb), (Fa, ca, ra) = base, arm
    Rb, Ra = reach_for(Fb, cb, rb, "cost-aware", kappa), reach_for(Fa, ca, ra, "cost-aware", kappa)
    tb, ta = stopping.estimate_table(Fb, Rb), stopping.estimate_table(Fa, Ra)
    n = len(Fb)
    m = tasks or n
    zero = np.zeros((n, 1))
    pb, pa = np.concatenate([zero, np.cumsum(cb, 1)], 1), np.concatenate([zero, np.cumsum(ca, 1)], 1)
    d_full = Fa[:, -1] - Fb[:, -1]
    full_mean, full_half = d_full.mean(), Z * d_full.std(ddof=1) / np.sqrt(n)
    var_full, cost_full = d_full.var(), (cb.sum() + ca.sum()) / n  # per task, both arms
    rng = np.random.default_rng(seed)

    def run(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ub = rng.random(idx.shape)
        ua = ub if shared else rng.random(idx.shape)
        sb, sa = stopping.sample_stop_steps(Rb[idx], ub), stopping.sample_stop_steps(Ra[idx], ua)
        d = ta[idx, sa] - tb[idx, sb]
        return d.mean(1), Z * d.std(1, ddof=1) / np.sqrt(idx.shape[1]), (pa[idx, sa] + pb[idx, sb]).mean(1)

    # the same tasks, only the stopping is random
    fixed = [run(np.tile(np.arange(n), (500, 1))) for _ in range(draws // 500)]
    mean, half, spent = (np.concatenate(v) for v in zip(*fixed))
    # tasks resampled with replacement, as in balanced.simulate
    boot = [run(rng.integers(0, n, (250, m))) for _ in range(replicates // 250)]
    b_mean, b_half, b_spent = (np.concatenate(v) for v in zip(*boot))
    full_verdict = verdict(np.array([full_mean]), np.array([full_half]))[0]
    return {
        "kappa": kappa, "uniforms": "shared" if shared else "separate", "tasks": m,
        "full_difference": full_mean, "full_half_width": full_half, "full_verdict": full_verdict,
        "same_verdict": float((verdict(mean, half) == full_verdict).mean()),
        "coverage_same_tasks": float((np.abs(mean - full_mean) <= half).mean()),
        "half_width_vs_full": float(half.mean() / full_half),
        "tokens_spent": float(spent.mean() / cost_full),
        "relative_compute_actual": float(b_mean.var() * b_spent.mean() / (var_full / m * cost_full)),
        "relative_compute_reported": float((b_half.mean() / Z) ** 2 * m * b_spent.mean() / (var_full * cost_full)),
        "coverage_resampled": float((np.abs(b_mean - full_mean) <= b_half).mean()),
    }


def cell(compute: float, coverage: float) -> str:
    return f"{compute:.0%} ({coverage:.0%})".rjust(14)


def signal(pred: pd.DataFrame, calibration: str) -> pd.DataFrame:
    step0 = pred[pred["k"] == 0].groupby("agent")
    t = signal_table(pred)
    t.insert(1, "calibration", calibration)
    t.insert(2, "solve_rate", step0["outcome"].mean().reindex(t["agent"]).to_numpy())
    t.insert(3, "mean_f0", step0["f"].mean().reindex(t["agent"]).to_numpy())
    return t


def section_signal(variants: dict, tag: str) -> None:
    sig = pd.concat([signal(variants[c][0], c) for c in ["as trained", "cross-fit"]], ignore_index=True)
    sig.to_csv(RESULTS / f"tool_ab_signal{tag}.csv", index=False)
    ref = pd.read_csv(RESULTS / f"lgbm_gold_loao{tag}_signal.csv")
    cols = ["brier_0.0", "brier_0.25", "brier_0.5", "brier_0.75", "auc_0.0", "auc_0.5"]
    print("\n1. Prediction quality at 0, 25, 50 and 75% of each run")
    print("   37 training agents, each held out: " + ", ".join(f"{c} {ref[c].mean():.3f}" for c in cols))
    print(sig[["agent", "calibration", "solve_rate", "mean_f0", *cols]].round(3).to_string(index=False))


def shift_note(shifts: dict) -> str:
    return ", ".join(f"{a.removeprefix('qwen-')} " + " / ".join(f"{d:+.2f}" for d in s) for a, s in sorted(shifts.items()))


def section_designs(variants: dict, kappas: dict, replicates: int, pool, tag: str) -> None:
    shown = ["as trained", "cross-fit", "oracle"]
    jobs = [(f"qwen-{arm}", *arm_arrays(variants[c][0], arm), kappas, replicates, [10, 25, 50], i)
            for c in shown for i, arm in enumerate(ARMS)]
    labels = [c for c in shown for _ in ARMS]
    des = pd.DataFrame([{"calibration": c, **row} for c, rows in zip(labels, pool.map(simulate, jobs)) for row in rows])
    des.to_csv(RESULTS / f"tool_ab_designs{tag}.csv", index=False)
    ref = pd.read_csv(RESULTS / "balanced.csv")
    print(f"\n2. One arm at a time ({replicates} resampled replicates; kappa {kappas['tracking']:.3f} tracking, "
          f"{kappas['cost-aware']:.3f} learned cost): compute for the reported width (coverage)")
    print(f"   {'':30s}" + "".join(f"{a:>14s}" for a in ARMS) + "   37 agents, 500 tasks")
    for c in shown:
        print(f"   {c}" + (f" (shifts: {shift_note(variants[c][1])})" if variants[c][1] else ""))
        for label, key in SHOWN.items():
            q, r = select(des[des["calibration"] == c], *key).set_index("agent"), select(ref, *key)
            cells = "".join(cell(q.loc[f"qwen-{a}", "relative_compute_reported"], q.loc[f"qwen-{a}", "coverage"])
                            for a in ARMS)
            print(f"     {label:28s}{cells}   {geo(r['relative_compute_reported']):.1%} ({r['coverage'].mean():.0%})")


def section_controls(rt: Runtime, pred: pd.DataFrame, variants: dict, kappas: dict, replicates: int,
                     pool, tag: str) -> None:
    inputs = pd.read_pickle("data/predictor/cost_aware_inputs.pkl")  # held-out, written by cost_aware.py
    parts = []
    print(f"\n3. Controls ({replicates} resampled replicates)")

    # a. the 37 training agents on the same 100 tasks
    same = inputs[inputs["task_id"].isin(set(pred["task_id"]))]
    jobs = [(a, *run_arrays(g), kappas, replicates, [50], i) for i, (a, g) in enumerate(same.groupby("agent", sort=True))]
    small = pd.DataFrame([row for rows in pool.map(simulate, jobs) for row in rows])
    rates = same[same["k"] == 0].groupby("agent")["outcome"].mean()
    small["solve_rate"] = small["agent"].map(rates)
    parts.append(small.assign(control="37 agents, the same 100 tasks"))
    weak = rates[rates < WEAK]
    names = ", ".join(f"{a.split('_')[-1]} {r:.0%}" for a, r in weak.items())
    ref = pd.read_csv(RESULTS / "balanced.csv")
    print(f"   a. The 37 training agents on the same 100 tasks; weak: under {WEAK:.0%} of them ({names})")
    for label, key in SHOWN.items():
        s, f = select(small, *key), select(ref, *key)
        w = s[s["agent"].isin(weak.index)]
        print(f"      {label:24s} all {geo(s['relative_compute_reported']):.1%} ({s['coverage'].mean():.1%}), "
              f"weak {geo(w['relative_compute_reported']):.0%} ({w['coverage'].mean():.1%})   | 500 tasks "
              f"{geo(f['relative_compute_reported']):.1%} ({f['coverage'].mean():.1%})")

    # b. Vedanti's in-batch correction, as in coverage/adaptive.py's settings for LightGBM
    jobs = [(*arm_arrays(pred, arm)[:2], replicates, 25, corrected, np.random.default_rng(i), 0.25, 0.05, 100)
            for i, arm in enumerate(ARMS) for corrected in [False, True]]
    out = pool.starmap(adaptive.simulate, jobs)
    oracle = variants["oracle"][1]
    rows = [{"control": "in-batch correction", "agent": f"qwen-{arm}", "method": method,
             "oracle_shift": oracle[f"qwen-{arm}"][0], **out[2 * i + j]}
            for i, arm in enumerate(ARMS) for j, method in enumerate(["static", "corrected"])]
    parts.append(pd.DataFrame(rows))
    print("   b. Vedanti's in-batch correction (tracking kappa 0.25, floor 0.05, batches of 25, damping 100)")
    for i, arm in enumerate(ARMS):
        s, c = rows[2 * i], rows[2 * i + 1]
        print(f"      {arm:12s} static {s['relative_compute']:.0%} ({s['coverage']:.0%}) -> corrected "
              f"{c['relative_compute']:.0%} ({c['coverage']:.0%}); final shift {c['final_delta']:+.2f}, "
              f"oracle {c['oracle_shift']:+.2f}")

    # c. a step-0 prediction that does not know the task, on the 37 agents and all their tasks
    f_blind, r_blind = rt.predict("", rt.tracker(""))  # no task: average task features, no steps
    blind = inputs.copy()
    k0 = blind["k"] == 0
    blind.loc[k0, "f"], blind.loc[k0, "r_hat"] = f_blind, r_blind
    labels = ["task-aware step 0", "task-blind step 0"]
    jobs = [(a, *run_arrays(g), kappas, replicates, [50], i)
            for d in [inputs, blind] for i, (a, g) in enumerate(d.groupby("agent", sort=True))]
    res = pool.map(simulate, jobs)
    k = len(res) // 2
    sims = {label: pd.DataFrame([row for rows in part for row in rows]) for label, part in zip(labels, [res[:k], res[k:]])}
    parts += [s.assign(control=label) for label, s in sims.items()]
    print(f"   c. Step-0 prediction without the task (every task gets f0 {f_blind:.2f} and "
          f"{r_blind:,.0f} tokens to go), the 37 agents on all their tasks")
    for label, key in list(SHOWN.items())[1:]:
        a, b = (select(sims[x], *key) for x in labels)
        print(f"      {label:24s} task known {geo(a['relative_compute_reported']):.1%} ({a['coverage'].mean():.1%}) "
              f"-> task-blind {geo(b['relative_compute_reported']):.1%} ({b['coverage'].mean():.1%})")
    pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"tool_ab_controls{tag}.csv", index=False)


def section_ab(variants: dict, kappa: float, draws: int, replicates: int, pool, tag: str) -> None:
    labels, jobs = [], []
    for j, arm in enumerate(ARMS[1:]):
        for c in ["as trained", "cross-fit baseline"]:
            pair = pair_arrays(variants[c][0], arm)
            for shared in [True, False]:
                for tasks in [None] if c == "as trained" else [None, 500]:
                    labels.append((arm, c))
                    jobs.append((*pair, kappa, shared, draws, replicates, 100 + j, tasks))
        pair = pair_arrays(variants["as trained"][0], arm)
        for scale in KAPPA_SCALES[1:]:
            labels.append((arm, "as trained"))
            jobs.append((*pair, kappa * scale, True, draws, replicates, 100 + j, None))
    ab = pd.DataFrame([{"change": a, "calibration": c, **r} for (a, c), r in zip(labels, pool.starmap(compare, jobs))])
    ab.to_csv(RESULTS / f"tool_ab_ab{tag}.csv", index=False)
    print("\n4. Tool change vs baseline, standard design (the tasks both arms ran: verdict kept, tokens, width vs "
          "full runs; resampled batches of `tasks`: compute for the same precision, coverage)")
    with pd.option_context("display.width", 250):
        show = ab.assign(kappa=ab["kappa"].round(3))[[
            "change", "calibration", "uniforms", "kappa", "tasks", "full_verdict", "same_verdict", "tokens_spent",
            "half_width_vs_full", "coverage_same_tasks", "relative_compute_actual", "relative_compute_reported",
            "coverage_resampled"]]
        print(show.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("data/tool_ab"))
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--control-replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    tag = f"_ens{args.ensemble}" if args.ensemble > 1 else ""
    rt = Runtime.load(Path("data/predictor/runtime") / f"all{tag}")
    if not rt.gold:
        raise SystemExit("these models list no gold patch table in meta.json; the check needs one")
    runs = {}
    for arm in ARMS:
        runs.update(load_trajectories(args.runs / f"qwen-{arm}.jsonl"))
    assert not {a for _, a in runs} & set(rt.meta["training_agents"])
    pred = predict_runs(rt, runs)
    variants = {c: calibrate(pred, c) for c in CALIBRATIONS}
    some_shift = variants["cross-fit"][1]["qwen-baseline"][0]
    gap = max(largest_gap_to_runtime(rt, runs, pred, s) for s in [0.0, some_shift])
    print(f"{len(rt.success_models)} success model(s); {len(runs)} runs, {len(pred)} step prefixes; "
          f"largest gap to Runtime.predict on 3 runs, unshifted and shifted {some_shift:+.2f}: {gap:.1e}")
    p0 = pred[pred["k"] == 0].pivot(index="task_id", columns="agent", values="f")
    print(f"step-0 prediction differs between arms by at most {(p0.max(axis=1) - p0.min(axis=1)).max():.1e} "
          "(it only sees the task)")

    tuned = pd.read_csv(RESULTS / "cost_aware.csv")
    kappas = {"tracking": float(tuned.loc[tuned["rule"] == "tracking", "kappa"].mode().iloc[0]),
              "cost-aware": float(rt.meta["kappa"])}
    with Pool(args.workers) as pool:
        section_signal(variants, tag)
        section_designs(variants, kappas, args.replicates, pool, tag)
        section_controls(rt, pred, variants, kappas, args.control_replicates, pool, tag)
        section_ab(variants, kappas["cost-aware"], args.draws, args.replicates, pool, tag)


if __name__ == "__main__":
    main()
