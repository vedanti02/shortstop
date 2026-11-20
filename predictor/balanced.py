"""Balanced stopping decisions: stratified stopping uniforms, with intervals that use them.

    python predictor/balanced.py

Each run stops at s = max{k : U < P_k} with one uniform U. Unbiasedness (coverage/THEORY.md,
Lemma 1) only needs each run's U to be uniform and independent of that run; the uniforms of
different runs may depend on each other. Here they are spread evenly within strata: runs are
sorted by their step-1 reach probability P_1 (ties by f0), cut into strata of B runs, and the
runs of a stratum get U = (position + V) / B for a random permutation of positions and
independent V ~ U(0, 1) (a Latin hypercube; stratified truncation in the sense of Vihola 2018).
Within a stratum the number of runs started, and reaching each later step, becomes nearly
fixed, so the stratum's average miscalibration largely cancels instead of entering the
variance (the fixed-size effect checked in predictor/README.md, one-shot).

The usual interval sd / sqrt(n) treats tasks as independent; under this design it stays valid
but conservative. Two design-based variance estimators use the stratum totals T_h in sort
order instead:
    paired       sum over pairs of adjacent strata of (T_a - T_b)^2, t with H/2 df
    successive   H / (2 (H - 1)) * sum_h (T_h - T_{h-1})^2, t with H - 1 df

Simulation as in coverage/adaptive.py: each replicate resamples the agent's tasks with
replacement, applies the policy and the design, and builds the intervals. Policies are
tracking and the learned-remaining-cost policy of cost_aware.py, each with the kappa tuned
there on the other agents; predictions are gold LightGBM (held out).

Output: predictor/results/balanced.csv
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stopping
from cost_aware import pad
from experiments import pad_runs

FLOOR = 0.05
Z = 1.96


def reach_for(F, cost, r_rel, policy: str, kappa: float) -> np.ndarray:
    f = np.clip(F[:, :-1], 1e-4, 1 - 1e-4)
    u2 = f * (1 - f) if policy == "tracking" else f * (1 - f) / r_rel
    target = np.where(cost > 0, kappa * np.sqrt(u2), 1.0)
    return np.minimum.accumulate(np.clip(target, FLOOR, 1.0), axis=1)


def stratified_uniforms(p1: np.ndarray, f0: np.ndarray, block: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Uniforms (R, n) spread within strata of `block` runs sorted by (P_1, f0); and stratum ids."""
    R, n = p1.shape
    order = np.lexsort((f0, p1), axis=-1)  # sort by p1, then f0
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(R, 0), axis=-1)
    stratum = rank // block
    # random position within the stratum: rank of a random key among the stratum's runs
    key = stratum * 2.0 + rng.random((R, n))  # sorts by stratum, then randomly
    pos_order = np.argsort(key, axis=-1)
    pos = np.empty_like(pos_order)
    np.put_along_axis(pos, pos_order, np.arange(n)[None, :].repeat(R, 0), axis=-1)
    within = pos - stratum * block
    size = np.minimum(block, n - stratum * block)
    return (within + rng.random((R, n))) / size, stratum


def design_variance(est, f0, p1, started, stratum, n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variance of the mean under the stratified design (without and with the finite-population
    correction 1 - m / B for a group of B runs with m started), and its degrees of freedom.

    mean = (1/n) sum Y_i + (1/n) sum (c_i - d_i), with c_i = Y_hat_i - f0_i and d_i = Y_i - f0_i;
    the two parts are uncorrelated, so Var = Var(Y) / n + design variance of the corrections.
      - Y is binary, so Var(Y) = theta (1 - theta), estimated as theta_hat (1 - theta_hat).
      - Only started runs carry a correction. In each stratum (merged with the next until it
        has 2 started runs) the corrections' total is estimated with-replacement style: the
        started runs' c_i * P (P = expected number started in the stratum) are draws whose
        spread gives the variance. The spread is taken within the stratum, so the stratum's
        average miscalibration does not count, which is where the balanced design gains.
    """
    R = est.shape[0]
    theta = np.clip(est.mean(axis=1), 0, 1)
    var = theta * (1 - theta) / n
    var_fpc = var.copy()
    df = np.zeros(R)
    c = est - f0
    for r in range(R):
        s_id, s_started = stratum[r], started[r]
        H = int(s_id.max()) + 1
        counts = np.bincount(s_id[s_started], minlength=H)
        group, g, acc = np.empty(H, dtype=int), 0, 0
        for h in range(H):  # consecutive strata until each group has >= 2 started runs
            group[h] = g
            acc += counts[h]
            if acc >= 2:
                g, acc = g + 1, 0
        if acc < 2 and g > 0:
            group[group == g] = g - 1
        run_group = group[s_id]
        total, total_fpc = 0.0, 0.0
        for k in range(int(group.max()) + 1):
            in_k = run_group == k
            sel = in_k & s_started
            m = int(sel.sum())
            if m >= 2:
                z = c[r, sel] * p1[r, in_k].sum()
                v = z.var(ddof=1) / m
                total += v
                total_fpc += v * max(0.0, 1.0 - m / int(in_k.sum()))
                df[r] += m - 1
        var[r] += total / n**2
        var_fpc[r] += total_fpc / n**2
    return var, var_fpc, np.maximum(df, 1.0)


def stratum_totals(est: np.ndarray, stratum: np.ndarray, H: int) -> np.ndarray:
    out = np.zeros((est.shape[0], H))
    np.add.at(out, (np.arange(est.shape[0])[:, None], stratum), est)
    return out


def simulate(job) -> list[dict]:
    agent, F, cost, r_rel, kappas, replicates, blocks, seed = job
    rng = np.random.default_rng(seed)
    n, y = len(F), F[:, -1]
    target, var_y, full_cost = y.mean(), y.var(), cost.sum(axis=1).mean()
    paid_table = np.concatenate([np.zeros((n, 1)), np.cumsum(cost, axis=1)], axis=1)
    idx = rng.integers(0, n, (replicates, n))
    rows = []
    for policy, kappa in kappas.items():
        reach = reach_for(F, cost, r_rel, policy, kappa)
        table = stopping.estimate_table(F, reach)
        for design in ["independent", *[f"stratified B={b}" for b in blocks]]:
            est, paid = np.empty((replicates, n)), np.empty((replicates, n))
            strata = None
            if design == "independent":
                u = rng.random((replicates, n))
            else:
                b = int(design.split("=")[1])
                u, strata = stratified_uniforms(reach[idx, 0], F[idx, 0], b, rng)
            steps_all = np.empty((replicates, n), dtype=int)
            for s in range(0, replicates, 50):  # chunks bound memory
                ii, uu = idx[s:s + 50], u[s:s + 50]
                steps = stopping.sample_stop_steps(reach[ii], uu)
                steps_all[s:s + 50] = steps
                est[s:s + 50] = table[ii, steps]
                paid[s:s + 50] = paid_table[ii, steps]
            mean = est.mean(axis=1)
            estimators = {"naive": (est.var(axis=1, ddof=1) / n, np.full(replicates, np.inf))}
            if strata is not None:
                H = int(strata.max()) + 1
                T = stratum_totals(est, strata, H)
                succ = H / (2 * (H - 1)) * (np.diff(T, axis=1) ** 2).sum(axis=1)
                estimators["successive"] = (succ / n**2, np.full(replicates, H - 1.0))
                started = steps_all >= 1
                v, v_fpc, df = design_variance(est, F[idx, 0], reach[idx, 0], started, strata, n)
                estimators["design"] = (v, df)
                estimators["design + fpc"] = (v_fpc, df)
                estimators["design + fpc, normal"] = (v_fpc, np.full(replicates, np.inf))
            actual = mean.var() * paid.mean() / (var_y / n * full_cost)
            for name, (v, df) in estimators.items():
                q = np.where(np.isinf(df), Z, student_t.ppf(0.975, np.where(np.isinf(df), 1.0, df)))
                half = q * np.sqrt(np.maximum(v, 0))
                rows.append({"agent": agent, "true_rate": target, "policy": policy, "design": design,
                             "interval": name, "coverage": float((np.abs(mean - target) <= half).mean()),
                             "miss_below": float((mean + half < target).mean()),
                             "miss_above": float((mean - half > target).mean()),
                             "bias": float(mean.mean() - target),
                             "relative_compute_actual": float(actual),
                             # compute needed to reach a given *reported* width, relative to full runs
                             "relative_compute_reported": float((half.mean() / Z) ** 2 * n * paid.mean()
                                                                / (var_y * full_cost))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", type=Path, default=Path("data/predictor/cost_aware_inputs.pkl"))
    parser.add_argument("--kappas", type=Path, default=Path(__file__).parent / "results" / "cost_aware.csv")
    parser.add_argument("--name", default="balanced")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--blocks", default="10,20,50")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    inputs = pd.read_pickle(args.inputs)
    tuned = pd.read_csv(args.kappas)
    rule = {"tracking": "tracking", "cost-aware": "+ predicted cost"}
    blocks = [int(b) for b in args.blocks.split(",")]
    jobs = []
    for i, (agent, g) in enumerate(inputs.groupby("agent", sort=True)):
        F, cost = pad_runs(g)
        r_hat = pad(g, "r_hat", 1.0)
        kappas = {p: float(tuned[(tuned["agent"] == agent) & (tuned["rule"] == r)]["kappa"].iloc[0])
                  for p, r in rule.items()}
        jobs.append((agent, F, cost, r_hat / r_hat[:, 0].mean(), kappas, args.replicates, blocks, i))
    with Pool(args.workers) as pool:
        res = pd.DataFrame([r for rows in pool.imap(simulate, jobs) for r in rows])
    out = Path(__file__).parent / "results" / f"{args.name}.csv"
    res.to_csv(out, index=False)

    geo = lambda s: float(np.exp(np.log(s).mean()))
    s = res.groupby(["policy", "design", "interval"], sort=False).agg(
        actual=("relative_compute_actual", geo), reported=("relative_compute_reported", geo),
        worst_actual=("relative_compute_actual", "max"), coverage=("coverage", "mean"),
        worst_coverage=("coverage", "min"), miss_below=("miss_below", "mean"),
        miss_above=("miss_above", "mean"), max_abs_bias=("bias", lambda b: b.abs().max()))
    with pd.option_context("display.width", 200):
        print(s.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
