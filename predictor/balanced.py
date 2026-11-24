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

Two balanced designs (runtime.py has both):
    stratified B=50     every run keeps its own P_1; runtime.py's "balanced-unproven" design.
                        Not covered by the proof: its started runs are not a simple random
                        sample of the stratum.
    equal-chance B=50   every run of a stratum gets the same P_1 = m / N, a whole number m of
                        the N slices (equal_chance_uniforms), so exactly m runs start;
                        runtime.py's "balanced" design. Proven in predictor/THEORY_BALANCED.md.

The usual interval sd / sqrt(n) treats tasks as independent; under these designs it still covers
in simulation but is too cautious. The design-based variance estimators:
    design       design_variance(): theta (1 - theta) / n plus the within-stratum spread of the
                 started runs' corrections, with and without the finite-population correction,
                 t with sum (m - 1) df. "balanced-unproven" uses it with the correction. The
                 correction is right for which runs start but also shrinks the noise from how far
                 they go, so the interval is slightly too narrow (THEORY_BALANCED.md, Section 4).
    design + fpc + later steps
                 proven_variance(), on equal-chance strata: the same plus gamma times each
                 started run's own estimate of that later-step noise (later_step_table), with
                 gamma = f + (m - f) / ((m - 1) (N - 1)). Its expectation is at least the true
                 variance for any runs (THEORY_BALANCED.md, Proposition B3). With ", gamma = f"
                 the cheaper variant whose bias is the depth-profile term A, not guaranteed >= 0.
    successive   H / (2 (H - 1)) * sum_h (T_h - T_{h-1})^2 on the stratum totals T_h in sort
                 order, t with H - 1 df

Simulation as in coverage/adaptive.py: each replicate resamples the agent's tasks with
replacement, applies the policy and the design, and builds the intervals. Policies are
tracking and the learned-remaining-cost policy of cost_aware.py, each with the kappa tuned
there on the other agents; predictions are gold LightGBM (held out). The equal-chance design
draws its uniforms from a separate generator, so adding it leaves every other row unchanged.

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


def equal_chance_uniforms(p1: np.ndarray, f0: np.ndarray, block: int, rng,
                          floor: float = FLOOR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-chance strata (THEORY_BALANCED.md, Section 1): uniforms (R, n), stratum ids (R, n),
    and the step-1 reach probability every run gets (R, n).

    Runs sorted by (P_1, f0) are cut into strata of `block` runs; the last stratum also takes the
    remainder, so it has block to 2 block - 1 runs. Every run of a stratum of N runs gets the same
    step-1 reach probability m / N, with m the stratum's sum of P_1 rounded, kept between
    max(2, ceil(N floor)) and N. Uniforms are dealt as in stratified_uniforms, one slice of
    width 1 / N per run, so exactly m runs of each stratum start.
    """
    R, n = p1.shape
    order = np.lexsort((f0, p1), axis=-1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(R, 0), axis=-1)
    H = max(1, n // block)
    stratum = np.minimum(rank // block, H - 1)
    size = np.full(H, block)
    size[-1] = n - block * (H - 1)
    total = np.zeros((R, H))
    np.add.at(total, (np.arange(R)[:, None], stratum), p1)
    m = np.clip(np.rint(total), np.maximum(2, np.ceil(floor * size - 1e-9)), size)
    p_common = np.take_along_axis(m / size, stratum, axis=1)
    key = stratum * 2.0 + rng.random((R, n))  # sorts by stratum, then randomly
    pos_order = np.argsort(key, axis=-1)
    pos = np.empty_like(pos_order)
    np.put_along_axis(pos, pos_order, np.arange(n)[None, :].repeat(R, 0), axis=-1)
    within = pos - stratum * block
    return (within + rng.random((R, n))) / size[stratum], stratum, p_common


def later_step_table(F: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """table[:, s]: for a run stopped after step s, its own estimate of the noise from the steps
    after the first, Var(correction | the run started) (THEORY_BALANCED.md, Lemma B2):

        omega_hat = sum over 2 <= k <= l <= s of w_k D_l (D_l + 2 (f_{l-1} - f_{k-1})) / P_l,

    with D_l = f_l - f_{l-1} and w_k = 1 / P_k - 1 / P_{k-1}. It uses only steps 0..s, so it is
    known when the run stops; it is 0 for s < 2. Same layout as stopping.estimate_table.
    """
    n = len(F)
    f_prev, delta = F[:, :-1], np.diff(F, axis=1)
    w = 1 / reach - 1 / np.concatenate([np.ones((n, 1)), reach[:, :-1]], axis=1)
    w[:, 0] = 0.0  # the sums start at k = 2
    W, Q = np.cumsum(w, axis=1), np.cumsum(w * f_prev, axis=1)
    term = delta / reach * ((delta + 2 * f_prev) * W - 2 * Q)
    return np.concatenate([np.zeros((n, 1)), np.cumsum(term, axis=1)], axis=1)


def omega_hat(f, reach) -> float:
    """later_step_table for one run: f = its predictions f_0..f_s (f_s = Y if it finished) and
    reach = its reach probabilities P_1..P_s, s = len(reach)."""
    s = len(reach)
    if s < 2:
        return 0.0
    F = np.asarray(f[: s + 1], dtype=float)[None, :]
    return float(later_step_table(F, np.asarray(reach, dtype=float)[None, :])[0, s])


def proven_variance(est, f0, started, stratum, omega, n, gamma: str = "safe") -> tuple[np.ndarray, np.ndarray]:
    """Variance of the mean under equal-chance strata, and its degrees of freedom
    (THEORY_BALANCED.md, Section 2):

        theta (1 - theta) / n + (1/n^2) sum over strata of [(1 - f) m s^2 + gamma sum_started omega_hat]

    for a stratum of N runs with m started (f = m / N), s^2 the sample variance of the started
    runs' corrections c = est - f0, and omega_hat from later_step_table. gamma "safe" is
    f + (m - f) / ((m - 1) (N - 1)), whose expectation is at least the true variance for any runs
    (Proposition B3); gamma "f" is f, whose bias is the depth-profile term A. df = sum (m - 1).
    """
    theta = np.clip(est.mean(axis=1), 0, 1)
    var = theta * (1 - theta) / n
    df = np.zeros(len(est))
    c = est - f0
    for h in range(int(stratum.max()) + 1):
        in_h = stratum == h
        sel = started & in_h
        m, N = sel.sum(axis=1), in_h.sum(axis=1)
        if (m < 2).any():
            raise ValueError("every stratum needs at least 2 started runs (equal_chance_uniforms ensures it)")
        f = m / N
        mean_c = np.where(sel, c, 0.0).sum(axis=1) / m
        s2 = np.where(sel, (c - mean_c[:, None]) ** 2, 0.0).sum(axis=1) / (m - 1)
        g = f + (m - f) / ((m - 1) * (N - 1)) if gamma == "safe" else f
        var = var + ((1 - f) * m * s2 + g * np.where(sel, omega, 0.0).sum(axis=1)) / n ** 2
        df = df + m - 1
    return var, df


def stratum_totals(est: np.ndarray, stratum: np.ndarray, H: int) -> np.ndarray:
    out = np.zeros((est.shape[0], H))
    np.add.at(out, (np.arange(est.shape[0])[:, None], stratum), est)
    return out


def equal_chance_draws(F, reach, block: int, idx, paid_table, rng):
    """One evaluation per row of idx under equal-chance strata: estimates, tokens paid,
    omega_hat, stratum ids and stop steps, each (R, n). The reach probabilities are the policy's
    with P_1 replaced by the stratum's common value and later steps capped at it."""
    R, n = idx.shape
    est, paid, omega = np.empty((R, n)), np.empty((R, n)), np.empty((R, n))
    strata, steps_all = np.empty((R, n), dtype=int), np.empty((R, n), dtype=int)
    for s in range(0, R, 50):  # chunks bound memory
        ii = idx[s:s + 50]
        r = len(ii)
        u, st, p_common = equal_chance_uniforms(reach[ii, 0], F[ii, 0], block, rng)
        reach_eq = np.minimum(reach[ii], p_common[..., None])
        reach_eq[..., 0] = p_common
        steps = stopping.sample_stop_steps(reach_eq, u)
        flat_F, flat_reach, flat_steps = F[ii].reshape(r * n, -1), reach_eq.reshape(r * n, -1), steps.reshape(-1)
        at = np.arange(r * n)
        est[s:s + r] = stopping.estimate_table(flat_F, flat_reach)[at, flat_steps].reshape(r, n)
        omega[s:s + r] = later_step_table(flat_F, flat_reach)[at, flat_steps].reshape(r, n)
        paid[s:s + r] = paid_table[ii, steps]
        strata[s:s + r], steps_all[s:s + r] = st, steps
    return est, paid, omega, strata, steps_all


def simulate(job) -> list[dict]:
    """job = (agent, F, cost, r_rel, kappas, replicates, blocks, seed[, equal_blocks]). blocks are
    the stratum sizes of the stratified design; equal_blocks (none by default) adds the
    equal-chance design with those sizes, from its own random generator."""
    agent, F, cost, r_rel, kappas, replicates, blocks, seed = job[:8]
    equal_blocks = job[8] if len(job) > 8 else []
    rng = np.random.default_rng(seed)
    rng_equal = np.random.default_rng([seed, 1])
    n, y = len(F), F[:, -1]
    target, var_y, full_cost = y.mean(), y.var(), cost.sum(axis=1).mean()
    paid_table = np.concatenate([np.zeros((n, 1)), np.cumsum(cost, axis=1)], axis=1)
    idx = rng.integers(0, n, (replicates, n))
    rows = []

    def add_rows(policy: str, design: str, estimators: dict, mean: np.ndarray, paid: np.ndarray) -> None:
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
            add_rows(policy, design, estimators, mean, paid)
        for b in equal_blocks:
            est, paid, omega, strata, steps = equal_chance_draws(F, reach, b, idx, paid_table, rng_equal)
            started, f0 = steps >= 1, F[idx, 0]
            estimators = {"design + fpc + later steps": proven_variance(est, f0, started, strata, omega, n),
                          "design + fpc + later steps, gamma = f":
                              proven_variance(est, f0, started, strata, omega, n, gamma="f")}
            add_rows(policy, f"equal-chance B={b}", estimators, est.mean(axis=1), paid)
    return rows


def main() -> None:
    # imported here, not at the top: runtime.py imports this module, and cost_aware's imports
    # (train_lgbm, the stand-in predictor) need scikit-learn, which a live harness need not have
    from cost_aware import pad

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", type=Path, default=Path("data/predictor/cost_aware_inputs.pkl"))
    parser.add_argument("--kappas", type=Path, default=Path(__file__).parent / "results" / "cost_aware.csv")
    parser.add_argument("--name", default="balanced")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--blocks", default="10,20,50")
    parser.add_argument("--equal-blocks", default="50",
                        help="stratum sizes for the equal-chance design (runtime.py's balanced); empty for none")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    inputs = pd.read_pickle(args.inputs)
    tuned = pd.read_csv(args.kappas)
    rule = {"tracking": "tracking", "cost-aware": "+ predicted cost"}
    blocks = [int(b) for b in args.blocks.split(",")]
    equal_blocks = [int(b) for b in args.equal_blocks.split(",") if b]
    jobs = []
    for i, (agent, g) in enumerate(inputs.groupby("agent", sort=True)):
        F, cost = pad_runs(g)
        r_hat = pad(g, "r_hat", 1.0)
        kappas = {p: float(tuned[(tuned["agent"] == agent) & (tuned["rule"] == r)]["kappa"].iloc[0])
                  for p, r in rule.items()}
        jobs.append((agent, F, cost, r_hat / r_hat[:, 0].mean(), kappas, args.replicates, blocks, i, equal_blocks))
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
