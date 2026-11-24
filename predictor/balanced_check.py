"""Check THEORY_BALANCED.md's Proposition B3 on real runs: the exact bias of each variance formula.

    python predictor/balanced_check.py                 # Verified, 20,000 draws per agent

For each agent (held-out gold LightGBM, learned-remaining-cost policy with its tuned kappa), the
500 tasks are fixed and cut into equal-chance strata of 50, as runtime.py's balanced design does.
Per stratum, from the full logged runs:

    Var       exact variance of the stratum's sum of estimates (slot integrals, Hoeffding's
              permutation variance)
    A         the depth-profile term of Proposition B3
    omega_j   each run's later-step noise, Var(correction | started)

and so the exact expectation of three variance formulas, as multiples of Var:

    old       design_variance with the finite-population factor (balanced-unproven's formula,
              here on equal-chance strata): E = Var + A - f^2 sum omega
    gamma=f   proven_variance(gamma="f"): E = Var + A
    proven    proven_variance(): E = Var + A + f (gamma - f) sum omega >= Var

Monte Carlo draws of the production code (equal_chance_uniforms, design_variance,
proven_variance) check each expectation and Var itself.

Output: predictor/results/balanced_check.csv
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stopping
from balanced import (
    design_variance,
    equal_chance_uniforms,
    later_step_table,
    proven_variance,
    reach_for,
)
from cost_aware import pad
from experiments import pad_runs

BLOCK = 50


def slot_moments(values: np.ndarray, reach: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance over each slot [a/N, (a+1)/N) of u -> values[j, s(u)], s(u) = #{k : P_k > u}."""
    n = len(values)
    upper = np.concatenate([np.ones((n, 1)), reach], axis=1)
    lower = np.concatenate([reach, np.zeros((n, 1))], axis=1)
    a = np.arange(N)
    overlap = np.clip(np.minimum(upper[:, None, :], (a + 1)[None, :, None] / N)
                      - np.maximum(lower[:, None, :], a[None, :, None] / N), 0, None)
    mean = N * np.einsum("jak,jk->ja", overlap, values)
    second = N * np.einsum("jak,jk->ja", overlap, values ** 2)
    return mean, np.maximum(second - mean ** 2, 0)


def check_agent(job) -> dict:
    agent, F, cost, r_rel, kappa, draws, seed = job
    rng = np.random.default_rng(seed)
    n = len(F)
    reach0 = reach_for(F, cost, r_rel, "cost-aware", kappa)
    f0 = F[:, 0]
    _, strata, p = equal_chance_uniforms(reach0[None, :, 0], f0[None, :], BLOCK, rng)
    strata, p = strata[0], p[0]
    reach = np.minimum(reach0, p[:, None])
    reach[:, 0] = p
    table, later = stopping.estimate_table(F, reach), later_step_table(F, reach)

    H = int(strata.max()) + 1
    exact = {"var": np.zeros(H), "old": np.zeros(H), "f": np.zeros(H), "proven": np.zeros(H), "A": np.zeros(H)}
    for h in range(H):
        idx = np.flatnonzero(strata == h)
        N = len(idx)
        m = int(np.rint(p[idx[0]] * N))
        f = m / N
        mu, tau2 = slot_moments(table[idx], reach[idx], N)
        d = mu - mu.mean(axis=1, keepdims=True) - mu.mean(axis=0, keepdims=True) + mu.mean()
        var = tau2.sum() / N + (d ** 2).sum() / (N - 1)
        nu = mu[:, :m] - f0[idx, None]  # the correction's slot means over the started slots
        W = nu - nu.mean(axis=1, keepdims=True)
        omega = (tau2[:, :m] + W ** 2).mean(axis=1)
        A = (m - f) / ((m - 1) * N * (N - 1)) * ((W.sum(axis=0) ** 2).sum() - (W ** 2).sum())
        gamma = f + (m - f) / ((m - 1) * (N - 1))
        exact["var"][h], exact["A"][h] = var, A
        exact["old"][h] = var + A - f * f * omega.sum()
        exact["f"][h] = var + A
        exact["proven"][h] = var + A + f * (gamma - f) * omega.sum()

    # Monte Carlo of the production code; the stratum part of each formula, summed over strata
    sums, mc = [], {"old": [], "f": [], "proven": []}
    for _ in range(draws // 1000):
        u, st, pp = equal_chance_uniforms(np.tile(reach0[:, 0], (1000, 1)), np.tile(f0, (1000, 1)), BLOCK, rng)
        assert (st == strata).all() and (pp == p).all()
        steps = stopping.sample_stop_steps(np.broadcast_to(reach, (1000,) + reach.shape), u)
        est, om = table[np.arange(n), steps], later[np.arange(n), steps]
        started = steps >= 1
        first = np.clip(est.mean(axis=1), 0, 1)
        first = first * (1 - first) / n
        f0s, ps = np.tile(f0, (1000, 1)), np.tile(p, (1000, 1))
        _, v_old, _ = design_variance(est, f0s, ps, started, st, n)
        v_f, _ = proven_variance(est, f0s, started, st, om, n, gamma="f")
        v_p, _ = proven_variance(est, f0s, started, st, om, n)
        for key, v in [("old", v_old), ("f", v_f), ("proven", v_p)]:
            mc[key].append((v - first) * n ** 2)
        sums.append(est.sum(axis=1))
    var = exact["var"].sum()
    row = {"agent": agent, "true_rate": F[:, -1].mean(), "strata": H, "A/var": exact["A"].sum() / var,
           "strata_with_A<0": int((exact["A"] < 0).sum()),
           "mc_var/var": np.concatenate(sums).var(ddof=1) / var}
    for key, values in mc.items():
        row[f"{key}/var"] = exact[key].sum() / var
        row[f"{key}_mc/var"] = np.concatenate(values).mean() / var
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", type=Path, default=Path("data/predictor/cost_aware_inputs.pkl"))
    parser.add_argument("--kappas", type=Path, default=Path(__file__).parent / "results" / "cost_aware.csv")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    inputs, tuned = pd.read_pickle(args.inputs), pd.read_csv(args.kappas)
    jobs = []
    for i, (agent, g) in enumerate(inputs.groupby("agent", sort=True)):
        F, cost = pad_runs(g)
        r_hat = pad(g, "r_hat", 1.0)
        kappa = float(tuned[(tuned["agent"] == agent) & (tuned["rule"] == "+ predicted cost")]["kappa"].iloc[0])
        jobs.append((agent, F, cost, r_hat / r_hat[:, 0].mean(), kappa, args.draws, i))
    with Pool(args.workers) as pool:
        res = pd.DataFrame(pool.map(check_agent, jobs))
    out = Path(__file__).parent / "results" / "balanced_check.csv"
    res.to_csv(out, index=False)

    gap = max((res[f"{k}_mc/var"] - res[f"{k}/var"]).abs().max() for k in ["old", "f", "proven"])
    print(f"{len(res)} agents, {args.draws} draws each. Exact expectation / true variance (median, min to max):")
    for key, label in [("old", "old formula (fpc on everything)"), ("f", "gamma = f"), ("proven", "proven")]:
        s = res[f"{key}/var"]
        print(f"  {label:32s} {s.median():.3f} ({s.min():.3f} to {s.max():.3f}); below 1 for {int((s < 1).sum())} agents")
    print(f"  A / Var: {res['A/var'].median():+.4f} ({res['A/var'].min():+.4f} to {res['A/var'].max():+.4f}); "
          f"negative in {int(res['strata_with_A<0'].sum())} of {int(res['strata'].sum())} strata")
    print(f"Monte Carlo checks: largest gap between simulated and exact expectation {gap:.4f} of Var; "
          f"simulated / exact variance {res['mc_var/var'].min():.3f} to {res['mc_var/var'].max():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
