import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).parent.parent / "coverage"))
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor"))
import stopping
from balanced import (
    equal_chance_uniforms,
    later_step_table,
    omega_hat,
    proven_variance,
    stratified_uniforms,
)


def test_each_stratum_is_a_latin_hypercube():
    rng = np.random.default_rng(0)
    R, n, block = 20, 47, 10
    p1, f0 = rng.random((R, n)), rng.random((R, n))
    u, stratum = stratified_uniforms(p1, f0, block, rng)
    assert ((u >= 0) & (u < 1)).all()
    for r in range(R):
        for h in np.unique(stratum[r]):
            members = u[r, stratum[r] == h]
            cells = np.floor(members * len(members)).astype(int)
            assert sorted(cells) == list(range(len(members)))


def test_every_run_gets_a_uniform():
    rng = np.random.default_rng(1)
    R, n = 4000, 30
    p1 = np.tile(rng.random(n), (R, 1))  # same runs every replicate, as in the simulation
    f0 = np.tile(rng.random(n), (R, 1))
    u, _ = stratified_uniforms(p1, f0, 7, rng)
    for i in [0, 13, 29]:
        assert kstest(u[:, i], "uniform").pvalue > 0.001


def test_estimates_stay_unbiased():
    rng = np.random.default_rng(2)
    n, T, R = 24, 6, 20000
    F = rng.uniform(0.05, 0.95, (n, T + 1))
    F[:, -1] = rng.integers(0, 2, n)
    reach = np.minimum.accumulate(rng.uniform(0.1, 1.0, (n, T)), axis=1)
    table = stopping.estimate_table(F, reach)
    u, _ = stratified_uniforms(np.tile(reach[:, 0], (R, 1)), np.tile(F[:, 0], (R, 1)), 6, rng)
    steps = stopping.sample_stop_steps(np.broadcast_to(reach, (R, n, T)), u)
    est = table[np.arange(n)[None, :], steps].mean(axis=0)
    sd = table[np.arange(n)[None, :], steps].std(axis=0) / np.sqrt(R)
    assert np.all(np.abs(est - F[:, -1]) < 5 * sd + 1e-9)


# --- equal-chance strata and their interval (predictor/THEORY_BALANCED.md) ---------------------


def random_runs(rng, n, T, p1=None):
    """n runs of T steps: predictions, 0/1 outcomes, and non-increasing reach probabilities with
    step-1 probability p1 (random if None) and later ones at least 0.05."""
    F = rng.uniform(0.02, 0.98, (n, T + 1))
    F[:, -1] = rng.integers(0, 2, n)
    first = rng.uniform(0.1, 1.0, n) if p1 is None else np.full(n, p1)
    later = np.sort(rng.uniform(0.05, 1.0, (n, T - 1)), axis=1)[:, ::-1] * first[:, None]
    return F, np.concatenate([first[:, None], np.maximum(later, 0.05)], axis=1)


def slot_means(values, reach, N):
    """Means over the slots [a/N, (a+1)/N) of u -> values[j, s(u)], s(u) = #{k : reach_k > u}."""
    n = len(values)
    upper = np.concatenate([np.ones((n, 1)), reach], axis=1)
    lower = np.concatenate([reach, np.zeros((n, 1))], axis=1)
    a = np.arange(N)
    overlap = np.clip(np.minimum(upper[:, None, :], (a + 1)[None, :, None] / N)
                      - np.maximum(lower[:, None, :], a[None, :, None] / N), 0, None)
    return N * np.einsum("jak,jk->ja", overlap, values)


def test_equal_chance_strata_start_exactly_m_runs():
    rng = np.random.default_rng(3)
    R, n, block = 40, 123, 20  # five strata of 20 and a last one of 23
    p1, f0 = rng.uniform(0.05, 0.6, (R, n)), rng.random((R, n))
    u, stratum, p = equal_chance_uniforms(p1, f0, block, rng)
    for r in range(R):
        assert np.bincount(stratum[r]).tolist() == [20] * 5 + [23]
        for h in range(6):
            members = stratum[r] == h
            size = members.sum()
            assert np.unique(p[r, members]).size == 1  # one chance to start per stratum
            m = int(np.clip(np.rint(p1[r, members].sum()), max(2, np.ceil(0.05 * size - 1e-9)), size))
            assert np.isclose(p[r, members][0], m / size)
            assert (u[r, members] < p[r, members]).sum() == m  # exactly m runs start
            assert sorted(np.floor(u[r, members] * size).astype(int)) == list(range(size))
    # strata follow the sort by P_1: every run of a stratum has P_1 at most the next stratum's
    order_ok = [p1[0, stratum[0] == h].max() <= p1[0, stratum[0] == h + 1].min() for h in range(5)]
    assert all(order_ok)


def test_later_step_noise_estimate_is_unbiased():
    """Lemma B2: given a start, omega_hat averages exactly to Var(correction | started)."""
    rng = np.random.default_rng(4)
    n, T = 40, 7
    F, reach = random_runs(rng, n, T)
    om = later_step_table(F, reach)
    P = np.concatenate([reach, np.zeros((n, 1))], axis=1)
    p_stop = (P[:, :-1] - P[:, 1:]) / reach[:, :1]  # P(stop after s | started), s = 1..T
    w = 1 / reach - 1 / np.concatenate([np.ones((n, 1)), reach[:, :-1]], axis=1)
    sq = (F[:, -1:] - F) ** 2
    omega = (w[:, 1:] * sq[:, 1:-1]).sum(axis=1) / reach[:, 0]
    assert np.allclose((p_stop * om[:, 1:]).sum(axis=1), omega)
    c = stopping.estimate_table(F, reach)[:, 1:] - F[:, :1]
    assert np.allclose((p_stop * c ** 2).sum(axis=1) - (p_stop * c).sum(axis=1) ** 2, omega)
    for j in range(5):  # one run at a time, as runtime.py computes it
        for s in range(1, T + 1):
            assert np.isclose(omega_hat(F[j, : s + 1], reach[j, :s]), om[j, s])


def test_variance_identities_are_exact():
    """Proposition B3, by enumerating every permutation of a small stratum and integrating the
    within-slot uniforms exactly: E[V] - Var = A + f (gamma - f) sum omega, the safe gamma makes
    it nonnegative, the old formula's bias is A - f^2 sum omega, and Var <= N/(N-1) sum v."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        N = int(rng.integers(3, 6))
        m = int(rng.integers(2, N + 1))
        T = int(rng.integers(1, 6))
        p = f = m / N
        F, reach = random_runs(rng, N, T, p1=p)
        c_tab = stopping.estimate_table(F, reach) - F[:, :1]
        nu, q = slot_means(c_tab, reach, N), slot_means(c_tab ** 2, reach, N)
        o = slot_means(later_step_table(F, reach), reach, N)
        e = F[:, -1] - F[:, 0]
        omega = q[:, :m].mean(axis=1) - (e / p) ** 2
        ev_old = ev_c2 = ev_omega = 0.0
        perms = list(itertools.permutations(range(N)))
        for sigma in perms:
            js = [j for j in range(N) if sigma[j] < m]  # the started runs
            mu = np.array([nu[j, sigma[j]] for j in js])
            cross = mu.sum() ** 2 - (mu ** 2).sum()
            q_sum = sum(q[j, sigma[j]] for j in js)
            ev_old += (1 - f) * m * ((1 - 1 / m) * q_sum - cross / m) / (m - 1)
            ev_c2 += q_sum + cross
            ev_omega += sum(o[j, sigma[j]] for j in js)
        ev_old, ev_c2, ev_omega = ev_old / len(perms), ev_c2 / len(perms), ev_omega / len(perms)
        var = ev_c2 - e.sum() ** 2
        assert np.isclose(ev_omega, f * omega.sum())
        W = nu[:, :m] - (e / p)[:, None]
        A = (m - f) / ((m - 1) * N * (N - 1)) * ((W.sum(axis=0) ** 2).sum() - (W ** 2).sum())
        assert np.isclose(ev_old + f * ev_omega - var, A, atol=1e-9)
        assert np.isclose(ev_old - var, A - f * f * omega.sum(), atol=1e-9)
        gamma = f + (m - f) / ((m - 1) * (N - 1))
        assert ev_old + gamma * ev_omega >= var - 1e-9
        assert var <= N / (N - 1) * (q.mean(axis=1) - e ** 2).sum() + 1e-9


def test_proven_variance_matches_its_expectation():
    """The production estimator, simulated on one stratum, against Proposition B3's exact value."""
    rng = np.random.default_rng(6)
    N, T, R = 6, 4, 40000
    F, reach = random_runs(rng, N, T, p1=0.5)  # m = 3 of 6 start
    u, strata, p = equal_chance_uniforms(np.full((R, N), 0.5), np.tile(F[:, 0], (R, 1)), N, rng)
    assert (p == 0.5).all() and (strata == 0).all()
    steps = stopping.sample_stop_steps(np.broadcast_to(reach, (R, N, T)), u)
    rows = np.arange(N)[None, :]
    est = stopping.estimate_table(F, reach)[rows, steps]
    omega_draws = later_step_table(F, reach)[rows, steps]
    var, df = proven_variance(est, np.tile(F[:, 0], (R, 1)), steps >= 1, strata, omega_draws, N)
    theta = np.clip(est.mean(axis=1), 0, 1)
    v_stratum = (var - theta * (1 - theta) / N) * N ** 2
    assert (df == 2).all()
    # exact value
    m, f = 3, 0.5
    c_tab = stopping.estimate_table(F, reach) - F[:, :1]
    nu, q = slot_means(c_tab, reach, N), slot_means(c_tab ** 2, reach, N)
    e = F[:, -1] - F[:, 0]
    omega = q[:, :m].mean(axis=1) - (e / f) ** 2
    W = nu[:, :m] - (e / f)[:, None]
    A = (m - f) / ((m - 1) * N * (N - 1)) * ((W.sum(axis=0) ** 2).sum() - (W ** 2).sum())
    gamma = f + (m - f) / ((m - 1) * (N - 1))
    second_moments = []  # E[C^2 | permutation], the within-slot uniforms integrated exactly
    for sigma in itertools.permutations(range(N)):
        js = [j for j in range(N) if sigma[j] < m]
        slots = [sigma[j] for j in js]
        mu = nu[js, slots]
        second_moments.append(q[js, slots].sum() + mu.sum() ** 2 - (mu ** 2).sum())
    exact_var = np.mean(second_moments) - e.sum() ** 2
    assert abs(est.sum(axis=1).var() / exact_var - 1) < 0.03
    expected = exact_var + A + f * (gamma - f) * omega.sum()
    assert abs(v_stratum.mean() / expected - 1) < 0.03
    sd = est.std(axis=0) / np.sqrt(R)
    assert np.all(np.abs(est.mean(axis=0) - F[:, -1]) < 5 * sd + 1e-9)  # every run stays unbiased
