import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).parent.parent / "coverage"))
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor"))
import stopping
from balanced import stratified_uniforms


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
