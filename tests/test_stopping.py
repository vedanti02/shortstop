import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "coverage"))
import stopping


def random_runs(n=6, T=9, seed=0):
    """Padded runs of different lengths with noise predictions."""
    rng = np.random.default_rng(seed)
    F = rng.random((n, T + 1))
    cost = rng.uniform(1, 5, (n, T))
    for i, length in enumerate(rng.integers(1, T + 1, n)):
        F[i, length:] = rng.integers(0, 2)
        cost[i, length:] = 0
    return F, cost


def enumerate_moments(F, cost, reach):
    """Mean, variance and expected cost by summing over every stop step."""
    table = stopping.estimate_table(F, reach)
    reach0 = np.concatenate([np.ones((len(F), 1)), reach], axis=1)
    stop_prob = reach0 - np.concatenate([reach, np.zeros((len(F), 1))], axis=1)
    paid = np.concatenate([np.zeros((len(F), 1)), np.cumsum(cost, axis=1)], axis=1)
    mean = (stop_prob * table).sum(axis=1)
    variance = (stop_prob * table**2).sum(axis=1) - mean**2
    return mean, variance, (stop_prob * paid).sum(axis=1)


POLICIES = [
    lambda F: stopping.reach_full(F),
    lambda F: stopping.reach_one_shot(F, 0.3),
    lambda F: stopping.reach_threshold(F, 0.2, 0.8, 0.25),
    lambda F: stopping.reach_tracking(F, 1.5, 0.05),
]


@pytest.mark.parametrize("policy", POLICIES)
def test_policies_are_unbiased_and_moments_match(policy):
    F, cost = random_runs()
    reach = policy(F)
    assert (reach > 0).all() and (np.diff(reach, axis=1) <= 1e-12).all()
    mean, variance, expected_cost = enumerate_moments(F, cost, reach)
    assert mean == pytest.approx(F[:, -1])
    exact_var, exact_cost = stopping.exact_moments(F, cost, reach)
    assert exact_var == pytest.approx(variance)
    assert exact_cost == pytest.approx(expected_cost)


def test_sampled_stop_steps_follow_reach_probabilities():
    F, _ = random_runs(n=1)
    reach = stopping.reach_tracking(F, 1.5, 0.05)
    u = np.random.default_rng(1).random((200_000, 1))
    steps = stopping.sample_stop_steps(np.broadcast_to(reach, (200_000, 1, reach.shape[1])), u)
    for k in range(1, reach.shape[1] + 1):
        assert (steps >= k).mean() == pytest.approx(reach[0, k - 1], abs=0.01)


def test_deterministic_stop():
    F = np.array([[0.5, 0.9, 0.5, 0.0], [0.5, 0.5, 0.5, 1.0], [0.1, 0.5, 0.5, 1.0]])
    cost = np.ones((3, 3))
    label, paid = stopping.deterministic_stop(F, cost, lo=0.2, hi=0.8)
    assert label.tolist() == [1.0, 1.0, 0.0]  # stopped high, ran to the end, stopped low
    assert paid.tolist() == [1.0, 3.0, 0.0]
