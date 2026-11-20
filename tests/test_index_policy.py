import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent / "coverage"))
import optimal_policy as op

P_MIN = 1e-3  # low enough that the floor never binds in these cases


def tree_runs(seed):
    """Runs on a 3-step tree whose prediction bin identifies the whole history.

    One root, 3 children, 9 grandchildren, each in its own prediction bin, so a policy
    that sees (step, prediction bin) sees everything. Success rates, predictions and step
    costs are random, so some steps are cheap and informative and others are neither.
    """
    rng = np.random.default_rng(seed)
    f1 = (np.array([1, 4, 8]) + rng.uniform(0.1, 0.9, 3)) / 10
    f2 = (np.arange(9).reshape(3, 3) + rng.uniform(0.1, 0.9, (3, 3))) / 10
    success = rng.integers(0, 9, (3, 3))  # successes out of 8 runs at each grandchild
    rows, node1, node2 = [], [], []
    for a in range(3):
        for b in range(3):
            for j in range(8):
                rows.append([0.55, f1[a], f2[a, b], float(j < success[a, b])])
                node1.append(a)
                node2.append(3 * a + b)
    F = np.array(rows)
    cost = rng.uniform(0.5, 1.5, (len(F), 3)) * rng.choice([0.1, 1.0, 10.0], 3)
    return F, cost, np.array(node1), np.array(node2)


def lagrangian(F, cost, reach, lam):
    """Variance added by stopping plus lam * cost, with cost in units of the mean run cost."""
    sq = (F[:, -1:] - F) ** 2
    added = ((sq[:, :-1] - sq[:, 1:]) / reach).sum(axis=1).mean()
    return added + lam * (cost * reach).sum(axis=1).mean() / cost.sum(axis=1).mean()


def brute_force(F, cost, node1, node2, lam):
    """Best value over every non-increasing assignment of reach probabilities to tree nodes.

    In u = 1 / reach the problem is convex, with u built from non-negative increments.
    """
    def reach_of(w):
        u1 = 1 + w[0]
        u2 = u1 + w[1:4][node1]
        return 1 / np.column_stack([np.full(len(F), u1), u2, u2 + w[4:13][node2]])

    result = minimize(lambda w: lagrangian(F, cost, reach_of(w), lam), np.ones(13),
                      method="L-BFGS-B", bounds=[(0, 1 / P_MIN)] * 13,
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000})
    return result.fun


def run_to_end_policy(F, cost, node1, node2, lam):
    """Learned tracking with exact per-node expectations."""
    sq = (F[:, -1:] - F) ** 2
    remaining = np.cumsum(cost[:, ::-1], axis=1)[:, ::-1] / cost.sum(axis=1).mean()
    target = np.empty((len(F), 3))
    for k, node in enumerate([np.zeros(len(F), int), node1, node2]):
        for v in np.unique(node):
            m = node == v
            target[m, k] = np.sqrt(sq[m, k].mean() / remaining[m, k].mean() / lam)
    return np.minimum.accumulate(np.clip(target, P_MIN, 1), axis=1)


def test_index_policy_matches_brute_force_optimum():
    strictly_better = 0
    for seed in range(12):
        F, cost, node1, node2 = tree_runs(seed)
        stats = op.index_stats(F, cost, np.full(len(F), 3), 3)
        index = op.fit_index_table([stats], shrink=0.0)
        for lam in (0.05, 0.3, 2.0):
            ours = lagrangian(F, cost, op.reach_index(F, index, 1 / np.sqrt(lam), P_MIN), lam)
            best = brute_force(F, cost, node1, node2, lam)
            assert abs(ours - best) <= 2e-4 * abs(best) + 1e-6, (seed, lam, ours, best)
            simple = lagrangian(F, cost, run_to_end_policy(F, cost, node1, node2, lam), lam)
            assert simple >= best - 1e-9
            strictly_better += simple > best * 1.0005
    assert strictly_better >= 5  # the look-ahead matters in these cases
