import json
import random

import pytest

from shortstop import load_trajectories, rr_estimate, sample_stop_step


def exact_expectation(f, p):
    """E[Y_hat], summing over every possible stop step."""
    T = len(p)
    total, reach = 0.0, 1.0
    for s in range(T + 1):
        stop_here = reach * (1 - p[s]) if s < T else reach
        total += stop_here * rr_estimate(f, p, s)
        if s < T:
            reach *= p[s]
    return total


@pytest.mark.parametrize("outcome", [0, 1])
def test_unbiased_even_with_a_bad_predictor(outcome):
    rng = random.Random(0)
    T = 12
    f = [rng.random() for _ in range(T)] + [outcome]  # predictions are pure noise
    p = [rng.uniform(0.2, 1.0) for _ in range(T)]
    assert exact_expectation(f, p) == pytest.approx(outcome)


def test_never_stopping_returns_the_outcome():
    f = [0.3, 0.6, 0.1, 1]
    assert rr_estimate(f, [1.0, 1.0, 1.0], s=3) == pytest.approx(1)


def test_sample_stop_step_bounds():
    rng = random.Random(0)
    assert sample_stop_step([1.0] * 5, rng) == 5
    assert all(0 <= sample_stop_step([0.5] * 5, rng) <= 5 for _ in range(100))


def test_load_trajectories(tmp_path):
    rows = [
        dict(task_id="t1", agent="a", step=2, tokens_in=5, tokens_out=1,
             action="x", observation="y", outcome=1),
        dict(task_id="t1", agent="a", step=1, tokens_in=4, tokens_out=1,
             action="x", observation="y", outcome=1),
    ]
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    runs = load_trajectories(path)
    assert [s.step for s in runs[("t1", "a")]] == [1, 2]
