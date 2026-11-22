import math

import pytest
from shortstop_plugin.policies import constant, load_factory, never_stop, tracking


def test_never_stop_always_continues():
    policy = never_stop()
    assert policy([], (0.5,)) == 1.0
    assert policy([], (0.01, 0.99, 0.5)) == 1.0


def test_constant_policy_validates_its_probability():
    assert constant(0.3)([], (0.5,)) == 0.3
    with pytest.raises(ValueError):
        constant(0.0)
    with pytest.raises(ValueError):
        constant(1.5)


def test_tracking_reach_never_increases_and_stays_above_floor():
    policy = tracking(kappa=1.0, floor=0.05)
    preds = (0.5, 0.9, 0.99, 0.5, 0.999)
    reach, reaches = 1.0, []
    for k in range(1, len(preds) + 1):
        p_k = policy([], preds[:k])
        assert 0 < p_k <= 1
        reach *= p_k
        reaches.append(reach)
    assert all(b <= a + 1e-12 for a, b in zip(reaches, reaches[1:]))
    assert reaches[-1] >= 0.05 - 1e-12
    # Reach after the confident 0.99 prediction is kappa * sqrt(0.99 * 0.01).
    assert reaches[2] == pytest.approx(math.sqrt(0.99 * 0.01))


def test_load_factory_resolves_dotted_paths():
    policy = load_factory("shortstop_plugin.policies.constant", {"p": 0.7})
    assert policy([], (0.5,)) == 0.7
    predictor = load_factory("shortstop.predictor.constant_predictor", {"p": 0.2})
    assert predictor([]) == 0.2
