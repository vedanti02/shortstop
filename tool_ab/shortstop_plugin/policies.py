"""Continue-or-stop policies for the plug-in.

A policy is called before step k with the steps run so far and the predictions
(f_0, ..., f_{k-1}), and returns p_k, the probability of running step k. It must be in
(0, 1]; the estimator divides by it.

Policies and predictors are named in the agent config by dotted path to a factory plus
keyword arguments, so the cost-optimal policy and the learned predictor can be swapped in
without touching the plug-in.
"""

from __future__ import annotations

import importlib
import json
import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

from shortstop import Step

StopPolicy = Callable[[Sequence[Step], Sequence[float]], float]


def load_factory(dotted_path: str, kwargs: dict[str, Any] | None = None) -> Callable:
    """Import `package.module.factory` and call it with kwargs."""
    module_name, _, attr = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a dotted path, got {dotted_path!r}")
    factory = getattr(importlib.import_module(module_name), attr)
    return factory(**(kwargs or {}))


_SHARED: dict[str, Callable] = {}
_SHARED_LOCK = threading.Lock()


def load_shared(dotted_path: str, kwargs: dict[str, Any] | None = None) -> Callable:
    """load_factory, but built once per (path, kwargs) and reused, across threads too."""
    key = json.dumps([dotted_path, kwargs or {}], sort_keys=True, default=repr)
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = load_factory(dotted_path, kwargs)
        return _SHARED[key]


def never_stop() -> StopPolicy:
    """The stand-in: always run the next step, so every run is a full run."""
    return lambda prefix, predictions: 1.0


def constant(p: float) -> StopPolicy:
    """Run each next step with the same probability p."""
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    return lambda prefix, predictions: p


def tracking(kappa: float = 0.5, floor: float = 0.05) -> StopPolicy:
    """Keep P(reach step k) at kappa * sqrt(f (1 - f)), clipped to [floor, 1], never rising.

    Matches the heuristic in coverage/stopping.py so live runs can be compared to the
    offline simulation. Here f is the prediction made just before step k.
    """
    if not 0 < floor <= 1:
        raise ValueError(f"floor must be in (0, 1], got {floor}")

    def reach_target(f: float) -> float:
        return min(1.0, max(floor, kappa * math.sqrt(f * (1 - f))))

    def policy(prefix: Sequence[Step], predictions: Sequence[float]) -> float:
        reach_before = 1.0
        for f in predictions[:-1]:
            reach_before = min(reach_before, reach_target(f))
        reach_now = min(reach_before, reach_target(predictions[-1]))
        return reach_now / reach_before

    return policy
