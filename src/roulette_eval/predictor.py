"""Predictor interface: trajectory prefix in, success probability out."""

from __future__ import annotations

from typing import Callable, Sequence

from .trajectory import Step

# Takes the steps seen so far (empty before step 1) and returns P(success) in [0, 1].
Predictor = Callable[[Sequence[Step]], float]


def constant_predictor(p: float = 0.5) -> Predictor:
    """Stand-in predictor that ignores the trajectory."""
    return lambda prefix: p
