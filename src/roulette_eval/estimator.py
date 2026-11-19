"""Reference randomized truncation ("Russian roulette") estimator.

Notation, for a run of T steps:
    f[0..T-1]  predictions after 0..T-1 steps, and f[T] = Y, the true outcome
    p[k-1]     probability of continuing into step k (k = 1..T), never zero
"""

from __future__ import annotations

import random
from typing import Sequence


def sample_stop_step(p: Sequence[float], rng: random.Random) -> int:
    """Sample s, the number of steps actually run (0..T)."""
    for k, p_k in enumerate(p, start=1):
        if rng.random() >= p_k:
            return k - 1
    return len(p)


def rr_estimate(f: Sequence[float], p: Sequence[float], s: int) -> float:
    """Unbiased estimate of Y from a run stopped after step s.

    Only f[0..s] is used, so f[s+1:] may be unknown when s < T.
    """
    estimate = f[0]
    reach = 1.0
    for k in range(1, s + 1):
        reach *= p[k - 1]
        estimate += (f[k] - f[k - 1]) / reach
    return estimate
