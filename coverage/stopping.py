"""Stopping rules and their estimators, vectorized over the runs of one agent.

Runs are padded to a common length. For n runs and at most T steps:
    F     (n, T + 1)  F[:, k] is the prediction after k steps; from the run's last step on
                      it holds the true outcome Y, so padded steps change nothing
    cost  (n, T)      cost[:, k - 1] is the cost of step k, zero for padded steps

A randomized policy is described by its reach probabilities, reach[:, k - 1] =
P(run reaches step k). They must be positive, non-increasing in k, and depend only on
predictions made before step k.
"""

from __future__ import annotations

import numpy as np

# --- deterministic early stopping (the baseline being tested) ------------------------------


def deterministic_stop(F: np.ndarray, cost: np.ndarray, lo: float, hi: float):
    """Stop at the first prediction outside (lo, hi) and score the run by that prediction.

    Returns (label, cost_paid) per run. Runs that never trigger are scored by their outcome.
    """
    preds = F[:, :-1]
    confident = (preds <= lo) | (preds >= hi)
    stop = np.where(confident.any(axis=1), confident.argmax(axis=1), preds.shape[1])
    rows = np.arange(len(F))
    stopped_early = stop < preds.shape[1]
    label = np.where(stopped_early, F[rows, np.minimum(stop, preds.shape[1] - 1)] >= hi, F[:, -1])
    paid = np.concatenate([np.zeros((len(F), 1)), np.cumsum(cost, axis=1)], axis=1)[rows, stop]
    return label.astype(float), paid


# --- reach probabilities for randomized policies ------------------------------------------


def reach_full(F: np.ndarray) -> np.ndarray:
    """Never stop early."""
    return np.ones_like(F[:, :-1])


def reach_one_shot(F: np.ndarray, q: float) -> np.ndarray:
    """Run each task to the end with probability q, otherwise use only the step-0 prediction.

    This is prediction-powered inference with a fixed predictor.
    """
    return np.full_like(F[:, :-1], q)


def reach_threshold(F: np.ndarray, lo: float, hi: float, q: float) -> np.ndarray:
    """Randomized version of deterministic stopping.

    At the first prediction outside (lo, hi), continue with probability q instead of stopping.
    """
    preds = F[:, :-1]
    triggered = np.maximum.accumulate((preds <= lo) | (preds >= hi), axis=1)
    return np.where(triggered, q, 1.0)


def reach_tracking(F: np.ndarray, kappa: float, p_min: float) -> np.ndarray:
    """Keep the reach probability proportional to how uncertain the predictor is.

    The target is kappa * sqrt(f (1 - f)), the predicted size of the remaining error, clipped
    to [p_min, 1]. Reach probabilities can only go down, so the running minimum is used.
    """
    preds = F[:, :-1]
    target = np.clip(kappa * np.sqrt(preds * (1 - preds)), p_min, 1.0)
    return np.minimum.accumulate(target, axis=1)


# --- the randomized truncation estimator ---------------------------------------------------


def estimate_table(F: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """table[:, s] is the estimate for a run stopped after step s, for s = 0..T."""
    increments = np.diff(F, axis=1) / reach
    return np.concatenate([F[:, :1], F[:, :1] + np.cumsum(increments, axis=1)], axis=1)


def sample_stop_steps(reach: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Stop steps for uniform draws u, one per run: the run reaches step k iff u < reach_k."""
    return (reach > u[..., None]).sum(axis=-1)


def exact_moments(F: np.ndarray, cost: np.ndarray, reach: np.ndarray):
    """Per-run variance of the estimate and expected cost, over the stopping randomness."""
    sq_err = (F[:, -1:] - F) ** 2  # squared error of the prediction after k steps
    variance = ((sq_err[:, :-1] - sq_err[:, 1:]) / reach).sum(axis=1) - sq_err[:, 0]
    return variance, (cost * reach).sum(axis=1)
