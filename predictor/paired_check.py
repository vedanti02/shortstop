"""Premise check for tool A/B tests: does a fully run reference arm sharpen the variant's prediction?

    python predictor/paired_check.py

In an A/B test of a tool change, the baseline setup has often been run to the end already.
Its outcome on each task is then known before the variant is evaluated, so using it as a
predictor input keeps the variant's estimate unbiased. The 37 SWE-bench agents contain
pairs that differ the way an A/B test would: the same model under two mini-swe-agent
versions or two reasoning efforts.

For each pair and direction, the variant's step-0 prediction combines its held-out
LightGBM prediction with the reference arm's outcome on the task, through a logistic
regression fitted on the other pairs only. Compute is the one-shot formula from
predictor/README.md, q + (1 - q) * Brier / Var(Y) with q = 0.05.

Output: predictor/results/paired_check.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

Q = 0.05
PAIRS = [  # (label, reference arm suffix, variant arm suffix)
    ("gpt-5.2: default vs high effort", "gpt-5.2-2025-12-11", "gpt-5.2-2025-12-11-high"),
    ("gpt-5.2-high: scaffold v1.17 vs v2.0", "gpt-5.2-2025-12-11-high", "gpt-5-2-high"),
    ("gpt-5-mini: scaffold v1.7 vs v2.0", "v1.7.0_gpt-5-mini", "v2.0.0_gpt-5-mini"),
    ("opus-4.5: scaffold v1.16 vs v2.0", "claude-opus-4-5-20251101", "claude-4-5-opus-high"),
    ("sonnet-4.5: scaffold v1.13 vs v2.0", "sonnet-4-5-20250929", "claude-4-5-sonnet-high"),
    ("deepseek-3.2: scaffold v1.17 vs v2.0", "deepseek-v3.2-reasoner", "deepseek-3-2-high"),
    ("glm: 4.6 vs 5", "glm-4.6", "glm-5-high"),
    ("devstral: small vs large", "devstral-small-2512", "devstral-2512"),
]


def main() -> None:
    p = pd.read_pickle("data/predictor/predictions_gold_loao.pkl").query("k == 0")
    Y = p.pivot(index="task_id", columns="agent", values="outcome")
    F0 = p.pivot(index="task_id", columns="agent", values="f").clip(1e-3, 1 - 1e-3)
    agents = sorted(Y.columns)

    def find(suffix: str) -> str:
        return next(a for a in agents if a.endswith(suffix))

    arms = [(label, find(r), find(v)) for label, r, v in PAIRS]
    both = arms + [(label + " (reversed)", v, r) for label, r, v in arms]

    def design(ref: str, var: str) -> np.ndarray:
        return np.column_stack([logit(F0[var]), Y[ref]])

    rows = []
    for i, (label, ref, var) in enumerate(both):
        others = [(r, v) for j, (_, r, v) in enumerate(both) if j % len(arms) != i % len(arms)]
        model = LogisticRegression().fit(np.vstack([design(r, v) for r, v in others]),
                                         np.concatenate([Y[v] for _, v in others]))
        y = Y[var].to_numpy()
        brier_lgbm = float(((F0[var] - y) ** 2).mean())
        brier_paired = float(((model.predict_proba(design(ref, var))[:, 1] - y) ** 2).mean())
        rows.append({"pair": label, "agreement": float((Y[ref] == Y[var]).mean()),
                     "rate_reference": float(Y[ref].mean()), "rate_variant": float(y.mean()),
                     "brier0_lgbm": brier_lgbm, "brier0_paired": brier_paired,
                     "compute_lgbm": Q + (1 - Q) * brier_lgbm / y.var(),
                     "compute_paired": Q + (1 - Q) * brier_paired / y.var()})
    d = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / "paired_check.csv"
    d.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(d.round(3).to_string(index=False))
    for name, sel in [("all pairs", d.index == d.index), ("agreement >= 0.87", d["agreement"] >= 0.87),
                      ("agreement < 0.87", d["agreement"] < 0.87)]:
        g = d[sel]
        print(f"{name:18s} ({len(g):2d}): one-shot compute {np.exp(np.log(g['compute_lgbm']).mean()):.3f}"
              f" -> {np.exp(np.log(g['compute_paired']).mean()):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
