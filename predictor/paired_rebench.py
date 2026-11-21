"""A/B premise check on the SWE-rebench leaderboard: real one-change pairs, 5 runs per task.

    python predictor/paired_rebench.py

Like paired_check.py, but on nebius/SWE-rebench-leaderboard-trajectories, which has pairs of
agent configurations that differ by one change, every task run several times:

    effort     same model and scaffold, another reasoning effort
    fix        same model and scaffold, a configuration fix or another provider
    scaffold   same model, another scaffold (common scaffold, Claude Code, Codex, Junie)
    version    same scaffold, a neighbouring model version

The reference arm is assumed fully run, so its per-task solve rate (over its runs) is known
before the variant is evaluated and can be a predictor input without biasing the variant's
estimate. The variant's step-0 prediction combines its held-out LightGBM prediction with
that rate through a logistic regression fitted on the other pairs. "Luck floor" is the
variant's own run-to-run variance per task: no step-0 prediction can do better.

Output: predictor/results/rebench_lb_paired.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

Q = 0.05
P = "swelike-py__"
PAIRS = [  # (kind, reference, variant)
    ("effort", P + "gpt-55-2026-04-23-medium-tools-2026", P + "gpt-55-2026-04-23-xhigh-tools-2026"),
    ("effort", P + "claude-opus-4-8-tools-2026", P + "claude-opus-4-8-xhigh-tools-2026"),
    ("effort", "claude-code-cli__claude-opus-4-8__effort-low", "claude-code-cli__claude-opus-4-8__effort-medium"),
    ("effort", "claude-code-cli__claude-opus-4-8__effort-medium", "claude-code-cli__claude-opus-4-8__effort-high"),
    ("effort", "claude-code-cli__claude-opus-4-8__effort-high", "claude-code-cli__claude-opus-4-8__effort-xhigh"),
    ("fix", P + "glm-52", P + "glm-52-fix"),
    ("fix", P + "deepseek-v4-pro-tools-no-demo", P + "ds-v4-pro-or"),
    ("scaffold", P + "claude-opus-4-7-tools-2026", "claude-code-cli__claude-opus-4-7"),
    ("scaffold", P + "claude-opus-4-8-tools-2026", "junie__claude-opus-4-8"),
    ("scaffold", P + "gpt-55-2026-04-23-medium-tools-2026", "codex-cli__gpt-5.5"),
    ("version", P + "claude-opus-4-7-tools-2026", P + "claude-opus-4-8-tools-2026"),
    ("version", P + "glm-51-tools-no-demo", P + "glm-52"),
    ("version", P + "qwen-36-27b", P + "qwen-36-moe-35b"),
]


def main() -> None:
    p = pd.read_pickle("data/predictor/rebench_lb/predictions_gold_loao.pkl").query("k == 0").copy()
    p["task"] = p["task_id"].str.split("@").str[0]
    rate = p.pivot_table(index="task", columns="agent", values="outcome", aggfunc="mean")
    runs = {a: g for a, g in p.groupby("agent")}
    both = PAIRS + [(k, v, r) for k, r, v in PAIRS]

    def design(ref: str, var: str) -> np.ndarray:
        g = runs[var]
        ref_rate = g["task"].map(rate[ref]).clip(0.02, 0.98)
        return np.column_stack([logit(g["f"].clip(1e-3, 1 - 1e-3)), logit(ref_rate)])

    rows = []
    for i, (kind, ref, var) in enumerate(both):
        others = [(r, v) for j, (_, r, v) in enumerate(both) if j % len(PAIRS) != i % len(PAIRS)]
        model = LogisticRegression().fit(np.vstack([design(r, v) for r, v in others]),
                                         np.concatenate([runs[v]["outcome"] for _, v in others]))
        g = runs[var]
        y = g["outcome"].to_numpy()
        per_task = g.groupby("task")["outcome"].agg(["size", "mean"])
        luck = float(np.average(per_task["size"] / (per_task["size"] - 1).clip(lower=1)
                                * per_task["mean"] * (1 - per_task["mean"]), weights=per_task["size"]))
        b_lgbm = float(((g["f"] - y) ** 2).mean())
        b_pair = float(((model.predict_proba(design(ref, var))[:, 1] - y) ** 2).mean())
        diff = (rate[ref] - rate[var]).abs().mean()
        rows.append({"kind": kind, "reference": ref.replace(P, ""), "variant": var.replace(P, ""),
                     "rate_reference": rate[ref].mean(), "rate_variant": y.mean(),
                     "mean_task_rate_gap": diff, "brier0_lgbm": b_lgbm, "brier0_paired": b_pair,
                     "luck_floor": luck,
                     "compute_lgbm": Q + (1 - Q) * b_lgbm / y.var(),
                     "compute_paired": Q + (1 - Q) * b_pair / y.var(),
                     "compute_floor": Q + (1 - Q) * luck / y.var()})
    d = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / "rebench_lb_paired.csv"
    d.to_csv(out, index=False)
    geo = lambda s: float(np.exp(np.log(s).mean()))
    summary = d.groupby("kind", sort=False).agg(
        pairs=("variant", "size"), task_rate_gap=("mean_task_rate_gap", "mean"),
        brier_lgbm=("brier0_lgbm", "mean"), brier_paired=("brier0_paired", "mean"),
        brier_luck=("luck_floor", "mean"), compute_lgbm=("compute_lgbm", geo),
        compute_paired=("compute_paired", geo), compute_floor=("compute_floor", geo))
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string())
        print(f"all ({len(d)}): compute {geo(d['compute_lgbm']):.3f} -> {geo(d['compute_paired']):.3f}"
              f" (luck floor {geo(d['compute_floor']):.3f}); paired better for "
              f"{int((d['brier0_paired'] < d['brier0_lgbm']).sum())}/{len(d)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
