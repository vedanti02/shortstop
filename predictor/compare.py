"""Paired comparison of predictors on the same held-out agents.

    python predictor/compare.py lgbm_gold_loao lgbm_gold_loao_vw standin_loao --ref lgbm_gold_loao

Reads predictor/results/<name>_policy.csv for each name. For every policy, prints the median
and geometric mean of relative compute, and against the reference the per-agent log ratio
(negative = cheaper than the reference), how many agents are cheaper, and a two-sided
Wilcoxon signed-rank p-value over the 37 agents.

The p-value treats agents as independent and ignores training randomness: one training run
shifts every agent together, and two seeds of the same LightGBM model already give p < 0.01
with a geometric-mean spread of up to 3.4%. Compare single runs against that spread, not
against p.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).parent / "results"
POLICIES = ["one-shot", "tracking", "learned tracking", "index"]


def table(names: list[str], ref: str, policies=POLICIES) -> pd.DataFrame:
    data = {n: pd.read_csv(RESULTS / f"{n}_policy.csv") for n in names}
    rows = []
    for policy in policies:
        wide = pd.DataFrame({n: d[d["policy"] == policy].set_index("agent")["relative_compute"]
                             for n, d in data.items()})
        for n in names:
            row = {"policy": policy, "predictor": n, "median": wide[n].median(),
                   "geo_mean": float(np.exp(np.log(wide[n]).mean())), "worst": wide[n].max()}
            if n != ref:
                log_ratio = np.log(wide[n] / wide[ref])
                row.update(geo_ratio=float(np.exp(log_ratio.mean())),
                           cheaper=f"{int((log_ratio < 0).sum())}/{len(log_ratio)}",
                           p=wilcoxon(log_ratio).pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    names = args.names if args.ref in args.names else [args.ref, *args.names]
    out = table(names, args.ref)
    with pd.option_context("display.width", 140):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
