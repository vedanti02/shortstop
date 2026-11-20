"""Oracle bound: shift each agent's predictions so they average to its true resolve rate.

    python predictor/oracle_shift.py data/predictor/predictions_gold_loao.pkl

Not a method (it uses the answer). It shows how much compute a correct agent-level
calibration would save; coverage/adaptive.py recovers part of it legitimately from the
agent's own finished runs. Every prediction of an agent is moved by the same amount on the
logit scale, chosen so that the mean step-0 prediction equals the agent's resolve rate.
Output: the same file with suffix _oracleshift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit


def shift(predictions: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in predictions.groupby("agent", sort=True):
        f0 = g.loc[g["k"] == 0, "f"].clip(1e-4, 1 - 1e-4)
        rate = g.loc[g["k"] == 0, "outcome"].mean()
        delta = brentq(lambda d, f0=f0, rate=rate: expit(logit(f0) + d).mean() - rate, -12, 12)
        parts.append(g.assign(f=expit(logit(g["f"].clip(1e-4, 1 - 1e-4)) + delta)))
    return pd.concat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    out = args.predictions.with_name(args.predictions.stem + "_oracleshift.pkl")
    shift(pd.read_pickle(args.predictions)).to_pickle(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
