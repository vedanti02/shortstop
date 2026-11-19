"""Headline figure from the experiment results.

    python coverage/plots.py                   # reads results/results_loao.csv
    python coverage/plots.py --split temporal

Left: interval coverage against compute used. Right: compute needed to reach a fixed
interval width, relative to full runs. Output: coverage/results/headline_<split>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["loao", "temporal"], default="loao")
    args = parser.parse_args()
    folder = Path(__file__).parent / "results"
    results = pd.read_csv(folder / f"results_{args.split}.csv")
    n_agents = results["agent"].nunique()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

    # Left: coverage against compute, one line per family, faint dots per agent.
    families = [("deterministic", "Deterministic stopping", ORANGE),
                ("tracking", "Randomized (ours)", BLUE)]
    for prefix, label, color in families:
        rows = results[results["method"].str.startswith(prefix)
                       & ~results["method"].str.contains("floor")]
        left.scatter(100 * rows["cost_fraction"], 100 * rows["coverage"], s=9, color=color,
                     alpha=0.18, linewidths=0)
        mean = rows.groupby("method", sort=False)[["cost_fraction", "coverage"]].mean()
        mean = mean.sort_values("cost_fraction")
        left.plot(100 * mean["cost_fraction"], 100 * mean["coverage"], color=color, linewidth=2,
                  marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                  label=label)
    left.axhline(95, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    left.annotate("95% target", (100, 95), xytext=(0, 5), textcoords="offset points",
                  ha="right", color=MUTED, fontsize=9)
    left.set_xlim(0, 102)
    left.set_ylim(-3, 103)
    left.set_xlabel("Compute used, % of full runs", color=MUTED)
    left.set_ylabel("Coverage of 95% interval, %", color=MUTED)
    left.set_title("Coverage of the 95% interval",
                   loc="left", color=INK, fontsize=11, fontweight="bold")
    left.legend(frameon=False, loc="lower right", labelcolor=INK)
    style(left)

    # Right: compute for a fixed interval width, per randomized policy.
    rows = results[(results["kind"] == "randomized") & (results["method"] != "full runs")]
    rows = rows[~rows["method"].str.contains("randomized threshold|floor")]
    order = rows.groupby("method", sort=False)["relative_compute"].median().sort_values().index
    for i, method in enumerate(order):
        values = 100 * rows.loc[rows["method"] == method, "relative_compute"]
        right.scatter(values, [i] * len(values), s=14, color=BLUE, alpha=0.25, linewidths=0)
        right.scatter([values.median()], [i], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
                      zorder=3)
        right.annotate(f"{values.median():.0f}%", (values.median(), i), xytext=(0, 9),
                       textcoords="offset points", ha="center", color=INK, fontsize=9)
    right.axvline(100, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    right.annotate("full runs", (100, len(order) - 0.45), xytext=(-4, 0),
                   textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    right.set_yticks(range(len(order)))
    right.set_yticklabels(order, color=INK)
    right.set_ylim(-0.6, len(order) - 0.2)
    right.invert_yaxis()
    right.set_xlabel("Compute to reach the same interval width, % of full runs", color=MUTED)
    right.set_title("Randomized stopping: compute for a fixed width",
                    loc="left", color=INK, fontsize=11, fontweight="bold")
    style(right)

    fig.text(0.01, 0.01, f"{n_agents} mini-swe-agent submissions on SWE-bench Verified, "
             f"stand-in predictor ({args.split}). Faint dots are single agents; "
             "large dots are the mean (left) and median (right).", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = folder / f"headline_{args.split}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
