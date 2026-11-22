"""Full-run verdict for each tool change: paired comparison against the baseline.

    python tool_ab/analyze.py
    python tool_ab/analyze.py --arms qwen-search --n-boot 20000

For every task run in both arms, the paired difference in outcome is arm minus baseline.
Reports both resolve rates, the difference with a 95% paired bootstrap interval, the exact
McNemar p-value, and mean tokens per run. The verdict is "better" or "worse" when the
interval excludes zero, otherwise "no detectable difference". These full-run verdicts are
the reference the early-stopped A/B results are checked against at integration.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np

from shortstop import load_trajectories

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARMS = ["qwen-descriptions", "qwen-search", "qwen-compressed"]


def load_runs(path: Path) -> dict[str, dict]:
    """{task_id: {outcome, tokens, steps}} for one arm, keeping evaluated runs only."""
    runs = {}
    for (task_id, _agent), steps in load_trajectories(path).items():
        outcome = steps[0].outcome
        if outcome in (0, 1):
            tokens = sum(s.tokens_in + s.tokens_out for s in steps)
            runs[task_id] = {"outcome": outcome, "tokens": tokens, "steps": len(steps)}
    return runs


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(diffs, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    diffs = np.asarray(diffs, dtype=float)
    if np.all(diffs == diffs[0]):
        return float(diffs[0]), float(diffs[0])
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare(base: dict, arm: dict, n_boot: int = 10_000, seed: int = 0) -> dict:
    tasks = sorted(base.keys() & arm.keys())
    if not tasks:
        raise ValueError("The two arms share no evaluated tasks")
    y_base = np.array([base[t]["outcome"] for t in tasks])
    y_arm = np.array([arm[t]["outcome"] for t in tasks])
    diffs = y_arm - y_base
    lo, hi = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    arm_better, base_better = int((diffs == 1).sum()), int((diffs == -1).sum())
    base_tokens = float(np.mean([base[t]["tokens"] for t in tasks]))
    arm_tokens = float(np.mean([arm[t]["tokens"] for t in tasks]))
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"
    return {
        "n_tasks": len(tasks),
        "base_rate": float(y_base.mean()),
        "arm_rate": float(y_arm.mean()),
        "diff": float(diffs.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "arm_better": arm_better,
        "base_better": base_better,
        "p_value": mcnemar_exact(arm_better, base_better),
        "base_tokens": base_tokens,
        "arm_tokens": arm_tokens,
        "token_ratio": arm_tokens / base_tokens if base_tokens else float("nan"),
        "verdict": verdict,
    }


def to_markdown(rows: list[dict]) -> str:
    header = (
        "| Change | Tasks | Baseline | With change | Difference (95% CI) | Wins / losses "
        "| McNemar p | Tokens vs baseline | Verdict |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['n_tasks']} | {r['base_rate']:.1%} | {r['arm_rate']:.1%} "
            f"| {r['diff']:+.1%} ({r['ci_low']:+.1%} to {r['ci_high']:+.1%}) "
            f"| {r['arm_better']} / {r['base_better']} | {r['p_value']:.3f} "
            f"| {r['token_ratio']:.2f}x | {r['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "tool_ab")
    parser.add_argument("--baseline", default="qwen-baseline")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    base = load_runs(args.data / f"{args.baseline}.jsonl")
    rows = []
    for arm in args.arms:
        path = args.data / f"{arm}.jsonl"
        if not path.exists():
            print(f"skip {arm}: {path} not found")
            continue
        rows.append({"arm": arm, **compare(base, load_runs(path), n_boot=args.n_boot)})
    if not rows:
        raise SystemExit("Nothing to compare")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "full_run_verdicts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = to_markdown(rows)
    (args.out / "full_run_verdicts.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
