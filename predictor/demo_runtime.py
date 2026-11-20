"""End-to-end check of the run-time interface on real logged runs of an agent it never saw.

    python predictor/export.py runtime --exclude <agent>    # once: models without that agent
    python predictor/demo_runtime.py <agent> --seeds 20

Replays the agent's logged SWE-bench Verified runs through predictor/runtime.py exactly as a live
harness would: plan the batch, then for every task ask should_continue() before each step, feed
the logged step with observe(), and call finish() when the logged run ends. Each seed is one
complete evaluation. Reports the scores and intervals against the agent's true resolve rate, the
share of the full-run tokens spent, and the full-run interval for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import Runtime

from shortstop import load_trajectories


def evaluate_once(rt: Runtime, runs: dict, seed: int) -> dict:
    plan = rt.plan(list(runs), seed=seed)
    for task_id, steps in runs.items():
        run, i = plan.start(task_id), 0
        while run.should_continue():
            run.observe(steps[i])
            i += 1
            if i == len(steps):
                run.finish(steps[0].outcome)
                break
    return plan.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("agent")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    model_dir = Path("data/predictor/runtime") / f"without_{args.agent}"
    if not model_dir.exists():
        raise SystemExit(f"run: python predictor/export.py runtime --exclude {args.agent}")
    rt = Runtime.load(model_dir)
    assert args.agent not in rt.meta["training_agents"]
    runs = {task: steps for (task, _), steps in
            load_trajectories(f"data/swebench_verified/{args.agent}.jsonl.gz").items()}
    truth = float(np.mean([s[0].outcome for s in runs.values()]))
    full_tokens = sum(st.tokens_in + st.tokens_out for s in runs.values() for st in s)
    n = len(runs)
    full_half = 1.96 * np.sqrt(truth * (1 - truth) / n)

    results = [evaluate_once(rt, runs, seed) for seed in range(args.seeds)]
    scores = np.array([r["score"] for r in results])
    halves = np.array([r["half_width"] for r in results])
    covered = np.mean([r["interval"][0] <= truth <= r["interval"][1] for r in results])
    spent = np.mean([r["tokens"] for r in results]) / full_tokens
    print(f"{args.agent}: {n} tasks, true resolve rate {truth:.3f}")
    print(f"  {args.seeds} evaluations: mean score {scores.mean():.3f} (sd {scores.std():.3f}), "
          f"interval contains the truth in {covered:.0%}")
    print(f"  mean half-width {halves.mean():.3f} vs {full_half:.3f} with full runs; "
          f"tokens spent {spent:.1%} of full runs; runs started {np.mean([r['runs_started'] for r in results]):.0f}, "
          f"finished {np.mean([r['runs_finished'] for r in results]):.0f}")
    rel = (halves.mean() / full_half) ** 2 * spent
    print(f"  compute for the reported width, relative to full runs: {rel:.1%}")
    for r in results[:3]:
        lo, hi = r["interval"]
        print(f"    e.g. score {r['score']:.3f}, 95% interval [{lo:.3f}, {hi:.3f}], tokens {r['tokens']:,}")


if __name__ == "__main__":
    main()
