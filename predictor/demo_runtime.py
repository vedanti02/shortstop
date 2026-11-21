"""End-to-end check of the run-time interface on real logged runs of an agent it never saw.

    python predictor/export.py runtime --exclude <agent>    # once: models without that agent
    python predictor/demo_runtime.py <agent> --seeds 20                     # both designs
    python predictor/demo_runtime.py <agent> --design standard              # or balanced
    python predictor/demo_runtime.py <agent> --ensemble 3  # the seed-averaged models (export --seeds 3)

Replays the agent's logged SWE-bench Verified runs through predictor/runtime.py exactly as a live
harness would: plan the batch, then for every task ask should_continue() before each step, feed
the logged step with observe(), and call finish() when the logged run ends. Each seed is one
complete evaluation, run under each design with the same seeds. Reports the scores and intervals
against the agent's true resolve rate, the share of the full-run tokens spent, and the full-run
interval for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import Runtime

from shortstop import load_trajectories

LABEL = {"standard": "independent stopping, mean +/- 1.96 sd / sqrt(n) (proven)",
         "balanced": "balanced stopping, design-based interval (not yet proven)"}


def evaluate_once(rt: Runtime, runs: dict, seed: int, design: str) -> dict:
    plan = rt.plan(list(runs), seed=seed, design=design)
    for task_id, steps in runs.items():
        run, i = plan.start(task_id), 0
        while run.should_continue():
            run.observe(steps[i])
            i += 1
            if i == len(steps):
                run.finish(steps[0].outcome)
                break
    return plan.result()


def report(design: str, results: list[dict], truth: float, full_tokens: int, full_half: float) -> None:
    scores = np.array([r["score"] for r in results])
    halves = np.array([r["half_width"] for r in results])
    covered = np.mean([r["interval"][0] <= truth <= r["interval"][1] for r in results])
    spent = np.mean([r["tokens"] for r in results]) / full_tokens
    print(f"  {design}: {LABEL[design]}")
    print(f"    {len(results)} evaluations: mean score {scores.mean():.3f} (sd {scores.std():.3f}), "
          f"interval contains the truth in {covered:.0%}")
    print(f"    mean half-width {halves.mean():.3f}; tokens spent {spent:.1%} of full runs; "
          f"runs started {np.mean([r['runs_started'] for r in results]):.0f}, "
          f"finished {np.mean([r['runs_finished'] for r in results]):.0f}")
    print(f"    compute for the reported width, relative to full runs: {(halves.mean() / full_half) ** 2 * spent:.1%}")
    for r in results[:3]:
        lo, hi = r["interval"]
        print(f"      e.g. score {r['score']:.3f}, 95% interval [{lo:.3f}, {hi:.3f}], tokens {r['tokens']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("agent")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--design", choices=["standard", "balanced", "both"], default="both")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="use the models averaged over this many seeds (export.py runtime --seeds N)")
    args = parser.parse_args()

    ens = f"_ens{args.ensemble}" if args.ensemble > 1 else ""
    model_dir = Path("data/predictor/runtime") / f"without_{args.agent}{ens}"
    if not model_dir.exists():
        raise SystemExit(f"run: python predictor/export.py runtime --exclude {args.agent}"
                         + (f" --seeds {args.ensemble}" if ens else ""))
    rt = Runtime.load(model_dir)
    assert args.agent not in rt.meta["training_agents"]
    runs = {task: steps for (task, _), steps in
            load_trajectories(f"data/swebench_verified/{args.agent}.jsonl.gz").items()}
    truth = float(np.mean([s[0].outcome for s in runs.values()]))
    full_tokens = sum(st.tokens_in + st.tokens_out for s in runs.values() for st in s)
    full_half = 1.96 * np.sqrt(truth * (1 - truth) / len(runs))
    print(f"{args.agent}: {len(runs)} tasks, true resolve rate {truth:.3f}, "
          f"half-width with full runs {full_half:.3f}, {len(rt.success_models)} success model(s)")
    for design in ["standard", "balanced"] if args.design == "both" else [args.design]:
        report(design, [evaluate_once(rt, runs, seed, design) for seed in range(args.seeds)],
               truth, full_tokens, full_half)


if __name__ == "__main__":
    main()
