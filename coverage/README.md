# coverage (Vedanti)

The finding, the estimator, and the confidence intervals.

- Data: SWE-bench Verified submissions (many agents, same tasks)
- Stand-in: simple predictor (logistic regression on step count and test-run features)
- Done when: coverage and cost plots are made with the stand-in predictor

## Run it

From the repo root, with the environment active:

```bash
python coverage/download_swebench.py     # about 30 minutes, 13 GB streamed, 410 MB kept
python coverage/features.py              # about 4 minutes
python coverage/standin_predictor.py     # add --split temporal for the second split
python coverage/experiments.py           # same flag
python coverage/plots.py                 # same flag
```

| File | What it does |
| --- | --- |
| `download_swebench.py` | Streams mini-swe-agent runs from the public SWE-bench bucket and writes them in the shared trajectory format, one `.jsonl.gz` per agent in `data/swebench_verified/` |
| `features.py` | One row per step prefix: step count, tokens so far, counts of test runs, edits, errors |
| `standin_predictor.py` | Logistic regression that scores each agent without having seen it |
| `stopping.py` | Deterministic stopping, the randomized policies, the estimator and its exact variance |
| `experiments.py` | Coverage and cost of every method, per agent |
| `plots.py` | The two-panel headline figure |

## Data

37 agents, 500 tasks each, 786,050 steps. Every agent's resolve rate was checked against
the leaderboard value.

- 17 submissions were left out: 15 have no per-task results in `swe-bench/experiments`,
  2 have results but no trajectories, and `20260226_mini-v2.0.0_gemini-3-pro-high` has
  a per-task file that marks every task unresolved although the leaderboard says 69.6%.
- Cost is `tokens_in + tokens_out` per step. The 13 submissions from before
  mini-swe-agent 1.13 did not log token usage, so theirs is estimated as characters / 4
  (`tokens_estimated` in `manifest.json`).
- A run with no model call is kept as one empty step so it still counts toward the score.

## Setup of the experiment

For each agent the target is its resolve rate on all 500 tasks. A replicate resamples the
tasks with replacement, applies a stopping method, and builds a 95% interval as
mean ± 1.96 sd / √n. Coverage is the share of 2,000 replicates whose interval contains
the target.

- **Deterministic stopping** (the baseline): stop at the first prediction outside
  (lo, hi) and score the run by that prediction.
- **Randomized, tracking**: the probability of reaching step k is kept at
  kappa · sqrt(f(1 − f)), clipped to [0.05, 1] and never increasing. Kappa trades compute
  for variance.
- **Randomized, one-shot**: finish each run with probability q, otherwise use the step-0
  prediction. This is prediction-powered inference with a fixed predictor.
- **Randomized threshold**: at the deterministic rule's trigger, continue with
  probability 0.2 instead of stopping.

Two splits for the predictor: `loao` trains on every other agent, `temporal` trains only
on agents submitted earlier (31 agents have at least 5 earlier ones).

## Results so far

Stand-in predictor, leave-one-agent-out, 37 agents. Full tables are in `results/`.

| Method | Compute used | Mean coverage | Worst agent | Mean abs. bias |
| --- | --- | --- | --- | --- |
| Full runs | 100% | 94.8% | 93.7% | 0.0 pts |
| Deterministic 0.02/0.98 | 95% | 94.8% | 93.3% | 0.1 pts |
| Deterministic 0.05/0.95 | 67% | 87.5% | 0.2% | 1.1 pts |
| Deterministic 0.1/0.9 | 36% | 62.0% | 0.0% | 4.8 pts |
| Deterministic 0.2/0.8 | 14% | 41.3% | 0.0% | 7.9 pts |
| Randomized, tracking (5 settings) | 14% to 77% | 94.9% to 95.0% | 94.0% | 0.1 pts |

- **The finding holds with this predictor.** Deterministic stopping keeps its coverage
  only while it saves almost nothing. At 0.05/0.95 the median bias is 0.6 points, 6 of 37
  agents fall below 90% coverage, and 8 of the 108 agent pairs within 3 points of each
  other swap order. At 0.1/0.9 it is 25 of 37 agents and 28 of 108 pairs.
- **The randomized estimators stay at 95%**: mean coverage is 94.8% to 95.0% for every randomized method, and no single agent falls below 93.4% on either split (full runs: 93.7%).
- **Compute for a fixed interval width**: tracking with kappa = 0.5 needs a median of 55%
  of full-run compute (63% on the temporal split). It is worse than full runs for 2 of
  37 agents, the two weakest (26% and 9% resolve rate, at 1.5x and 4.8x), because the
  predictor expects about 50% for any new agent.

![Headline figure](results/headline_loao.png)

## What this does not show yet

- The baseline is deterministic stopping with my stand-in predictor and my thresholds,
  not EarlyEval's classifier at its operating point. The coverage loss is real for this
  setup; how it compares to their reported 1-point shift is open.
- The tracking rule is a heuristic, not a derived optimum. The cost-optimal policy is
  still to do; `stopping.exact_moments` gives the variance and expected cost it needs:
  Var = Σ_k [(Y − f_{k−1})² − (Y − f_k)²] / P(reach k) − (Y − f_0)².
- Intervals are plain normal intervals. They cover here, but weights reach 1 / 0.05, so
  smaller task sets or a lower floor should be checked before claiming validity in general.
- Leave-one-agent-out keeps other versions of the same model in training.
