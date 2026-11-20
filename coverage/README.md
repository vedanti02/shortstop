# coverage (Vedanti)

The finding, the estimator, and the confidence intervals. The derivation of the optimal
stopping policy and the argument for the intervals are in [THEORY.md](THEORY.md).

- Data: SWE-bench Verified submissions (many agents, same tasks)
- Stand-in: simple predictor (logistic regression on step count and test-run features)
- Done when: coverage and cost plots are made with the stand-in predictor

## Run it

From the repo root, with the environment active:

```bash
python coverage/download_swebench.py     # about 30 minutes, 13 GB streamed, 410 MB kept
python coverage/features.py              # about 4 minutes
python coverage/standin_predictor.py     # --split temporal or --split lofo for the others
python coverage/experiments.py           # same flag
python coverage/plots.py                 # same flag
python coverage/optimal_policy.py        # about 1 minute
python coverage/adaptive.py --floor 0.2 --prior 50   # about 15 minutes
python coverage/midrun_signal.py         # about 25 minutes
```

| File | What it does |
| --- | --- |
| `download_swebench.py` | Streams mini-swe-agent runs from the public SWE-bench bucket and writes them in the shared trajectory format, one `.jsonl.gz` per agent in `data/swebench_verified/` |
| `features.py` | One row per step prefix: step count, tokens so far, counts of test runs, edits, errors |
| `standin_predictor.py` | Logistic regression that scores each agent without having seen it |
| `stopping.py` | Deterministic stopping, the randomized policies, the estimator and its exact variance |
| `experiments.py` | Coverage and cost of every method, per agent |
| `optimal_policy.py` | Fits five stopping policies on the other agents and compares them |
| `midrun_signal.py` | Controlled check of the index policy with a predictor that is given a signal mid-run |
| `adaptive.py` | Corrects the predictor during the evaluation from the agent's own results |
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

Three splits for the predictor: `loao` trains on every other agent, `temporal` trains only
on agents submitted earlier (31 agents have at least 5 earlier ones), and `lofo` leaves out
every agent from the same vendor as the one being scored.

## Formulas

Proofs and assumptions are in [THEORY.md](THEORY.md). This is the short version, with the
function that implements each one.

**Notation.** One run has steps $k = 1, \dots, T$ and outcome $Y \in \{0, 1\}$. The
predictor outputs $f_k$ after step $k$, with $f_T = Y$. Step $k$ costs $c_k$ tokens.
$P_k$ is the probability that the run reaches step $k$; it is non-increasing in $k$, at
least $p_{\min}$, and computed only from what was seen before step $k$.

**Stop step** (`stopping.sample_stop_steps`). One uniform draw $U$ per run:

$$s = \max\{k : U < P_k\}$$

**Estimate for one run** (`stopping.estimate_table`):

$$\hat Y = f_0 + \sum_{k=1}^{s} \frac{f_k - f_{k-1}}{P_k}, \qquad E[\hat Y \mid \text{run}] = Y$$

**Variance and expected cost of one run** (`stopping.exact_moments`), with $e_k = Y - f_k$:

$$\operatorname{Var}(\hat Y \mid \text{run}) = \sum_{k=1}^{T} \frac{e_{k-1}^2 - e_k^2}{P_k} - e_0^2
= \sum_{k=1}^{T} \Big(\frac{1}{P_k} - \frac{1}{P_{k-1}}\Big) e_{k-1}^2,
\qquad E[\text{cost} \mid \text{run}] = \sum_{k=1}^{T} c_k P_k$$

**Score and interval for an agent** (`experiments.interval_stats`), over $n$ tasks:

$$\hat\theta = \frac{1}{n} \sum_{i=1}^{n} \hat Y_i, \qquad
\hat\theta \pm 1.96 \, \frac{\hat\sigma}{\sqrt n}, \qquad
\hat\sigma^2 = \frac{1}{n-1} \sum_{i=1}^{n} (\hat Y_i - \hat\theta)^2$$

**Compute for a fixed interval width, relative to full runs**
(`optimal_policy.relative_compute`). This is the number reported as "compute" in the
tables below; under 100% means cheaper than finishing every run.

$$\text{relative compute} =
\frac{\big(\operatorname{Var}(Y) + E[\operatorname{Var}(\hat Y \mid \text{run})]\big) \cdot E\big[\sum_k c_k P_k\big]}
{\operatorname{Var}(Y) \cdot E\big[\sum_k c_k\big]}$$

**Deterministic stopping, the baseline** (`stopping.deterministic_stop`). Stop at the
first $k$ with $f_k \le lo$ or $f_k \ge hi$ and score the run as $\mathbf 1\{f_k \ge hi\}$.
Its score is not unbiased, which is why its intervals fail.

**Stopping policies.** Every one below is followed by a running minimum,
$P_k = \min(P_{k-1}, \text{target}_k)$ with $P_0 = 1$, and the target is clipped to
$[p_{\min}, 1]$.

| Policy | Target for $P_k$ | Code |
| --- | --- | --- |
| Full runs | $1$ | `stopping.reach_full` |
| One-shot | $q$ | `stopping.reach_one_shot` |
| Randomized threshold | $q$ once some $f_j \notin (lo, hi)$, else $1$ | `stopping.reach_threshold` |
| Tracking | $\kappa \sqrt{f_{k-1}(1 - f_{k-1})}$ | `stopping.reach_tracking` |
| Step schedule | a value per step bin, minimizing $(V_0 + \sum_k a_k / P_k)(\sum_k c_k P_k)$ | `optimal_policy.fit_schedule` |
| Learned tracking | $\kappa \sqrt{E[e_{k-1}^2 \mid s] \, / \, E[\sum_{j \ge k} c_j \mid s]}$ | `optimal_policy.reach_learned` |
| Index (derived optimum) | $\kappa \sqrt{\nu_k(s)}$ | `optimal_policy.reach_index` |

Here $s$ is the state (step bin and prediction decile), expectations are measured on the
training agents, and $\kappa$ is tuned on them. In the step schedule,
$V_0 = \operatorname{Var}(Y) - E[e_0^2]$ and $a_k = E[e_{k-1}^2 - e_k^2]$.

**The index** (`optimal_policy.fit_index_table`). The best ratio of error removed to cost
paid, over every rule $\tau$ for how far to look ahead:

$$\nu_k(s) = \max_{\tau \ge k}
\frac{E\big[e_{k-1}^2 - e_\tau^2 \mid s\big]}{E\big[\sum_{j=k}^{\tau} c_j \mid s\big]}$$

It is computed by backward induction. For a price $\mu$ per token,

$$M_k(s; \mu) = a_k(s) - \mu \, \bar c_k(s) + E\big[\max(M_{k+1}(s'; \mu), 0) \mid s\big], \qquad M_{T+1} = 0,$$

with $a_k(s) = E[e_{k-1}^2 - e_k^2 \mid s]$ and $\bar c_k(s) = E[c_k \mid s]$, and
$\nu_k(s)$ is the price at which $M_k(s; \mu) = 0$. Learned tracking is the choice
$\tau = T$.

**Correcting the predictor during an evaluation** (`adaptive.solve_delta`). After each
batch, with $S$ the tasks seen so far, $m$ the prior weight (`--prior`) and
$\sigma(x) = 1 / (1 + e^{-x})$, find the shift $\delta$ that solves

$$\frac{1}{|S|} \sum_{i \in S} \sigma\big(\operatorname{logit} f_{i,0} + \delta\big)
= \frac{\sum_{i \in S} \hat Y_i + m \cdot \frac{1}{|S|} \sum_{i \in S} f_{i,0}}{|S| + m}$$

and use $\sigma(\operatorname{logit} f_{i,k} + \delta)$ as the prediction for the next
batch.

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

## Follow-up checks

**Smaller task sets.** With 100 tasks per replicate (`experiments.py --tasks 100`), the
randomized methods cover 94.2% to 94.7% on average and full runs cover 94.4%; the worst
single agent is 92.0% against 92.6% for full runs. Lowering the probability floor from
0.05 to 0.01 changes nothing measurable. The normal interval is about as good with
randomized stopping as without it.

**Is the baseline at EarlyEval's operating point?** Their abstract reports 13% to 26% of
steps removed, up to 44% of input tokens, and resolve rates shifting 1 to 2 points on
average. Deterministic 0.05/0.95 here saves 33% of tokens with a 1.1 point mean shift, so
it sits in the same region, and that is the row where coverage is 87.5%. Their classifier
(paired LightGBM models with reference-solution features) is stronger than this stand-in,
so this is a matched operating point, not a reproduction.

**Fitted policies** (`optimal_policy.py`, results in `results/policy_loao.csv`). Each
policy is fitted on the other 36 agents to minimize variance x expected cost, then scored
on the held-out agent. Numbers are compute for a fixed interval width relative to full
runs.

| Policy | Median | Worst agent | Agents worse than full runs | Compute used |
| --- | --- | --- | --- | --- |
| One-shot, q tuned | 55% | 4.9x | 3 | 5% |
| Tracking, kappa tuned | 53% | 5.3x | 2 | 7% |
| Step schedule, optimized | 58% | 4.3x | 2 | 7% |
| Learned tracking, kappa tuned | 52% | 3.9x | 2 | 7% |
| Index (the derived optimum), kappa tuned | 54% | 4.4x | 2 | 6% |

- All five land between 52% and 58%. With this predictor the best policy is close to
  "finish about 1 run in 15 and use the step-0 prediction for the rest": per-step
  stopping adds about 3 points over one-shot.
- One-shot and learned tracking tune to the lowest setting allowed by the 0.05 floor, so
  the floor, not the policy shape, is what binds.
- The gain from stopping inside a run therefore depends on a predictor whose estimate
  moves during the run. That is the learned predictor's job; this stand-in mostly knows
  task difficulty.
- **The derived optimum does not beat learned tracking here.** THEORY.md shows the optimal
  policy follows an index: the best ratio of expected error removed to expected cost over
  every look-ahead horizon. Learned tracking is the horizon "to the end of the run". On
  this data the two are within noise (54% against 52% held out; 55.7% against 55.4% when
  fitted and scored on all agents), because the largest drop in error comes when the
  tests reveal the outcome, so the end of the run is already the best horizon.
- **It does win when the predictor learns early in the run** (`midrun_signal.py`). With
  the same runs and costs but a predictor shown the outcome with 90% accuracy a tenth of
  the way in, the index policy needs 42% against 54% for learned tracking and is better
  for 33 of 37 agents. At a quarter of the way in it is 47% against 54%; at halfway they
  tie. With a 75%-accurate signal it is slightly worse (50% against 48%). The signal is
  artificial, so this shows when the rule can pay off, not that it will.
- No fitted policy fixes the two weak agents. Their cost comes from the step-0 prediction
  being far from the agent's true rate, which a policy fitted on other agents cannot see.

**Correcting the predictor during the evaluation** (`adaptive.py`). Tasks are evaluated
in batches of 25. Before each batch, all predictions are shifted on the logit scale so
that the step-0 predictions on tasks seen so far average to the current estimate of the
agent's score. The shift uses only earlier batches, so the estimate stays unbiased
(largest bias over 37 agents: 0.6 points). Tracking policy with kappa = 0.25:

| Floor on reach probability | Method | Median compute | Worst agent | Mean coverage | Worst coverage |
| --- | --- | --- | --- | --- | --- |
| 0.05 | static | 52% | 5.6x | 94.5% | 91.9% |
| 0.05 | adaptive | 56% | 1.2x | 92.4% | 86.2% |
| 0.1 | static | 57% | 5.2x | 94.7% | 92.6% |
| 0.1 | adaptive | 53% | 1.3x | 93.7% | 91.6% |
| 0.2 | static | 61% | 4.0x | 94.9% | 91.6% |
| 0.2 | adaptive | 56% | 1.2x | 94.3% | 91.6% |

- **It fixes the weak agents.** The worst case drops from 4 to 5.6 times full-run compute
  to about 1.2 times, and only one agent (9% resolve rate) is still above full runs.
- **It costs coverage at a low floor.** At floor 0.05 the intervals cover 92.4% on average
  and 86% for the weakest agent.
- **The cause is skew, not a wrong variance.** The estimated variance of the mean matches
  its true variance on average (`variance_ratio` between 0.91 and 1.07 for every agent),
  but the intervals miss more often on one side. Once the corrected predictor is
  confident, few runs are finished and a handful of heavily weighted ones make the
  estimate skewed. THEORY.md gives the rate: the coverage error scales with
  1 / sqrt(n x floor), so what matters is the number of finished runs.
- Rows above use 500 to 800 replicates, so each number carries a few percent of noise.

**Damped correction** (`--prior 50`, 1,500 replicates). The first batches give a noisy
estimate of the agent's score, and correcting toward it at full strength pushes the
predictor to extremes. Here the uncorrected predictor keeps the weight of 50 tasks when
the shift is computed.

| Floor | Method | Median compute | Worst agent | Mean coverage | Worst coverage |
| --- | --- | --- | --- | --- | --- |
| 0.1 | static | 56% | 4.2x | 94.6% | 93.3% |
| 0.1 | adaptive, damped | 51% | 1.4x | 94.0% | 93.0% |
| 0.2 | static | 61% | 3.9x | 94.9% | 94.0% |
| 0.2 | adaptive, damped | 55% | 1.3x | 94.6% | 93.5% |

- **Recommended setting: damped correction with floor 0.2.** 55% median compute, 1.3x in
  the worst case, 94.6% coverage against 94.8% for full runs, and no agent below 93.5%.
  Largest bias over 37 agents: 0.2 points.
- With damping the corrected method is cheaper than the static one for the typical agent
  as well as the weak ones.
- Hall's skewness-corrected interval was also tried on five agents and did not help
  consistently, so the interval stays the plain normal one.

**Leakage check** (`--split lofo`). Leave-one-agent-out keeps other models from the same
vendor in training (12 OpenAI agents, 8 Anthropic). Leaving the whole vendor out instead:

| | Leave one agent out | Leave one vendor out |
| --- | --- | --- |
| Deterministic 0.05/0.95, mean coverage | 87.5% | 85.3% |
| Deterministic 0.05/0.95, agents below 90% | 6 of 37 | 9 of 37 |
| Deterministic 0.05/0.95, mean abs. bias | 1.1 pts | 1.6 pts |
| Randomized methods, mean coverage | 94.8% to 95.0% | 94.7% to 95.0% |
| Randomized methods, worst single agent | 93.4% | 93.0% |
| Tracking kappa = 0.5, median compute | 55% | 55% |

The finding does not depend on same-vendor models being in training. The baseline gets
slightly worse without them and the randomized estimators do not change.

## What this does not show yet

- THEORY.md was revised after one outside read-through, but nobody has verified it line by
  line. The interval result is asymptotic; the finite-sample statement is a heuristic.
- The index policy is estimated on a coarse state (step, prediction decile). It beats
  learned tracking only with an artificial early signal, and not yet with a real
  predictor.
- One agent (9% resolve rate) still needs more compute than full runs under every setting.
- Policy tables are built from predictions whose models saw the held-out agent. The
  vendor-out split addresses the predictor, not this.
- 13 older submissions have estimated token counts.
