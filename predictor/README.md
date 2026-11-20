# predictor (Deepika)

The step predictor: LightGBM baseline and the learned predictor, scored on agents they never
saw. It reuses the features, stopping policies and metric in `coverage/` (Vedanti's slice,
merged in #1), so every number here is directly comparable with `coverage/README.md`.

- Data: SWE-bench Verified mini-swe-agent runs (37 agents, from `coverage/`), plus Nebius
  and CoderForge OpenHands runs
- Done when: variance reduction over LightGBM is shown on held-out agents

## Run it

From the repo root, with the environment active. On macOS, `brew install libomp` is needed
for LightGBM, and python.org Python needs `export SSL_CERT_FILE=$(python -m certifi)` for
the downloads.

```bash
python coverage/download_swebench.py          # Vedanti's download, about 15 minutes
python predictor/step_features.py                  # about 1 minute with 8 workers
python predictor/train_lgbm.py --features gold --split loao     # about 11 minutes
python predictor/evaluate.py data/predictor/predictions_gold_loao.pkl --name lgbm_gold_loao
python predictor/convert_openhands.py nebius  # about 8 minutes, 1.6 GB written
python predictor/step_features.py --data data/openhands \
    --meta data/meta/swebench_verified.parquet data/meta/swerebench_gold.parquet \
    --out data/openhands/features_rich.pkl --workers 1      # about 10 minutes
```

`data/meta/swebench_verified.parquet` (gold patches and tests for Verified) is read from
`princeton-nlp/SWE-bench_Verified` on Hugging Face; see "Data" below.

| File | What it does |
| --- | --- |
| `step_features.py` | One row per step prefix: Vedanti's counts (identical values), plus *process* features (last test result, steps since the last edit, reproduction scripts, repeated commands, format errors) and *gold* features (has the agent viewed or edited the files the reference patch touches, has it run the tests that must pass) |
| `train_lgbm.py` | LightGBM, one model per held-out agent; feature groups `base` / `process` / `gold`, splits `loao` / `lofo` / `insample` |
| `evaluate.py` | Scores any predictions file with Vedanti's five stopping policies, each tuned on the other agents, plus how much the prediction improves through the run |
| `variance_weights.py` | Row weights that make the training loss equal to the estimator's variance (THEORY.md, Lemma 2) |
| `convert_openhands.py` | Nebius / CoderForge OpenHands runs into the shared step format |
| `compare.py` | Paired comparison of predictors over the same held-out agents |
| `oracle_shift.py` | Bound: each agent's predictions shifted to average its true rate |
| `adaptive_eval.py` | Vedanti's in-evaluation correction (`coverage/adaptive.py`, unchanged) run on any predictions |
| `train_ability.py` | Ability-aware predictor: the agent's resolve rate is an input, on top of a uniform logit shift |
| `ability_eval.py` | Simulated evaluations with the ability-aware predictor, the rate estimated from earlier batches |
| `paired_check.py` | Premise check for tool A/B tests: a fully run reference arm as a predictor input |
| `deep_dive.py` | Why the learned predictor did not beat LightGBM: cost decomposition, luck, timing, sensitivity |
| `probe_estimate.py` | Offline estimate of mid-run test probes (hidden tests run after the decisive edit) |
| `convert_rebench_lb.py` | SWE-rebench leaderboard trajectories (36 configurations, 5 runs per task) into the shared format |
| `paired_rebench.py` | A/B reference-arm check on the leaderboard's real one-change pairs |
| `allocation_check.py` | Premise check: what per-task, cost-aware allocation is worth (one-shot policies) |
| `cost_aware.py` | Learned remaining cost and error per held-out agent, fed to the stopping policy |
| `cost_models.py` | Which loss for the remaining-cost model (log squared error, Tweedie, gamma, smearing) |
| `balanced.py` | Balanced (stratified) stopping uniforms and a design-based interval, with either policy |
| `export.py` | Predictions for coverage/ (`coverage`) and deployable models for a live harness (`runtime`) |
| `runtime.py` | Run-time interface: predictions, cost-aware stopping, balanced plan, estimate and interval |
| `demo_runtime.py` | Replays a held-out agent's logged runs through `runtime.py` as a live harness would |

## The metric, and a formula that explains most of it

The number reported throughout is Vedanti's *relative compute*: compute needed for a given
interval width, relative to finishing every run (under 100% is cheaper).

For the one-shot policy (finish a run with probability q, otherwise use the step-0
prediction) it reduces exactly to

$$\text{relative compute} = q + (1 - q)\,\frac{E[(Y - f_0)^2]}{\operatorname{Var}(Y)}$$

checked against `evaluate.py` to 5e-8 on all 37 agents. With q at the 0.05 floor, **the
cost is set by the step-0 Brier score divided by the variance of the outcomes.** Stopping
inside a run can only improve on this when the prediction gets better early, while little
has been spent. This is why the two weakest agents are expensive: qwen2.5-coder-32b
resolves 9% of tasks (Var(Y) = 0.082) while the predictor expects about 60% for every agent
(step-0 Brier 0.455), so it costs 5.3 times full runs.

## How large a difference is real

Retraining gold LightGBM with three seeds (bagging, validation agents and the cross-fit
halves all change) moves the geometric-mean compute by up to 3.4%:

| Policy | Seed 0 | Seed 1 | Seed 2 | Spread |
| --- | --- | --- | --- | --- |
| One-shot | 56.6% | 57.1% | 56.7% | 0.9% |
| Tracking | 53.0% | 52.2% | 53.9% | 3.2% |
| Learned tracking | 52.6% | 52.0% | 53.0% | 1.9% |
| Index | 54.1% | 53.3% | 55.1% | 3.4% |

The paired Wilcoxon test in `compare.py` treats the 37 agents as independent, but one
training run shifts all of them together: seed 2 against seed 0 gives p = 0.001 to 0.004 for
the same model. **Those p-values ignore training randomness, so differences under about 3%
between single training runs are noise**, whatever p says. Below, such differences are
reported as "no difference".

## Checks before building the learned predictor

### Check 0: the pipeline reproduces Vedanti's numbers (passed)

`evaluate.py` on Vedanti's stand-in predictions gives the same median, geometric mean and
worst case as `coverage/results/policy_loao.csv` for all five policies (largest per-agent
difference 0.001, identical tuned settings). The base features from `step_features.py` match
`coverage/features.py` on all 786,050 rows.

### Check 5: what the external datasets contain (passed, with caveats)

| | Nebius | CoderForge |
| --- | --- | --- |
| Runs | 67,074 | 258,134 (+ a 155K pass-only subset that duplicates them) |
| Tasks | 6,306 (10.6 runs each) | 51,201 across SWE_Rebench / SWE_Smith / R2E_Gym |
| Resolve rate | 48% | 58 to 63%; 900 runs have reward -1 (errored) and are dropped |
| Model and scaffold | Qwen3-Coder-480B, OpenHands 0.54 | Qwen3-Coder-480B, OpenHands-style tools |
| Token counts | not logged; estimated as characters / 4 | not logged (a 50 GB tokenized copy exists) |
| Overlap with Verified | no task, no repository | no task; 799 tasks from Verified repositories, which the converter drops |
| Licence, access | CC-BY-4.0, not gated | per-row licence field, not gated |

- **Both are a single model.** They cannot show anything about held-out *agents*, so the
  held-out evaluation uses Vedanti's 37 agents; the OpenHands data is for extra training and
  a transfer test.
- 4,044 tasks appear in both datasets, so any split must keep a task on one side.
- **Both are converted and featurized** (`convert_openhands.py`, then `step_features.py`):

  | | Runs | Steps | Resolved | Dropped |
  | --- | --- | --- | --- | --- |
  | Nebius | 67,074 | 4.3M | 48% | none (exact Verified-repository filter; 137 tasks from related repositories such as Django plugins should be excluded when training for Verified) |
  | CoderForge (3 splits) | 249,462 | 14.2M | 60.5% | 900 errored, 7,772 from Verified or related repositories (11 shards entirely) |

  Gold patches come from `nebius/SWE-rebench` (21,336 tasks) and `SWE-bench/SWE-smith`
  (59,136 tasks); the R2E-Gym tasks have none, so their gold features are missing.
- Several runs per task make both useful for measuring luck (see the deep dive): run-to-run
  randomness of one agent on one task accounts for 15 to 33% of the variation in outcomes,
  depending on the task source.

### Other datasets (survey, Nov 20)

What the project lacks most is many agents on the same tasks (held-out-agent tests), repeated
runs (luck), and pairs that differ by one change (A/B). All of these include failing runs;
existence, size, row counts and licence were checked against the Hugging Face API.

| Dataset | Runs / tasks | Agents | Repeats / A-B | Tokens | Licence, size | Best use |
| --- | --- | --- | --- | --- | --- | --- |
| `nebius/SWE-rebench-leaderboard-trajectories` | 27,348 / 221 | 30 models, 52 configs (common scaffold plus Claude Code, Codex, Cursor) | 5 runs per task; reasoning-effort panels | per run, with cost | CC-BY-4.0, 0.67 GB, one schema | Held-out agents on a second, fresh benchmark; luck; A/B |
| `JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro-Trajectories` | 25,200 / 700 | 4 models, mini-swe-agent | 3 prompt conditions × 3 runs | per call | "other" (terms to check), 6.3 GB | A/B with real prompt changes, the closest proxy for tool changes |
| `nvidia/Open-SWE-Traces` | 1.14M | 6 open models × 5 scaffolds | ~2 runs per task in some configs; 3 reasoning efforts | none | CC-BY-4.0, 98 GB | Training volume; scaffold A/B pairs (stream, do not store) |
| `harborframework/terminal-bench-2-leaderboard` | ~75 agents × 89 tasks × ≥5 trials | ~75 agent+model combos | ≥5 trials per task | per step for some agents | Apache-2.0, ≥4.4 GB, >100K files | Held-out agents and luck outside coding; formats vary by agent |
| SWE-bench Verified, non-mini submissions (`swe-bench-submissions` bucket, `verified/`) | ~120 systems × 500 | ~120 scaffold+model systems | 1 run | varies | none stated | More held-out agents on our 500 tasks; free-form formats |

Smaller or weaker fits: `nebius/Longtail-SWE-trajectories` (6,666 runs, 11 models, 3 runs
each), `melissapan/swe-bench-lite-agent-traces-v14` (1,890 runs, 7 models × 3 CLI agents),
tau2-bench submissions, Toolathlon, HAL traces (encrypted), Multi-SWE-bench,
`nebius/SWE-agent-trajectories` (Llama only). EarlyEval and "Doomed from the Start" have not
released data; SWE-Dev, R2E-Gym SFT and SWE-Factory trajectories are success-only.

### A leak found and fixed: leave-one-out task solve rate

The first gold-feature model was *worse* than the process model (step-0 Brier 0.152 against
0.129, median compute 63% against 49%). The cause was target leakage. For a training run the
stand-in's `task_rate` leaves that run's own outcome out, so it is slightly lower exactly when
the run passed. The gold group adds task-level constants (patch size, number of tests) that
let LightGBM identify the task, and from there read the run's own outcome off the rate. On a
held-out agent the rate is not shifted and the learned rule misfires.

On two agents checked directly the step-0 Brier went 0.513 → 0.445 (weakest agent) and
0.115 → 0.096 (a strong one) with the fix: `task_rate` for training rows is now
*cross-fitted*, measured on the other half of the training agents, so it never depends on
the row's own outcome. Vedanti's stand-in is not affected in practice, because its features
cannot identify a task, but the same encoding should not be combined with task-level
features anywhere in the project.

### Check 2: is there early signal in real runs? (weak; richer features do not lower cost)

How much better the prediction is at a point in the run than at step 0 (Brier skill), and
compute for a fixed interval width. Held-out agents (leave one agent out), mean or median
over 37. All LightGBM rows use the cross-fitted task rate.

| Predictor | Step-0 Brier | Step-0 AUC | Skill at 10% | 25% | 50% | 75% | Tracking, median | Tracking, geo. mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stand-in (Vedanti) | 0.130 | 0.921 | 5.7% | 6.0% | 4.2% | 4.8% | 52.5% | 55.9% |
| LightGBM, base features | 0.126 | 0.918 | 0.6% | 1.8% | 2.9% | 3.8% | 47.7% | 53.5% |
| LightGBM, + process | 0.126 | 0.918 | 2.4% | 4.6% | 6.4% | 6.9% | 46.9% | 52.9% |
| LightGBM, + gold | 0.127 | 0.917 | 1.6% | 4.9% | 9.6% | 10.5% | 47.7% | 53.0% |

Paired over the 37 agents (`compare.py`), against LightGBM with gold features:

| Policy | Stand-in | Base | + process |
| --- | --- | --- | --- |
| Tracking | 1.055x, p = 0.004 | 1.011x, p = 0.11 | 0.998x, p = 0.93 |
| Learned tracking | 1.059x, p = 0.002 | 1.016x, p = 0.03 | 1.006x, p = 0.26 |
| Index | 1.051x, p = 0.015 | 1.017x, p = 0.18 | 1.003x, p = 0.81 |

(Geometric mean of the per-agent compute ratio; above 1 is costlier than gold LightGBM.
Base and process are within the 3.4% seed spread, so they count as no difference.)

- **LightGBM is about 5% cheaper than the stand-in**, whatever the features; this is larger
  than the seed spread.
- **Richer features improve the prediction during the run but not the cost.** Gold features
  triple the improvement by halfway (9.6% against 2.9%), yet compute is the same as with
  base features. The improvement arrives after most of the cost is spent, while the cost is
  set mostly by the step-0 error, which no in-run feature can change.
- **The early signal is far below what Vedanti's controlled experiment needed.** In
  `coverage/midrun_signal.py` a signal 90% accurate a quarter of the way in gained 7 points
  and a 75%-accurate one gained nothing. Here the best prediction a quarter of the way in is
  5% better than at step 0.

### Agent calibration is the larger lever (oracle check)

Every predictor gives a mean step-0 prediction of about 0.61 for every agent, from 9% to
77% true resolve rate: at step 0 nothing about the agent is known. `oracle_shift.py` moves
each agent's predictions on the logit scale so they average to its true rate. It is an
oracle, not a method (it uses the answer), and shows how much a correct agent-level
calibration is worth. Gold LightGBM, held-out agents:

| Policy | As trained, median | Oracle shift, median | Paired ratio | Agents cheaper | Worst agent |
| --- | --- | --- | --- | --- | --- |
| One-shot | 52.5% | 44.5% | 0.840x | 36/37 | 5.2x → 0.88x |
| Tracking | 47.7% | 44.4% | 0.844x | 31/37 | 4.7x → 0.73x |
| Learned tracking | 50.1% | 43.8% | 0.849x | 28/37 | 3.7x → 0.75x |
| Index | 50.0% | 44.7% | 0.861x | 33/37 | 3.4x → 0.70x |

- **This is the largest effect measured here**: 14 to 16% less compute in geometric mean
  (p < 0.001 for every policy), and no agent left above full-run cost. Richer in-run
  features gave 0 to 2%.
- Coverage pays for it at the tail: mean 94.1 to 94.3%, worst agent 85.8 to 90.4%
  (as trained: 94.2 to 94.9% and 90.6 to 92.8%; 500 replicates, so about ±1 point per
  agent). A confident predictor finishes fewer runs, the same skew Vedanti found with the
  in-evaluation correction.
- The achievable part is what `coverage/adaptive.py` recovers from the agent's own finished
  runs. A predictor can add to it only if something cheap and early reveals the agent's
  strength; Check 4 measures how much knowing the agent is worth.

**Early steps do not reveal an agent's strength well enough to use.** For each agent, the
mean of each feature after its first K steps (over its 500 runs) was used to predict its
resolve rate, leave one agent out, with the two most correlated features picked inside each
fold:

| Steps seen | Share of a run's tokens (median) | Error in the agent's rate (RMSE) | Guessing the mean |
| --- | --- | --- | --- |
| 3 | 3.1% | 0.145 | 0.148 |
| 5 | 6.9% | 0.228 | 0.148 |
| 10 | 19.4% | 0.137 | 0.148 |

Correlations exist (after 10 steps, the share of gold files viewed correlates 0.74 with the
agent's rate), but with 37 agents and one far outlier (9%) a model fitted on 36 does not
predict the 37th. The agent-level correction has to come from the agent's own outcomes
during the evaluation.

### Check 3: does training on the variance objective help? (no)

`variance_weights.py` weights each training row by how much the tuned tracking policy
(kappa 0.25) cuts the reach probability right after it, so the weighted loss is the
estimator's variance. 74% of the weight falls on the step-0 rows, because the tuned policy
stops most runs at the start. LightGBM with gold features, retrained with these weights:

| Policy | Unweighted, median | Variance-weighted, median | Paired ratio | Agents cheaper | p |
| --- | --- | --- | --- | --- | --- |
| One-shot | 52.5% | 52.9% | 1.005x | 14/37 | 0.11 |
| Tracking | 47.7% | 48.9% | 1.030x | 9/37 | 0.003 |
| Learned tracking | 50.1% | 50.5% | 1.024x | 8/37 | < 0.001 |
| Index | 50.0% | 51.6% | 1.023x | 12/37 | 0.007 |

The weighted model is no better under any policy. It is 2 to 3% costlier under the per-step
policies, which is within the 3.4% seed spread, so the result is "no gain" rather than a
loss. This is what the theory note below predicts: the variance-optimal predictor is the
calibrated success probability, which the unweighted model already targets, and the weights
only take data away from the later steps.

### Check 4: how much is lost to never having seen the agent?

**First version, too generous** (`--split insample`): the scored agent is in the training
data, including the very runs it is scored on.

| Policy | Held out, median | In sample, median | Paired ratio | Agents cheaper | Worst agent |
| --- | --- | --- | --- | --- | --- |
| Tracking | 47.7% | 43.8% | 0.897x | 35/37 | 4.7x → 2.7x |
| Learned tracking | 50.1% | 42.4% | 0.881x | 34/37 | 3.7x → 2.0x |
| Index | 50.0% | 45.0% | 0.868x | 29/37 | 3.4x → 1.6x |

The in-run improvement triples (31% by three quarters against 10%) while step-0 Brier barely
moves (0.124 against 0.127). Token counts nearly fingerprint a run, so part of this is the
model memorizing runs it was scored on, not knowing the agent.

**Fair version** (`--split agentseen`, `agentseen_id`): the agent is seen on half of its
tasks and scored on the other half (two folds), and `task_rate` never uses its own
outcomes; `_id` also gives the model the agent's name.

| Gold LightGBM | Held out | Agent seen, no name | Agent seen, with name | Oracle shift |
| --- | --- | --- | --- | --- |
| Tracking, median | 47.7% | 48.3% | 44.9% | 44.4% |
| Tracking, geometric mean | 53.0% | 51.5% | 46.0% | 44.7% |
| Learned tracking, geometric mean | 52.6% | 51.0% | 45.6% | 44.7% |
| Index, geometric mean | 54.1% | 52.4% | 46.2% | 46.6% |
| Worst agent (tracking) | 4.7x | 2.7x | 0.89x | 0.73x |
| Coverage, mean / worst agent (tracking) | 94.6% / 91.8% | 95.1% / 93.6% | 95.0% / 92.8% | 94.1% / 88.2% |
| Step-0 Brier | 0.127 | 0.122 | 0.103 | 0.099 |
| Brier at 75% of the run | 0.107 | 0.101 | 0.098 | 0.102 |

- **Knowing the agent is worth 13 to 15% of compute** (geometric mean, against a 3.4% seed
  spread), removes every agent from above full-run cost, and keeps coverage at 95%.
- **Almost all of it is the step-0 calibration**: knowing the agent lowers step-0 Brier from
  0.127 to 0.103, about what the one-number oracle shift gives (0.099), while later in the
  run it adds little (0.107 → 0.098 at three quarters). The in-sample model's large in-run
  gains (0.080) were memorization of the scored runs.
- So the held-out gap is one number per agent, its strength, not agent-specific patterns
  inside runs.
- **The model has to be told which agent it is.** Seeing half of the agent's runs without
  its name gives about 3% (within the seed spread), though the worst agent drops from 4.7x
  to 2.7x. The gain needs an explicit per-agent parameter, which is what an agent-adaptive
  predictor would estimate during the evaluation.

## Gate decision (Nov 20)

| Check | Question | Answer |
| --- | --- | --- |
| 0 | Does the pipeline reproduce Vedanti's numbers? | Yes, exactly |
| 5 | Are Nebius and CoderForge usable? | Yes, but each is one model; no overlap with Verified; tokens estimated |
| 2 | Is there early signal in real runs? | Weak: 5% better a quarter of the way in, 10% by halfway; richer features do not lower cost |
| 3 | Does training on the variance objective help? | No, as the theory note predicts |
| 4 | How much is lost by never having seen the agent? | 13 to 15% of compute and the whole worst-case problem, almost all of it step-0 calibration |
| | Can early steps reveal the agent's strength? | No, not with 37 agents |

This is the outcome where the early signal is weak and the objective does not matter. A
predictor trained on the variance objective, as written in the proposal, would not beat
LightGBM: the theory says the two aim at the same function, and the data agree. The lever
that exists is the agent's strength, learned from its own outcomes during the evaluation,
which keeps the estimate unbiased by Theorem 2b in `coverage/THEORY.md`.

## Agent-adaptive predictor (Nov 20)

All rows in this section use the simulation of `coverage/adaptive.py`: tasks resampled with
replacement, batches of 25, the tracking policy with kappa 0.25, 1,500 replicates. Numbers
are the geometric mean of relative compute over the 37 held-out agents; coverage is mean /
worst agent. `adaptive_eval.py` reproduces Vedanti's committed results for the stand-in
(adaptive, floor 0.2, damping 50: median 55.1% against 54.9%).

### Vedanti's correction on LightGBM predictions

| Floor | Damping | Static | + correction | Agent known (bound) | Oracle shift (bound) |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0 | 51.9% (94.6 / 92.7) | 53.4% (92.5 / 88.3) | 45.5% (94.7 / 93.2) | 44.0% (94.1 / 86.9) |
| 0.1 | 50 | 58.2% (94.6 / 93.2) | 52.6% (93.8 / 92.5) | 50.1% (94.7 / 93.5) | 49.6% (94.6 / 91.7) |
| 0.2 | 50 | 64.2% (94.8 / 93.0) | 57.9% (94.4 / 92.6) | 57.1% (94.8 / 93.2) | 55.9% (94.7 / 92.9) |

Damping at floor 0.05 (correction only):

| Damping (tasks) | 0 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- |
| Compute | 53.4% | 47.5% | 46.8% | 47.0% | 47.8% |
| Coverage | 92.5 / 88.3 | 92.9 / 91.3 | 93.3 / 91.2 | 93.7 / 90.9 | 94.1 / 92.2 |

- At floors 0.1 and 0.2 the correction closes 65 to 76% of the gap to the oracle and fixes
  the worst agent.
- **The floor itself is expensive**: the oracle at floor 0.2 (55.9%) costs more than no
  correction at floor 0.05 (51.9%). The prize is a correction that works at floor 0.05.
- At floor 0.05 the right damping is about 100 tasks (a noise-to-spread ratio: each task's
  estimate has variance about 1.3, agents' rates vary by about 0.13, and 1.3 / 0.13² ≈ 80),
  which closes 80% of the gap to the agent-known bound. Coverage drops to 93.3%, and the
  misses are one-sided: the interval lies entirely above the truth twice as often as below
  (4.6% against 2.2%), on weak, middle and strong agents alike.

### Why a uniform shift skews the estimate

Exact skewness of a task's estimate at floor 0.05 (no simulation), median over agents:
static −2.7, oracle uniform shift −4.4 (+11.6 for the weakest agent), agent-known model −2.7,
although the agent-known model has the most runs at the floor (24% against 11% static).
Skew comes from confident predictions that turn out wrong. A uniform shift raises every
task for a strong agent, including the ones it fails, whose rare finished runs then carry a
weight of up to 20. The agent-known model learned which tasks move with the agent's
strength.

### Ability-aware predictor (`train_ability.py`)

The agent's resolve rate theta is an input: logit f = [logit theta − logit(mean training
theta)] + trees(features, theta), with a monotone constraint on theta. The uniform shift is
the starting point and extrapolates to agents weaker or stronger than any training agent;
the trees learn where it should be smaller or larger. For training runs theta is the
agent's rate on the other half of its tasks.

With theta set to the true rate (a bound), it is the cheapest predictor tried:

| | Tracking, tuned | Learned tracking | Index | Floor-0.05 simulation | Median skew |
| --- | --- | --- | --- | --- | --- |
| Uniform oracle shift | 44.7% | 44.7% | 46.6% | 44.0% (94.1 / 86.9) | −4.4 |
| Agent known | 46.0% | 45.6% | 46.2% | 45.5% (94.7 / 93.2) | −2.7 |
| Ability-aware, true theta | **42.2%** | **43.0%** | **44.2%** | **41.6%** (94.6 / 90.1) | −2.7 |

With theta estimated from earlier batches (`ability_eval.py`), the advantage is gone:

| Floor 0.05 | Compute | Worst agent | Coverage |
| --- | --- | --- | --- |
| LightGBM + Vedanti's correction, damping 100 | 46.8% | 1.29x | 93.3 / 91.2 |
| Ability-aware, damping 100 | 46.4% | 1.33x | 93.0 / 90.8 |
| Ability-aware, precision-weighted (empirical Bayes) | 47.6% | 1.85x | 92.7 / 90.5 |
| Ability-aware, first 50 tasks run to the end | 91.6% | 1.77x | 92.9 / 91.0 |
| Ability-aware, first 100 tasks run to the end | 128.1% | 2.57x | 93.0 / 90.7 |
| Ability-aware, first 100 tasks at floor 0.3 | 62.9% | 1.40x | 92.8 / 90.9 |
| Ability-aware, true theta (bound) | 41.6% | 0.75x | 94.6 / 90.1 |

At floor 0.1 (damping 50) it is 52.1% against 52.6% for the correction, at floor 0.2 57.8%
against 57.9%.

- **Online, the ability-aware predictor ties the uniform correction.** The bottleneck is the
  estimate of theta, not the predictor: at floor 0.05 a task's estimate has variance about
  1.3, so after 100 tasks theta is still uncertain by about ±0.11, as large as the spread
  between agents.
- **Learning theta first does not pay.** The aggressive evaluation spends only about 7% of
  full-run tokens, so finishing even 50 tasks (10% of full cost) more than doubles the bill.
- **The coverage loss comes from estimating theta inside the evaluation**: with theta fixed
  at the true rate, mean coverage is 94.6%; with theta from earlier batches it is about 93%
  whichever rule is used, and even after 100 exact tasks. This belongs with the interval
  theory (`coverage/THEORY.md`, Section 3).

### Where the strength can come from: a reference arm in tool A/B tests (`paired_check.py`)

A cheap evaluation does not contain enough information to learn the agent's strength. In a
tool A/B test it does not have to: the baseline setup has usually been run to the end
already, and its outcome on each task is known before the variant is evaluated, so using it
as an input keeps the variant's estimate unbiased. The 37 agents include pairs that differ
the way an A/B test would (same model, another mini-swe-agent version or reasoning effort).
The variant's step-0 prediction combines its held-out LightGBM prediction with the reference
arm's outcome, through a logistic regression fitted on the other pairs:

| Pairs (both directions) | One-shot compute, LightGBM | With the reference arm |
| --- | --- | --- |
| Close: the arms agree on at least 87% of tasks (8) | 47.7% | **40.2%** |
| Distant: below 87% (8) | 53.5% | 56.2% |
| All (16) | 50.5% | 47.5% |

Close pairs are the same model under a scaffold or effort change (gpt-5.2, gpt-5.2-high,
opus-4.5, sonnet-4.5); for them the variant needs 16% less compute. Distant pairs (glm 4.6 →
5, devstral small → large, gpt-5-mini v1.7 → v2.0, deepseek-3.2 v1.17 → v2.0) get worse,
because a combination learned on all pairs trusts the reference arm too much for them. Tool changes
keep the model and the scaffold, so they should sit at the close end; Rishi's runs will
show it.

## Why the learned predictor did not beat LightGBM (deep dive, Nov 20)

`deep_dive.py` and `probe_estimate.py`; tables in `results/deep_dive_*.csv`. Policy-dependent
numbers use tracking at kappa 0.25 with a 0.05 floor; compute is the geometric mean over the
37 held-out agents.

### 1. Most of the cost comes from the step-0 prediction

Lemma 2 splits each task's estimator variance exactly into the outcome's own variance (there
even with full runs), the term from runs stopped before step 1 and scored by the step-0
prediction, and the terms from predictions during the run.

| Predictor | Outcome's variance | Step-0 term | In-run terms | Tokens spent | Compute | If step 0 were perfect | If in-run were perfect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stand-in | 13.7% | 67.1% | 19.2% | 7.1% | 55.9% | 18.0% | 45.0% |
| LightGBM gold | 14.8% | 67.6% | 17.6% | 7.2% | 52.7% | 16.5% | 43.3% |
| Ability-aware, true theta | 17.9% | 73.9% | 8.1% | 7.4% | 42.2% | 10.9% | 38.7% |

(Shares are means over agents; the last two columns keep the policy fixed.) Two thirds of the
variance comes from the step-0 prediction. A perfect step-0 prediction would cut compute to a
third; perfect predictions during the run only by a fifth.

### 2. The step-0 prediction is limited by information, not by training

The variance-weighted model put 74% of its training weight on step-0 rows and got the same
step-0 error; it only lost accuracy later in the run:

| Brier score | Step 0 | Steps 1-4 | 5-19 | 20-49 | 50+ |
| --- | --- | --- | --- | --- | --- |
| Plain LightGBM | 0.1267 | 0.1214 | 0.1145 | 0.1143 | 0.1242 |
| Variance-weighted | 0.1275 | 0.1238 | 0.1209 | 0.1207 | 0.1368 |
| Share of training weight | 73.5% | 12.3% | 8.8% | 3.8% | 1.7% |

At step 0 the model knows only the task, and 60% of its split gain is one feature, how often
the other agents solved the task (then repeated commands 7%, gold patch size 5%, everything
else 2.6% or less). More training effort on the same information cannot help.

### 3. What the step-0 error is made of

| Part of the step-0 Brier score (mean over agents) | Size | Share | How it was measured |
| --- | --- | --- | --- |
| Not knowing the agent's strength | 0.029 | 23% | held out (0.127) minus ability-aware with the true rate (0.098) |
| Which tasks this agent solves, beyond its strength and the task's difficulty | 0.046 | 36% | the rest |
| Luck: the same agent on the same task passing or failing | 0.052 | 41% | below |

Luck is measured two ways. Where one agent ran each task several times, the run-to-run
variance is: Nebius 0.051 (about 10 runs per task; 20% of the outcome variance), CoderForge
SWE-smith tasks 0.037 (4 runs; 15%), SWE-rebench tasks 0.070 (8 runs; 29%), R2E-Gym tasks
0.076 (8 runs; 33%). On SWE-bench, two near-identical configurations of one model disagree on
a task with probability d, which bounds the luck by d / 2: 0.054 for gpt-5.2 default against
high effort, 0.050 for opus-4.5 across a scaffold version (0.06 to 0.13 for less similar
pairs). No predictor can remove the luck, so even with the agent known a step-0 prediction
keeps a Brier score of at least about 0.05 on SWE-bench Verified.

**A model-family prior does not help.** Adding how often the new agent's same-vendor siblings
solved each task changes step-0 Brier from 0.1267 to 0.1280 (better for 18 of 37 agents).

### 4. The deciding information arrives at the end, and the logs do not show it

Share of the step-0 uncertainty (Brier score) resolved after a share of the run's tokens:

| Tokens spent | 10% | 25% | 50% | 75% | 90% | Last step before the outcome |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM gold | 5.4% | 7.8% | 10.1% | 11.2% | 12.0% | 13.0% |
| LightGBM base | 2.1% | 2.1% | 3.1% | 4.6% | 5.1% | 5.1% |

One step before the end, 87% of the uncertainty is still there: the outcome is decided by the
hidden tests, which the agent's own checks do not reproduce. Yet the code that decides it
exists much earlier:

| Share of the run's tokens at... (median, interquartile range) | Runs that pass | Runs that fail |
| --- | --- | --- |
| the last edit of any file | 89% (76-95%) | 91% (79-95%) |
| the last edit to a file the reference patch touches | **27%** (16-42%) | **25%** (13-42%) |

Agents finish changing the files that matter about a quarter of the way through and spend the
rest testing, writing scratch scripts and polishing.

### 5. The ability-aware predictor needs the agent's strength to within ±0.05

| Error in theta | −0.20 | −0.15 | −0.10 | −0.05 | 0 | +0.05 | +0.10 | +0.15 | +0.20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Compute | 65.6% | 57.0% | 49.0% | 43.5% | 42.1% | 44.9% | 51.3% | 59.2% | 70.2% |

Typical error of the in-evaluation estimate (damping 100, normal approximation): 0.127 after
25 tasks, 0.097 after 100, 0.078 after 200, 0.056 at the end. For most of an evaluation the
estimate is off by 0.06 to 0.13, which is why online it ties the plain correction.

### What this says about improving it

- **More features, other losses, family priors: no.** The step-0 prediction is capped by
  information, and in-run features see little before the end.
- **Mid-run test probes: the most promising.** Run the benchmark's hidden tests on the
  agent's code after it edits, at no LLM-token cost, and feed the result to the predictor.
  `probe_estimate.py` approximates this offline: from the step after a run's last gold-file
  edit (detected in 87% of runs), the prediction becomes a signal that matches the final
  outcome with a given accuracy.

  | Probe | Tracking | Learned tracking | Index | Worst agent (tracking) |
  | --- | --- | --- | --- | --- |
  | None | 53.0% | 52.6% | 54.1% | 4.7x |
  | 75% accurate | 52.6% | 51.6% | 54.0% | 5.0x |
  | 90% accurate | 50.8% | 55.5% | 53.4% | 4.4x |
  | Always right | **30.0%** | **35.3%** | **36.0%** | 2.5x |

  The gain hinges on accuracy: a wrong confident signal on a stopped run is very costly. Tests
  run on the final code are the final grade by definition, so after the last real edit a probe
  should almost never be wrong; how early it settles can only be measured on live runs, which
  needs a probe hook in Rishi's plug-in.
- **A reference arm in A/B tests** (`paired_check.py`): 16% less compute for close pairs.

## Using the predictor (integration, Nov 20)

### For coverage/ (Vedanti's plots)

The held-out gold LightGBM predictions in Vedanti's format (agent, task_id, k, f, next_cost,
outcome):

```bash
python predictor/train_lgbm.py --features gold --split loao        # or --split temporal
python predictor/export.py coverage --split loao                   # -> data/predictor/for_coverage/
python coverage/experiments.py --data data/predictor/for_coverage --split loao
```

Her scripts write to `coverage/results/`, so the last command overwrites the stand-in's results
there. The same predictions scored with her policies are already in
`results/lgbm_gold_{loao,temporal}_policy.csv` (`evaluate.py`). The temporal split covers the
same 31 agents as hers; tracking needs a median of 56.7% of full-run compute there, against
63% for the stand-in.

### For a live harness (Rishi's plug-in)

`runtime.py` runs everything at evaluation time: the success prediction, the learned remaining
cost, the cost-aware stopping policy, the balanced plan, and the estimate with its design-based
interval.

```bash
python predictor/export.py runtime     # trains on all 37 agents -> data/predictor/runtime/all/
```

```python
from runtime import Runtime
rt = Runtime.load("data/predictor/runtime/all")
plan = rt.plan(task_ids, seed=0)              # all tasks of a batch, before any run starts
for task_id in task_ids:
    run = plan.start(task_id)
    while run.should_continue():              # decided before every agent step
        step = ...                            # one agent step: a shortstop.Step or a dict of its fields
        run.observe(step)
        if submitted:
            run.finish(outcome)               # 0 or 1, from the benchmark's tests
            break
print(plan.result())   # score, 95% interval, tokens, runs started and finished
```

- **Step format.** Steps are in the shared format: actions with the command in a ```bash block,
  as mini-swe-agent writes them, and observations with `<returncode>`.
- **Task features.** These exist for SWE-bench Verified tasks. Other tasks fall back to average
  difficulty: the estimate stays unbiased but costs more.
- **Batch planning.** The balanced plan needs every task of a batch planned before the first run
  starts. `rt.predictor(task_id)` gives the plain `shortstop.Predictor` interface.
- **Two probability conventions.** `shortstop.rr_estimate` takes conditional continue
  probabilities, p_k = P(continue into step k | reached step k - 1). `coverage/stopping.py` and
  `runtime.py` work with P_k = P(reach step k). `runtime.py` converts before calling
  `rr_estimate`. Passing P_k directly double-counts and biases the estimate badly;
  `tests/test_runtime.py` caught this.
- **Tests.** `tests/test_runtime.py` checks that the step features match the offline ones, that
  the policy never increases and respects the floor, that the estimate matches the formula, and
  that the live loop is unbiased.
- **Name clash.** `step_features.py` was renamed from `features.py`, which clashed with
  `coverage/features.py` on the import path.

**End to end on real runs of agents the models never saw** (`demo_runtime.py`, models from
`export.py runtime --exclude <agent>`, 20 complete evaluations each):

| Held-out agent | True rate | Mean score (sd) | Interval contains truth | Half-width (full runs) | Tokens spent | Compute for the reported width |
| --- | --- | --- | --- | --- | --- | --- |
| claude-4.5-sonnet-high | 0.714 | 0.717 (0.035) | 19 / 20 | 0.077 (0.040) | 7.5% | 28% |
| gpt-oss-120b | 0.260 | 0.266 (0.045) | 20 / 20 | 0.110 (0.038) | 9.6% | 79% |

## A trained component that beats the hand-made policies: learned remaining cost (Nov 20)

The theory note and Check 3 show that training the success predictor on the variance objective
cannot beat a calibrated classifier. Theorem 1 says what else the optimal policy needs: the
expected cost still to come. Vedanti's policies either ignore it (tracking) or estimate it on
a coarse state (learned tracking: step bin and prediction decile), yet run costs vary a lot
(coefficient of variation 1.3 across one agent's tasks).

**Premise** (`allocation_check.py`, one-shot policies; `cost_aware.py`, per-step): told each
run's true cost, a policy spends much less.

| Policy (gold LightGBM, held out) | Compute | vs reference |
| --- | --- | --- |
| One-shot, same probability for every task | 56.6% | |
| One-shot, by uncertainty f (1 - f) | 54.2% | 0.958x |
| One-shot, by uncertainty / cost predicted from the task | 52.5% | 0.928x |
| One-shot, by uncertainty / true run cost (bound) | 46.2% | 0.817x |
| Tracking (per step) | 52.6% | |
| Tracking with the true remaining cost (bound) | 45.1% | 0.857x |

**Learned inputs** (`cost_aware.py`): two LightGBM regressions per held-out agent, trained on
the other agents, predict from each step's features (plus how expensive the task was for the
other agents) the log of the tokens still to be spent and the squared error of the current
prediction. The predicted remaining cost (relative to its mean at step 0 over the agent's runs)
enters the tracking target as kappa * sqrt(f (1 - f) / r_hat). Correlation of predicted and
true log remaining cost: median 0.61 over agents (0.38 to 0.78).

| Policy | Median | Geo. mean | vs tracking | Worst agent | Coverage, mean / worst |
| --- | --- | --- | --- | --- | --- |
| Tracking | 47.9% | 52.6% | | 4.46x | 94.4% / 91.8% |
| + learned error | 48.2% | 53.7% | 1.021x | 5.27x | 94.5% / 91.4% |
| **+ learned remaining cost** | **45.3%** | **48.9%** | **0.931x** | **3.02x** | **95.1% / 93.0%** |
| + both | 44.7% | 49.2% | 0.936x | 3.30x | 94.8% / 92.8% |
| + true remaining cost (bound) | 42.6% | 45.1% | 0.857x | 2.88x | 94.5% / 92.4% |

- **The learned remaining cost recovers half of what a perfect cost oracle gives** and beats
  every hand-made policy (tracking 52.6%, learned tracking 52.6%, index 54.1%, one-shot 56.6%),
  with coverage intact.
- **A learned error estimate does not help**, as the theory note predicts: f (1 - f) from a
  calibrated classifier is already the right uncertainty.

**Robustness.** The gain holds across training seeds and on the second benchmark:

| Run | Tracking | + learned remaining cost | Ratio | Agents cheaper | Coverage, mean / worst | True-cost bound |
| --- | --- | --- | --- | --- | --- | --- |
| Verified, seed 0 | 52.6% | 48.9% | 0.931x | 34/37 | 95.1% / 93.0% | 45.1% |
| Verified, seed 1 | 52.1% | 48.3% | 0.928x | 34/37 | 94.9% / 91.8% | 44.8% |
| Verified, seed 2 | 53.0% | 49.3% | 0.930x | 34/37 | 94.9% / 92.6% | 45.1% |
| SWE-rebench leaderboard (36 configurations) | 57.9% | 53.3% | 0.921x | 30/34 * | 94.8% / 93.0% | 50.2% |

\* configurations with a 20 to 80% resolve rate; over all 36 the ratio is 0.921x as well. The
ratio varies by 0.3 points across seeds, against the 3.4% spread of absolute levels, because a
seed moves both policies together. On both benchmarks the learned cost model captures 50 to
60% of what the true remaining cost would give.

**The cost model's loss does not matter** (`cost_models.py`, Verified): squared error on
log(1 + r) gives 48.9%; Tweedie and gamma losses, which target E[r] directly, 49.1%; Duan's
smearing correction is a constant factor and cancels in the relative cost (48.9%). The gap to
the true-cost bound is the part of a run's length that the features cannot predict, not the
choice of loss.

## Balanced stopping decisions, stacked with the learned cost (Nov 20)

`balanced.py`; tests in `tests/test_balanced.py`.

**Design.** Each run still stops at s = max{k : U < P_k} with a single uniform U. By Lemma 1 the
estimate stays exactly unbiased as long as each run's U is uniform and independent of that run,
so the uniforms of different runs may depend on each other. They are spread evenly within
strata, in four steps:

1. Sort the runs by their step-1 reach probability P_1 (ties by f0).
2. Cut them into strata of B = 50 runs.
3. Assign the runs of a stratum U = (position + V) / B, with a random permutation of positions
   and independent V ~ U(0, 1). This is a Latin hypercube, Vihola's (2018) stratified truncation.
4. The numbers of runs that start, and that reach each later step, become nearly fixed within a
   stratum, so its average miscalibration cancels.

The tests check three things: each stratum is a Latin hypercube, every run's U is uniform
(Kolmogorov-Smirnov), and the estimates average to the true outcomes (20,000 Monte Carlo draws).

**Intervals.** Under this design, sd / sqrt(n) stays valid but conservative, and it does not
get narrower. The design-based estimator splits the variance of the mean into two parts:

- **The outcome's own variance**, Var(Y) / n. Y is binary, so this is θ̂ (1 − θ̂) / n.
- **The corrections' design variance.** Only runs that start carry a correction c = Ŷ − f0.
  Within each stratum, merged with the next until it has at least 2 started runs, it is
  estimated from the spread of c · (expected number started), with the finite-population
  correction 1 − m / B, and Student's t with Σ (m − 1) degrees of freedom.

The spread is measured within a stratum, so the stratum's average miscalibration does not count
toward the variance. Two cheaper variants under-cover slightly or barely help: dropping the
finite-population correction leaves the intervals as wide as the naive ones, and using the
normal quantile instead of t gives 94.2% coverage. Paired- or successive-difference estimators
on stratum totals have too few degrees of freedom and count the planned differences between
strata as noise.

**Results** (gold LightGBM held out, policies tuned on the other agents, 1,000 replicates). The
reported column is the compute needed to reach the interval width that is actually reported,
relative to full runs.

| SWE-bench Verified (37 agents) | Actual | Reported | Coverage, mean / worst | Worst agent (actual) |
| --- | --- | --- | --- | --- |
| Tracking, independent stopping, sd / sqrt(n) (the baseline) | 53.3% | 51.3% | 94.4% / 91.9% | 4.79x |
| Learned remaining cost, independent | 48.9% | 47.7% | 94.8% / 92.5% | 3.18x |
| Tracking, balanced, design-based interval | 45.3% | 46.5% | 94.7% / 92.1% | 1.43x |
| **Learned remaining cost, balanced, design-based interval** | **43.3%** | **43.2%** | **94.6% / 92.9%** | **1.11x** |

| SWE-rebench leaderboard (36 configurations) | Actual | Reported | Coverage, mean / worst |
| --- | --- | --- | --- |
| Tracking, independent, sd / sqrt(n) | 58.2% | 57.0% | 94.7% / 92.4% |
| Learned remaining cost, independent | 53.3% | 52.4% | 94.9% / 93.1% |
| **Learned remaining cost, balanced, design-based interval** | **46.7%** | **48.0%** | **94.8% / 93.5%** |

The two improvements stack. Against the baseline they cut the compute for a given reported
interval width by 16% on both benchmarks: the per-agent geometric-mean ratio is 0.842 on
Verified (37 of 37 agents cheaper) and 0.843 on the leaderboard (35 of 36). Coverage matches
the baseline's.

**Who gains** (Verified, final against baseline, geometric-mean ratio by resolve rate):

| Resolve rate | Up to 30% | 30-50% | 50-65% | Above 65% |
| --- | --- | --- | --- | --- |
| Ratio | 0.40 | 0.68 | 0.94 | 0.87 |
| Agents | 2 | 3 | 14 | 18 |

The worst agent drops from 4.3x to 1.3x full-run compute. On the leaderboard the 0.7%-resolve
configuration improves from 42.9x to 16.4x but stays far above full runs.

**Caveats.**

- **B = 50 was chosen among {25, 50, 100}.** B = 25 is within a point, so the choice matters
  little.
- **The design needs f0 and P_1 for a batch of runs before deciding which to start.** Both use
  only task features and the learned step-0 cost, so this works in a live evaluation.
- **θ̂ (1 − θ̂) / n targets the population of tasks** (Theorem 2a's setting). For a fixed task
  set the interval is conservative.

## Literature survey: Monte Carlo methods and novelty (Nov 20)

Citations were checked against arXiv, DOI records or publisher pages.

**Already optimal.** The sqrt(error removed / cost) allocation behind tracking, learned
tracking and the index policy is the state-dependent form of the optimal randomization
distribution for randomized truncation estimators (Rhee and Glynn 2015, Props. 1 to 3, with
the variance x compute efficiency criterion of Glynn and Whitt 1992). It also matches Neyman
allocation, multilevel Monte Carlo allocation (Giles 2008), and active statistical inference
with costs (Zrnic and Candès 2024, arXiv 2403.03208). The coupled-sum estimator used here is
the right one for a single run: the single-term estimator pays for the same prefix but uses
only the last increment, and the independent-sum estimator needs increments generated
separately. `coverage/THEORY.md` should cite Rhee and Glynn for the non-adaptive version of
Theorem 1.

**Novelty.** No prior work applies randomized truncation to AI-agent or LLM evaluation.

- **Closest problem: EarlyEval** (arXiv 2609.02783). It uses LightGBM, leave-one-agent-out
  and SWE-bench Verified, but stops at fixed thresholds and substitutes the prediction for the
  outcome: 1 to 2 points of bias, no intervals, no cost model.
- **An independent audit** (arXiv 2609.25647) finds that such predictors' calibration does not
  transfer to new agents, which matches the deep dive.
- **Closest mechanism:** efficiency-aware Russian roulette and splitting in rendering (EARS,
  Rath et al. 2022), which sets survival probabilities from variance and cost estimates, and
  Beatson and Adams (ICML 2019), who tune a truncation distribution online per unit of compute.
- **The estimator is classical:** augmented inverse-probability weighting with known
  missingness probabilities and surrogate markers (Robins and Rotnitzky 1992), and two-phase
  sampling (Särndal and Swensson 1987).
- **What is new here:** per-run stopping inside agent trajectories driven by a learned
  remaining-cost model, exact unbiasedness with intervals, and the variance decomposition.

**Most promising methods to adopt**, ranked by expected gain and feasibility:

1. **Balanced or fixed-size stopping decisions** (Hájek 1964; the cube method, Deville and
   Tillé 2004; stratified truncation levels, Vihola 2018). See below.
2. **A weight window plus skew-aware intervals.** A weight window (Vorba and Křivánek 2016)
   floors P_k so that no weighted increment exceeds a cap W, P_k ≥ max(f, 1 - f) / W.
   Skew-aware intervals are Johnson's t or bootstrap-t / BCa. This targets the coverage loss
   seen under online correction.
3. **A PPI++-style variance-tuned predictor** (arXiv 2311.01453), a + b f fitted on earlier
   tasks, with a fallback to full runs when estimated relative compute exceeds 1. This targets
   the weak-agent worst case.
4. **Adaptive task order for ability** (IRT adaptive testing; tinyBenchmarks, arXiv
   2402.14992). Run first the tasks that are most informative about theta per unit of cost.
5. **Finite-sample intervals** (betting, Waudby-Smith and Ramdas). Valid at any n, but wider;
   use them as a check.

Splitting (re-running uncertain runs) attacks only luck, and trajectory verifiers are capped:
perfect in-run predictions give 40.8% against 49.8% on the leaderboard.

**Premise check for balanced decisions** (one-shot, 5% of runs finished, exact). Independent
coin flips are Poisson sampling, so the step-0 term pays the uncentred E[(Y - f0)^2]. A fixed
number of finished runs pays only the variance of Y - f0, and fixed numbers within groups of
similar f0 pay only its within-group variance. The agent's average miscalibration drops out by
design, with no fitted correction and no extra skew.

| One-shot, 5% finished | Verified: geo. mean | Worst agent | Leaderboard: geo. mean | Worst agent |
| --- | --- | --- | --- | --- |
| Independent coin flips (now) | 56.6% | 5.21x | 64.5% | 44.0x |
| A fixed number of finished runs | 49.6% | 1.84x | 58.3% | 15.2x |
| **A fixed number in each of 5 groups of similar f0** | **46.7%** | **0.92x** | **51.4%** | **1.68x** |
| Balanced on f0 (regression residual) | 48.0% | 0.95x | 52.9% | 1.00x |

This matches the oracle per-agent shift (44.5% one-shot on Verified) without knowing the agent.
Two open steps:

- **The per-step version.** Stratify each run's stopping uniform across runs with similar f0;
  every run keeps its exact marginal stopping probability. It has to be measured by simulation.
- **A design-based variance estimator.** Without one, the usual sd / sqrt(n) interval stays
  valid but conservative and does not get narrower. Two-phase sampling variance formulas, or
  random groups, are the candidates.

Both sit on the estimator side, so this is joint work with Vedanti.

## A second benchmark: the SWE-rebench leaderboard (Nov 20)

`convert_rebench_lb.py` converts the Python subset of
`nebius/SWE-rebench-leaderboard-trajectories`: 36 agent configurations (25 models; the common
scaffold plus Claude Code, Codex, Cursor and Junie) on the same 110 fresh tasks, 5 runs per
task (2 for the Claude Code effort panel). 18,468 runs, 1.0M steps, 46.9% resolved, gold
patches for every task (`nebius/SWE-rebench-leaderboard`). Each tool call becomes one step,
rewritten as the equivalent shell command; per-run token totals are spread over the steps in
proportion to the characters/4 estimate. Resolve rates range from 0.7% to 63%.

**LightGBM, each configuration held out** (`train_lgbm.py --data data/rebench_lb`; `task_rate`
is now computed per task over all runs of the other configurations, unchanged for Verified):

| Policy | Base, median / geo. mean | Gold, median / geo. mean | Gold coverage, mean / worst |
| --- | --- | --- | --- |
| One-shot | 53.8% / 65.0% | 53.5% / 64.5% | 94.4% / 90.6% |
| Tracking | 48.9% / 59.4% | 49.1% / 57.8% | 94.8% / 92.0% |
| Learned tracking | 51.2% / 60.6% | 50.8% / 59.5% | 94.7% / 92.4% |
| Index | 50.9% / 61.9% | 52.2% / 60.6% | 94.7% / 93.4% |

- The median matches SWE-bench Verified (about 48 to 50%) and coverage holds. Gold features
  are 2 to 3% cheaper than base, within training noise.
- The geometric mean is higher because of two very weak configurations; the worst (DeepSeek
  V3.2 without demonstrations, 0.7% resolved) costs 44 times full runs, the weak-agent problem
  in its extreme form: when almost every run fails, Var(Y) is tiny and any error dominates.

**The deep dive replicates** (agents with 20 to 80% resolve rate): variance shares 15.1%
outcome, 66.9% step-0 term, 17.9% in-run terms (Verified 14.8 / 67.6 / 17.6); compute 49.8%,
16.3% with a perfect step-0 prediction, 40.8% with perfect in-run predictions; 11% of the
step-0 uncertainty resolved one step before the end (Verified 13%); the last edit to a gold
file comes at 33% of the run's tokens (median, runs where one is detected).

**Luck, measured directly** (5 runs per task, `results/rebench_lb_luck.csv`): for the 34
configurations with a 20 to 80% resolve rate, run-to-run variance is 0.074 (median), 30% of
the outcome variance (13 to 45%); on 30% of tasks an agent's runs do not all agree.

**A/B with a reference arm, on real one-change pairs** (`paired_rebench.py`): the reference
configuration's per-task solve rate (over its runs) as an input to the variant's step-0
prediction, combined with held-out LightGBM through a logistic regression fitted on the other
pairs. One-shot compute, geometric mean over pairs in both directions:

| Pair type | Pairs | Mean per-task rate gap | LightGBM | With reference arm | Luck floor |
| --- | --- | --- | --- | --- | --- |
| Same model and scaffold, another reasoning effort | 10 | 0.096 | 55.0% | **37.9%** | 28.8% |
| Same model, another scaffold | 6 | 0.164 | 55.2% | 50.7% | 30.5% |
| Same scaffold, neighbouring model version | 6 | 0.156 | 52.8% | 50.2% | 35.0% |
| Configuration fix or provider change | 4 | 0.144 | 50.4% | 49.8% | 39.6% |
| All | 26 | | 53.8% | **45.1%** | 32.1% |

The reference arm helps in 21 of 26 directions, and the closer the arms, the more: for effort
changes (same model, same scaffold, the closest analogue of a tool tweak) it cuts compute by
31%, two thirds of the way to the floor that luck sets. With 5 runs per task the reference
rate is much sharper than SWE-bench Verified's single run, which is why the gain here is
larger than in `paired_check.py`.

## A theory note for contribution 3

Training the predictor "to minimize the estimator's variance" and training it as an ordinary
success classifier aim at the same function. By Lemma 2 the variance is
$\sum_k w_k (Y - f_{k-1})^2$ with $w_k \ge 0$ fixed by what was seen before step k. For any
such weights the minimizer of $E[w_k (Y - f_{k-1})^2]$ over functions of the information at
step k-1 is $E[Y \mid \text{that information}]$, which is what log loss also targets. The
cost term does not involve f once the policy is fixed. So with enough data and model
capacity, the variance-optimal predictor is the calibrated conditional probability, whatever
the policy. Training on the variance objective can only help through finite data and
capacity, through a policy that sees nothing but the prediction, or through the shift from
training agents to a new one. Check 3 measures whether that matters in practice; the larger
levers above are information that arrives early and agent-level calibration.

## What this does not show yet

- Policy tuning uses predictions for the training agents that came from models which saw
  the held-out agent (the same caveat as in `coverage/README.md`).
- Each comparison is one training run; the seed spread (up to 3.4%) is measured for gold
  LightGBM only.
- Coverage numbers use 500 replicates (about ±1 point per agent); Vedanti's use 2,000.
- Inside an evaluation, agent adaptation recovers about 10% (51.9% → 46.4% at floor 0.05)
  of the 12 to 20% the bounds promise (45.5% agent known, 41.6% ability-aware with the true
  rate), at a cost of about 1.5 points of coverage.
- The paired checks use step-0 predictions and the one-shot formula only, and the pairs are
  effort, scaffold and version changes, not tool changes; Rishi's runs are the real test.
- The leaderboard's multilingual subset (Go, Java, Rust, TypeScript) is not converted: the
  features' test-run patterns are Python-specific.
- Nebius and CoderForge are used only to measure luck so far, not to train or test a
  predictor.
- No result yet on Rishi's runs.
