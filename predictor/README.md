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
python predictor/features.py                  # about 1 minute with 8 workers
python predictor/train_lgbm.py --features gold --split loao     # about 11 minutes
python predictor/evaluate.py data/predictor/predictions_gold_loao.pkl --name lgbm_gold_loao
python predictor/convert_openhands.py nebius  # about 8 minutes, 1.6 GB written
python predictor/features.py --data data/openhands \
    --meta data/meta/swebench_verified.parquet data/meta/swerebench_gold.parquet \
    --out data/openhands/features_rich.pkl --workers 1      # about 10 minutes
```

`data/meta/swebench_verified.parquet` (gold patches and tests for Verified) is read from
`princeton-nlp/SWE-bench_Verified` on Hugging Face; see "Data" below.

| File | What it does |
| --- | --- |
| `features.py` | One row per step prefix: Vedanti's counts (identical values), plus *process* features (last test result, steps since the last edit, reproduction scripts, repeated commands, format errors) and *gold* features (has the agent viewed or edited the files the reference patch touches, has it run the tests that must pass) |
| `train_lgbm.py` | LightGBM, one model per held-out agent; feature groups `base` / `process` / `gold`, splits `loao` / `lofo` / `insample` |
| `evaluate.py` | Scores any predictions file with Vedanti's five stopping policies, each tuned on the other agents, plus how much the prediction improves through the run |
| `variance_weights.py` | Row weights that make the training loss equal to the estimator's variance (THEORY.md, Lemma 2) |
| `convert_openhands.py` | Nebius / CoderForge OpenHands runs into the shared step format |

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
difference 0.001, identical tuned settings). The base features from `features.py` match
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
- Nebius is converted (4.3M steps) and featurized; gold patches for SWE-rebench tasks come
  from `nebius/SWE-rebench` (21,336 tasks).

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
- No agent-adaptive predictor yet: the 13 to 15% is a bound (agent known from half its tasks,
  or an oracle), not something achieved inside an evaluation.
- The Nebius data is converted and featurized but not yet used; CoderForge is not converted.
- No result yet on Rishi's runs.
