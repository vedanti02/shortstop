# predictor (Deepika)

Vedanti's estimator (`coverage/`, merged in #1) lets an evaluation stop agent runs early, at
random, and still give an exactly fair score with a valid 95% error bar. How much compute that
saves depends on two things this folder builds: the **predictor** of each run's chance of
success, and the **stopping rule** that decides which runs continue. Every number here is
measured on agents the models never saw, with Vedanti's metric, so it is directly comparable with
`coverage/README.md`.

## In short

**Results.** Compute needed for the error-bar width that is reported, relative to finishing every
run (lower is better):

| | SWE-bench Verified (37 agents) | SWE-rebench leaderboard (36 configurations) |
| --- | --- | --- |
| LightGBM + Vedanti's tracking rule (the baseline here) | 51.3% | 57.0% |
| + learned remaining cost: the runtime's `standard` design, proven error bars | 47.7% | 52.4% |
| + balanced stopping: the runtime's `balanced` design, proven error bars ([THEORY_BALANCED.md](THEORY_BALANCED.md)) | 45.4% | 49.8% |
| The earlier balanced design, `balanced-unproven`: error bars slightly too narrow, kept for comparison | 43.2% | 48.0% |

In Vedanti's own simulation (`evaluate.py`), the tracking rule needs 55.9% with her stand-in
predictor and 53.0% with LightGBM. With the optional seed-averaged models, the four Verified
numbers drop to 50.8%, 47.1%, 44.7% and 42.7%.

**What worked**

- **LightGBM** predicts runs better than the stand-in: about 5% less compute.
- **Learned remaining cost.** A second model predicts how many tokens a run still needs, and the
  stopping rule uses it: 7 to 8% less compute on both benchmarks, with Vedanti's proven error
  bars.
- **Balanced stopping.** Stop/continue draws are dealt evenly within groups of 50 similar tasks,
  with an error-bar formula that is now proven ([THEORY_BALANCED.md](THEORY_BALANCED.md)): 12 to
  13% less compute than the baseline in total, 5% less than `standard`. Working out the proof
  showed that the earlier formula, which gave 16%, is slightly too narrow; it stays available as
  `balanced-unproven`.
- **A/B hints.** In a tool A/B test, the reference version's per-task results improve the
  variant's predictions: 16 to 31% less compute on real one-change pairs.

**What did not:** richer features inside the run, training on the variance objective (the theory
note shows it cannot win), learning the agent's strength during the evaluation (it ties
Vedanti's simpler correction), priors from the same model family, and more training data
(OpenHands runs, a solve rate from 128 leaderboard systems, SWE-rebench runs). Averaging three
training seeds is the one training change with a small, reliable gain, about 1% on every setup;
it is an option (`--seeds 3`) next to the single-seed models.

**Why** (the deep dive):

- About two thirds of the cost comes from the prediction made before the run takes a step.
- 15 to 33% of the variation in outcomes is luck, depending on the dataset: the same agent on the
  same task passing one time and failing the next. No predictor can remove it.
- A run's outcome becomes predictable only at the very end, when the hidden tests run.

**Still open:** everything is measured by replaying logged runs, the balanced design's proof
needs a second reader, and very weak agents are still expensive. See
[What this does not show yet](#what-this-does-not-show-yet).

## Contents

- [In short](#in-short): results, what worked and what did not
- [Key terms](#key-terms): the words used throughout
- [How to use it](#how-to-use-it): the runtime's three designs, and predictions for Vedanti's plots
- [Run it](#run-it): every command, in order
- [Files](#files): what each script does
- [Learned remaining cost: a trained component that beats the hand-made policies](#learned-remaining-cost-a-trained-component-that-beats-the-hand-made-policies): main result 1
- [Balanced stopping decisions and the three designs](#balanced-stopping-decisions-and-the-three-designs): main result 2
- [A second benchmark: the SWE-rebench leaderboard](#a-second-benchmark-the-swe-rebench-leaderboard): everything repeated on a second benchmark
- [How we got here](#how-we-got-here): the detailed record: checks, deep dive, literature, theory
- [What this does not show yet](#what-this-does-not-show-yet): limits and open work

## Key terms

- **Agent, task, run, step.** An agent (a model plus a scaffold) attempts a task (a SWE-bench
  issue). One attempt is a run, made of steps (one command and its output each).
- **Held-out agent.** The agent being scored was never in the models' training data (leave one
  agent out, "loao").
- **f0, f_k.** The predicted chance of success before the run's first step (from the task alone),
  and after k steps.
- **Reach probability P_k.** The chance that the run is still going at step k. Each run has one
  lottery number U and continues while U < P_k.
- **Stopping rule (policy).** How P_k is set. P never rises during a run and never drops below a
  floor, usually 0.05; κ is a constant tuned on the other agents.
  - *Tracking*: P = κ·√(f(1−f)), so uncertain runs continue more.
  - *Learned remaining cost*: P = κ·√(f(1−f) / r), with r the predicted tokens still needed.
  - *One-shot*: decide once, at the start.
  - *Learned tracking* and *index*: Vedanti's other per-step rules (`coverage/README.md`).
- **Estimate.** Each run's score Ŷ = f0 + Σ_k (f_k − f_{k−1}) / P_k, exactly unbiased for that run
  (Lemma 1 in `coverage/THEORY.md`). The evaluation's score is the mean over tasks.
- **Relative compute.** Compute needed for a given precision, relative to finishing every run.
  Lower is better; above 100% means early stopping costs more than full runs. *Actual* uses the
  real spread of the scores; *reported* uses the width of the error bar that is reported.
- **Coverage.** How often the 95% error bar contains the true resolve rate (target 95%).
- **Brier score.** The mean squared error of a predicted probability (lower is better).
- **Design.** How the lottery numbers are drawn (independently, or balanced within groups of 50
  tasks) and the error-bar formula that goes with it.
- **Luck.** The same agent on the same task passing one time and failing the next.

## How to use it

### For a live harness (Rishi's plug-in): three designs

`runtime.py` runs everything at evaluation time: the success prediction, the learned remaining
cost, the cost-aware stopping policy, and the score with its 95% interval. It has three designs,
picked with the `design` argument. They share the models and the stopping policy and differ in
two things only: how the stop/continue draws are made, and which interval is reported.

| | `design="standard"` (default) | `design="balanced"` | `design="balanced-unproven"` |
| --- | --- | --- | --- |
| Stop/continue draws | Each run draws its own random number | Dealt evenly within groups of 50 different tasks with similar predictions (one run per task); every task of a group gets the same chance to start | Dealt evenly within the same groups; each task keeps its own chance to start |
| Interval | mean ± 1.96 sd / √n, Vedanti's (`coverage/experiments.py`) | Design-based, plus each started run's own estimate of its later-step noise, with a t quantile (`balanced.py: proven_variance`) | Design-based, with a t quantile (`balanced.py: design_variance`) |
| Proven? | Yes: `coverage/THEORY.md`, Theorem 2a | Yes: [THEORY_BALANCED.md](THEORY_BALANCED.md), Theorem B4 (still needs a second reader) | No, and slightly too narrow in general (THEORY_BALANCED.md, Section 4) |
| Compute for the reported width, Verified / SWE-rebench | 47.7% / 52.4% | 45.4% / 49.8% | 43.2% / 48.0% |
| Coverage on Verified, mean / worst agent | 94.8% / 92.5% | 95.2% / 93.6% | 94.6% / 92.9% |
| Agents costing more than full runs (Verified) | 2 of 37 | 1 of 37 | 1 of 37 |

The score is exactly unbiased under all three designs (Lemma 1); only the interval differs.

**Which one to use.**

- **`standard`** is the default. Its interval rests on the textbook central limit theorem
  (Theorem 2a), and it is probably the better choice for small batches and for agents the
  predictor already judges well.
- **`balanced`** for a large batch of tasks, when narrower intervals for the same tokens matter.
  - It is much cheaper for agents the predictor misjudges, such as agents much weaker than the
    37 training agents (gpt-oss-120b: 83% of full-run compute, against 124% under `standard`).
    For agents the predictor judges well it can cost slightly more than `standard`: it is
    cheaper for 23 of the 37 Verified agents, and 5% cheaper overall.
  - It only helps across a batch: it needs the task list before the first run starts, and
    enough tasks to form groups and to measure the spread within them. It works with 500 tasks;
    below about 150 to 250 tasks `standard` probably gives narrower intervals (a rough estimate,
    not yet measured; see the caveats under "Balanced stopping decisions and the three designs").
  - Until a second reader has checked THEORY_BALANCED.md, label its intervals "design-based,
    proof under review".
- **`balanced-unproven`** only to compare with earlier results: its interval is slightly too
  narrow in general.
- **Comparing agents or tool versions:** use the same design for every arm, so the intervals
  mean the same thing.

**Step 1. Build the models once.**

```bash
python predictor/export.py runtime     # trains on all 37 agents -> data/predictor/runtime/all/
python predictor/export.py runtime --seeds 3   # optional: 3 seed-averaged models -> runtime/all_ens3/
```

To use the seed-averaged models, load `data/predictor/runtime/all_ens3` instead of
`data/predictor/runtime/all`. They work with any design, cost about 1% less compute (see
"Does more training data help?"), and each prediction calls three models instead of one.

**Step 2. Plan the batch with one design.** List every task before the first run starts: every
design measures each run's remaining cost against the batch's average, and the balanced designs
also group similar tasks. A plan keeps its design for the whole batch, and the same seed gives the
same draws.

```python
from runtime import Runtime
rt = Runtime.load("data/predictor/runtime/all")
rt = rt.calibrate(earlier)   # optional: {task_id: 0 or 1} from earlier full runs of this agent
plan = rt.plan(task_ids, seed=0)                        # standard
plan = rt.plan(task_ids, seed=0, design="balanced")     # or balanced, or "balanced-unproven"
```

**Calibrating to the agent (optional).** The models expect an agent like the 37 they were
trained on: their mean step-0 prediction is about 0.6, whatever the agent. `rt.calibrate(earlier)`
moves every success prediction by one logit shift, chosen so that the mean step-0 prediction on
those tasks equals the agent's solve rate there. `earlier` maps task ids to 0/1 outcomes from
earlier full runs of the same agent; tasks outside the batch are fine. Calibrate before the batch
starts and keep the shift fixed: it is then part of the predictor, so every run's estimate stays
exactly unbiased.

- **It matters most for agents unlike the training agents.** On Rishi's Qwen runs (26 to 39%
  solved; the v2 data of Nov 23), the standard design's compute falls from 84–171% of full runs
  to 47–69%, and its coverage rises from 87–88% to 94–96% (`tool_ab_check.py`; each half of the
  tasks is calibrated on the other half).
- **For a tool A/B test**, give both arms the same shift (the baseline's) and the same seed, so
  their step-0 predictions and stopping draws match.
- **Without earlier runs**, Vedanti's in-batch correction (`coverage/adaptive.py`) learns the
  shift during the batch. On Qwen's 100 tasks it learns only about half of it: the baseline
  arm's compute goes from 97% to 82% (tracking policy).
- **The price of a uniform shift is at the tail.** On the 37 training agents its oracle version
  saves 14 to 16% of compute, but the worst agent's coverage drops to 86–90% (see "Agent
  calibration is the larger lever").

**Step 3. Run the agent.** This code is the same for every design.

```python
for task_id in task_ids:
    run = plan.start(task_id)
    while run.should_continue():      # asked before every agent step
        step = ...                    # one agent step: a shortstop.Step or a dict of its fields
        run.observe(step)
        if run_over:                  # the agent submitted, or a step or cost limit was reached
            run.finish(outcome)       # 0 or 1, from the benchmark's tests
            break
```

**When a step never completes.** A context overflow or a crash can end a run during a step,
after `should_continue()` said yes. Call `run.finish(outcome)` without `observe()`: the attempted
step then counts as the one that reveals the outcome. Writing the outcome over the last
prediction instead would bias the estimate. Two more rules keep the loop honest:

- Asking `should_continue()` again before the step is observed (a retry after a format error)
  returns the same answer without a new draw.
- The runtime refuses what would bias the score: `observe()` without asking first, `finish()`
  on a run that was stopped, and `plan.result()` while a started run was neither stopped nor
  finished.

**Step 4. Read the result.** Real output, from replaying claude-4.5-sonnet-high (true resolve
rate 0.714) with models that never saw it, seed 0:

```python
plan.result()
# standard: {'design': 'standard', 'score': 0.703, 'interval': (0.619, 0.787), 'half_width': 0.084,
#            'tasks': 500, 'runs_started': 69, 'runs_finished': 31, 'tokens': 35579545, 'shift': 0.0}
# balanced: {'design': 'balanced', 'score': 0.699, 'interval': (0.615, 0.784), 'half_width': 0.084,
#            'tasks': 500, 'runs_started': 77, 'runs_finished': 34, 'tokens': 37857020, 'shift': 0.0}
# balanced-unproven: {'design': 'balanced-unproven', 'score': 0.701, 'interval': (0.620, 0.782),
#            'half_width': 0.081, 'tasks': 500, 'runs_started': 77, 'runs_finished': 34,
#            'tokens': 37856874, 'shift': 0.0}
```

`runs_started` counts runs that took at least one step, `runs_finished` those that ran to the
end, and `tokens` the total spent: 6.7% (`standard`) and 7.1% (both balanced designs) of the 530
million that full runs cost.
`shift` is the calibration shift, 0 without calibration.

**Try them before going live.** `demo_runtime.py` replays a held-out agent's logged runs
through the runtime, exactly as a live harness would:

```bash
python predictor/export.py runtime --exclude <agent>     # models that never saw <agent>
python predictor/demo_runtime.py <agent>                  # standard and balanced, 20 evaluations each
python predictor/demo_runtime.py <agent> --design all     # or one of standard, balanced, balanced-unproven
```

**Each design's offline numbers.** `balanced.py` simulates every design, after `cost_aware.py`
has written its inputs:

```bash
python predictor/balanced.py --blocks 25,50 --replicates 1000                  # Verified
python predictor/balanced.py --inputs data/predictor/cost_aware_rebench_inputs.pkl \
    --kappas predictor/results/cost_aware_rebench.csv --name balanced_rebench --blocks 50 --replicates 1000
```

In `results/balanced.csv` the three designs are the rows with policy `cost-aware` and:

- `standard`: design `independent`, interval `naive`;
- `balanced`: design `equal-chance B=50`, interval `design + fpc + later steps`;
- `balanced-unproven`: design `stratified B=50`, interval `design + fpc`.

`balanced_check.py` checks the proof's exact formula for the interval's bias on the 37 agents' real runs.

**Details for every design.**

- **Step format.** Steps are in the shared format: actions with the command in a ```bash block,
  as mini-swe-agent writes them, and observations with `<returncode>`.
- **Task features.** These exist for SWE-bench Verified tasks. Other tasks fall back to average
  difficulty: the estimate stays unbiased but costs more.
- **Gold patch table.** `Runtime.load` reads the table that `meta.json` names
  (`data/meta/swebench_verified.parquet`) from the repo root, whatever the working directory, and
  stops with an error if it is missing. Before, a missing table was skipped silently and every
  prediction lost its task features.
- **Plain predictor.** `rt.predictor(task_id)` gives the plain `shortstop.Predictor` interface.
- **Two probability conventions.** `shortstop.rr_estimate` takes conditional continue
  probabilities, p_k = P(continue into step k | reached step k - 1). `coverage/stopping.py` and
  `runtime.py` work with P_k = P(reach step k). `runtime.py` converts before calling
  `rr_estimate`. Passing P_k directly double-counts and biases the estimate badly;
  `tests/test_runtime.py` caught this.
- **Tests.** `tests/test_runtime.py` checks, under every design, that the policy never increases
  and respects the floor, that the estimate matches the formula, and that the live loop is
  unbiased, including runs that end during a step. It also checks that the step features match
  the offline ones, that the standard design's interval equals `interval_stats` in
  `coverage/experiments.py`, that an unknown design name is rejected, that the loop refuses
  misuse, that calibration moves every prediction by one logit shift, and that a missing gold
  patch table is an error.
- **Name clash.** `step_features.py` was renamed from `features.py`, which clashed with
  `coverage/features.py` on the import path.

**End to end on real runs of agents the models never saw** (`demo_runtime.py`, models from
`export.py runtime --exclude <agent>`, 20 complete evaluations per design, same seeds; the
3-seed rows use `--seeds 3` models and `demo_runtime.py --ensemble 3`). With full runs the
half-widths would be 0.040 (Sonnet) and 0.038 (gpt-oss-120b).

| Held-out agent (true rate) | Design | Models | Mean score (sd) | Interval contains truth | Half-width | Tokens spent | Compute for the reported width |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-4.5-sonnet-high (0.714) | standard | 1 seed | 0.716 (0.030) | 20 / 20 | 0.083 | 7.3% | 32% |
| | standard | 3 seeds | 0.715 (0.031) | 20 / 20 | 0.083 | 7.3% | 32% |
| | balanced | 1 seed | 0.716 (0.035) | 20 / 20 | 0.080 | 7.5% | 31% |
| | balanced | 3 seeds | 0.717 (0.029) | 20 / 20 | 0.078 | 7.6% | 29% |
| | balanced-unproven | 1 seed | 0.717 (0.035) | 19 / 20 | 0.077 | 7.5% | 28% |
| | balanced-unproven | 3 seeds | 0.717 (0.030) | 20 / 20 | 0.076 | 7.6% | 28% |
| gpt-oss-120b (0.260) | standard | 1 seed | 0.271 (0.073) | 19 / 20 | 0.135 | 9.5% | 118% |
| | standard | 3 seeds | 0.266 (0.069) | 19 / 20 | 0.133 | 9.7% | 116% |
| | balanced | 1 seed | 0.268 (0.045) | 20 / 20 | 0.114 | 9.5% | 84% |
| | balanced | 3 seeds | 0.264 (0.048) | 20 / 20 | 0.108 | 9.6% | 76% |
| | balanced-unproven | 1 seed | 0.266 (0.045) | 20 / 20 | 0.110 | 9.6% | 79% |
| | balanced-unproven | 3 seeds | 0.263 (0.045) | 20 / 20 | 0.105 | 9.7% | 72% |

- **Twenty evaluations are few.** The offline simulation (1,000 replicates, `results/balanced.csv`)
  gives 29% (standard), 28% (balanced) and 26% (balanced-unproven) for Sonnet, and 124%, 83% and
  76% for gpt-oss-120b.
- **The 3-seed models are the same or slightly cheaper.** The weaker agent gains most: its
  balanced design drops from 84% to 76% (`balanced-unproven`: 79% to 72%).
- **Under the standard design, gpt-oss-120b costs more than full runs.** The predictor, trained
  mostly on stronger agents, overrates it: its mean step-0 prediction is 0.62 against a true
  rate of 0.26. The balanced designs cancel that misjudgment within each stratum, and
  independent stopping cannot.
- **Agents above full-run cost.** Under the standard design 2 of 37 agents cost more than full
  runs for the reported width: Qwen2.5-Coder-32B (9% resolve rate) at 2.9x, and gpt-oss-120b at
  1.2x. Under the balanced designs only Qwen2.5-Coder-32B does, at 1.4x (`balanced`) and 1.3x
  (`balanced-unproven`).

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

## Run it

From the repo root, with the environment active (`source .venv/bin/activate`). On macOS,
`brew install libomp` is needed for LightGBM, and python.org Python needs
`export SSL_CERT_FILE=$(python -m certifi)` for the downloads. Data and models go to `data/`,
which is not committed; result tables go to `predictor/results/`, which is.

**1. SWE-bench Verified: data, features and the predictor**

```bash
python coverage/download_swebench.py                  # Vedanti's download, about 15 minutes
mkdir -p data/meta && python - <<'EOF'                # gold patches and tests for Verified
import pandas as pd
from huggingface_hub import HfFileSystem
src = "datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet"
cols = ["instance_id", "repo", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "created_at"]
pd.read_parquet(HfFileSystem().open(src), columns=cols).to_parquet("data/meta/swebench_verified.parquet")
EOF
python predictor/step_features.py                     # about 1 minute (10 workers by default)
python predictor/train_lgbm.py --features gold --split loao     # about 3 minutes
python predictor/evaluate.py data/predictor/predictions_gold_loao.pkl --name lgbm_gold_loao
python predictor/train_lgbm.py --features gold --split loao --seeds 0,1,2 --tag ens3   # optional: seed average
```

**2. The stopping rule and the three designs**

```bash
python predictor/cost_aware.py                        # learned remaining cost -> results/cost_aware.csv
python predictor/balanced.py --blocks 25,50 --replicates 1000   # every design -> results/balanced.csv
python predictor/balanced_check.py                    # the proof's bias formula on real runs -> results/balanced_check.csv
```

**3. The runtime, and predictions for Vedanti's plots**

```bash
python predictor/export.py runtime                    # models for a live harness -> data/predictor/runtime/all/
python predictor/export.py runtime --seeds 3          # optional: the seed-averaged models -> runtime/all_ens3/
python predictor/export.py runtime --exclude <agent>  # the same without one agent
python predictor/demo_runtime.py <agent>              # replay that agent (--design all for every design)
python predictor/export.py coverage --split loao      # predictions in Vedanti's format
python -I predictor/tool_ab_check.py                  # Rishi's tool A/B runs, unzipped into data/tool_ab/
```

**4. The second benchmark: the SWE-rebench leaderboard**

```bash
python predictor/convert_rebench_lb.py
python predictor/step_features.py --data data/rebench_lb --meta data/meta/rebench_lb_gold.parquet
python predictor/train_lgbm.py --data data/rebench_lb --out data/predictor/rebench_lb --features gold --split loao
python predictor/evaluate.py data/predictor/rebench_lb/predictions_gold_loao.pkl --name rebench_lgbm_gold_loao
python predictor/cost_aware.py --features data/rebench_lb/features_rich.pkl \
    --predictions data/predictor/rebench_lb/predictions_gold_loao.pkl --name cost_aware_rebench
python predictor/balanced.py --inputs data/predictor/cost_aware_rebench_inputs.pkl \
    --kappas predictor/results/cost_aware_rebench.csv --name balanced_rebench --blocks 50 --replicates 1000
python predictor/paired_rebench.py                    # A/B hints on real one-change pairs
```

**5. The OpenHands runs (used to measure luck)**

```bash
python predictor/convert_openhands.py nebius          # about 8 minutes, 1.6 GB written
python predictor/convert_openhands.py coderforge      # all three splits
python predictor/step_features.py --data data/openhands \
    --meta data/meta/swebench_verified.parquet data/meta/swerebench_gold.parquet \
    --out data/openhands/features_rich.pkl --workers 1      # about 10 minutes
```

The other scripts (the checks, the deep dive, the agent-adaptive predictor) give their command
at the top of the file.

## Files

| File | What it does |
| --- | --- |
| **Data** | |
| `convert_openhands.py` | Nebius / CoderForge OpenHands runs into the shared step format |
| `convert_rebench_lb.py` | SWE-rebench leaderboard runs (36 configurations, 5 runs per task) into the shared format |
| `step_features.py` | One row per step: Vedanti's counts (identical values), plus *process* features (last test result, steps since the last edit, reproduction scripts, repeated commands, format errors) and *gold* features (has the agent viewed or edited the files the reference patch touches, has it run the tests that must pass) |
| **Predictors** | |
| `train_lgbm.py` | LightGBM, one model per held-out agent; feature groups `base` / `process` / `gold`; splits `loao` / `lofo` / `temporal` / `insample` / `agentseen` / `agentseen_id`; `--data` for another benchmark; `--seeds 0,1,2` averages several seeds |
| `variance_weights.py` | Row weights that make the training loss equal to the estimator's variance (THEORY.md, Lemma 2) |
| `train_ability.py` | Ability-aware predictor: the agent's resolve rate is an input, on top of a uniform logit shift |
| `extra_data_check.py` | The held-out predictor with another dataset's runs added to training (OpenHands or SWE-rebench) |
| `leaderboard_rate_check.py` | Downloads 128 SWE-bench Verified leaderboard systems' per-task results and adds their solve rate as a feature |
| **Scoring** | |
| `evaluate.py` | Scores any predictions file with Vedanti's five stopping policies, each tuned on the other agents, plus how much the prediction improves through the run |
| `compare.py` | Paired comparison of predictors over the same held-out agents |
| **Stopping rule and designs** | |
| `allocation_check.py` | Premise check: what per-task, cost-aware allocation is worth (one-shot policies) |
| `cost_aware.py` | Learned remaining cost and error per held-out agent, fed to the stopping policy |
| `cost_models.py` | Which loss for the remaining-cost model (log squared error, Tweedie, gamma, smearing) |
| `balanced.py` | Every design side by side: independent or balanced stopping draws (each run with its own chance to start, or equal chances within a group), with the standard, design-based or proven interval |
| `balanced_check.py` | The proof's exact formula for the balanced interval's bias, checked on the 37 agents' real runs |
| `THEORY_BALANCED.md` | Why the `balanced` design's interval is valid (the proof), and what was wrong with the earlier one |
| **Analyses** | |
| `oracle_shift.py` | Bound: each agent's predictions shifted to average its true rate |
| `adaptive_eval.py` | Vedanti's in-evaluation correction (`coverage/adaptive.py`, unchanged) run on any predictions |
| `ability_eval.py` | Simulated evaluations with the ability-aware predictor, the rate estimated from earlier batches |
| `deep_dive.py` | Why the learned predictor did not beat LightGBM: cost decomposition, luck, timing, sensitivity |
| `probe_estimate.py` | Offline estimate of mid-run test probes (hidden tests run after the decisive edit) |
| `paired_check.py` | Premise check for tool A/B tests: a fully run reference arm as a predictor input |
| `paired_rebench.py` | A/B reference-arm check on the leaderboard's real one-change pairs |
| **Integration** | |
| `export.py` | Predictions for coverage/ (`coverage`) and deployable models for a live harness (`runtime`; `--seeds 3` for the seed-averaged models) |
| `runtime.py` | Run-time interface: predictions, cost-aware stopping, estimate and interval, under any design (`standard`, `balanced` or `balanced-unproven`) |
| `demo_runtime.py` | Replays a held-out agent's logged runs through `runtime.py` as a live harness would, under any design (`--ensemble 3` for the seed-averaged models) |
| `tool_ab_check.py` | Rishi's tool A/B runs (Qwen, 4 arms × 100 tasks) with the runtime's predictions, as trained and calibrated: prediction quality, each arm alone, controls, and each tool change against the baseline |

## Learned remaining cost: a trained component that beats the hand-made policies

The theory note and Check 3 (both below) show that training the success predictor on the
variance objective cannot beat a calibrated classifier. Theorem 1 says what else the optimal policy needs: the
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
true log remaining cost: median 0.62 over agents (0.38 to 0.78).

| Policy | Median | Geo. mean | vs tracking | Worst agent | Coverage, mean / worst |
| --- | --- | --- | --- | --- | --- |
| Tracking | 47.9% | 52.6% | | 4.46x | 94.4% / 91.8% |
| + learned error | 48.2% | 53.7% | 1.021x | 5.27x | 94.5% / 91.4% |
| **+ learned remaining cost** | **45.3%** | **48.9%** | **0.931x** | **3.02x** | **95.1% / 93.0%** |
| + both | 44.7% | 49.2% | 0.936x | 3.30x | 94.8% / 92.8% |
| + true remaining cost (bound) | 42.6% | 45.1% | 0.857x | 2.88x | 94.5% / 92.4% |

- **The learned remaining cost recovers about half of what a perfect cost oracle gives** and
  beats every hand-made policy (tracking 52.6% in the same simulation; in `evaluate.py`,
  tracking 53.0%, learned tracking 52.6%, index 54.1%, one-shot 56.6%), with coverage intact.
- **A learned error estimate does not help**, as the theory note predicts: f (1 - f) from a
  calibrated classifier is already the right uncertainty.

**Robustness.** The gain holds across training seeds and on the second benchmark:

| Run | Tracking | + learned remaining cost | Ratio | Agents cheaper | Coverage, mean / worst | True-cost bound |
| --- | --- | --- | --- | --- | --- | --- |
| Verified, seed 0 | 52.6% | 48.9% | 0.931x | 34/37 | 95.1% / 93.0% | 45.1% |
| Verified, seed 1 | 52.1% | 48.3% | 0.928x | 34/37 | 94.9% / 91.8% | 44.8% |
| Verified, seed 2 | 53.0% | 49.3% | 0.930x | 34/37 | 94.9% / 92.6% | 45.1% |
| SWE-rebench leaderboard (36 configurations) | 57.9% | 53.3% | 0.921x | 32/36 * | 94.8% / 93.0% | 50.2% |

\* 30 of the 34 configurations with a 20 to 80% resolve rate are cheaper (ratio 0.922x there).

- **Seeds move both policies together.** The ratio varies by 0.3 points across seeds, while the
  absolute levels move by 1.9 to 2.0%.
- **How much of the oracle it captures.** The learned cost model gets about half of what the
  true remaining cost would give on Verified (47 to 51% across seeds), and 59% on the
  leaderboard.

**The cost model's loss does not matter** (`cost_models.py`, Verified): squared error on
log(1 + r) gives 48.9%; Tweedie and gamma losses, which target E[r] directly, 49.1%; Duan's
smearing correction is a constant factor and cancels in the relative cost (48.9%). The gap to
the true-cost bound is the part of a run's length that the features cannot predict, not the
choice of loss.

**In the runtime**, this policy with independent stopping draws and Vedanti's interval is the
`standard` design (see "How to use it").

## Balanced stopping decisions and the three designs

`balanced.py`; tests in `tests/test_balanced.py`; the proof is in [THEORY_BALANCED.md](THEORY_BALANCED.md).

**Design.** Each run still stops at s = max{k : U < P_k}, where U is the run's one lottery
number. By Lemma 1 the estimate stays exactly unbiased as long as each run's U is uniformly
random and unrelated to how that run goes, so the numbers of *different* runs may depend on each
other. Balanced stopping uses that freedom, in four steps:

1. Before any run starts, sort the tasks by their step-1 reach probability P_1 (ties by f0), so
   similar tasks sit next to each other.
2. Cut them into groups (strata) of B = 50 different tasks, one run per task.
3. Within a group, split 0 to 1 into 50 equal slices and deal one slice to each task in random
   order, at a random spot inside it: U = (slot + V) / B with V ~ U(0, 1). This is a Latin
   hypercube, Vihola's (2018) stratified truncation.
4. The number of runs that start, and roughly the number that reach each later step, become
   nearly fixed within a group, so the group's average misjudgment cancels.

**Example.** Take a group of 50 tasks, each with P_1 = 20%, so about 10 should start.

- **Independent draws:** the count varies from one evaluation to the next, usually between 5
  and 15.
- **Balanced draws:** exactly 10 slices lie below 20%, so exactly 10 runs start, every time.
- **Still fair:** each run's own chance of starting is still exactly 20%.
- **More accurate:** if the predictor overrates the agent, every run that starts pulls the score
  down to correct it. With independent draws, luck in how many start becomes extra error;
  balanced draws remove it.

**Only a batch changes.** A single run behaves exactly the same under independent and balanced
draws; the gain comes from coordinating the runs of a batch. So the design needs the task list
before the first run starts, and enough tasks to form groups.

**Equal chances: the runtime's `balanced` design.** In `balanced-unproven` each run keeps its own
step-1 reach probability P_1, close to its neighbours' because the tasks are sorted by it. The
proven design gives every run of a group the same P_1 = m / 50, with m the group's sum of P_1
rounded (at least 3, because of the floor), and caps the later reach probabilities at it. Then
exactly m runs start in every group, and the runs that start are a simple random sample of the
group, which the proof needs. Equal chances cost almost nothing: actual compute 43.4% against
43.3% on Verified. When the number of tasks is not a multiple of 50, the last group also takes
the remainder.

The tests check that each group is a Latin hypercube, that every run's U is uniform
(Kolmogorov-Smirnov), that the estimates average to the true outcomes (20,000 Monte Carlo
draws), that exactly m runs start in every equal-chance group, and the proof's identities
(exactly, by enumerating every permutation of small groups).

**Intervals.** Under this design the usual sd / sqrt(n) still covers in simulation (96%), but it
is too cautious and does not get narrower: it cannot see the dealing, so it still counts the luck
the dealing removed. The design-based interval adds up two pieces:

- **The outcomes' own uncertainty**, there even with full runs: Var(Y) / n, which is
  θ̂ (1 − θ̂) / n because each outcome is 0 or 1.
- **The uncertainty added by stopping early.** Only runs that start carry a correction
  c = Ŷ − f0. This piece is measured from how much the started runs' corrections differ from
  each other within each group: the spread of c · (expected number started), shrunk by the
  finite-population factor 1 − m / B for m started runs out of B. A group with fewer than 2
  started runs is merged with the next.

The half-width is a Student's t quantile with Σ (m − 1) degrees of freedom times the square root
of the sum. The spread is measured within a group, so the group's average misjudgment does not
count toward the variance. Two cheaper variants under-cover slightly or barely help: dropping the
finite-population factor leaves the intervals as wide as the naive ones, and using the normal
quantile instead of t gives 94.2% coverage. A successive-difference estimator on group totals
(also in `balanced.py`) has too few degrees of freedom and counts the planned differences
between groups as noise: it needs 95.6% compute for its reported width, with 98.7% coverage.

**What the proof found, and the fix.** The noise from stopping early has two sources in a group:
*which* runs start, and *how far* each started run goes.

- **The finite-population factor 1 − m / B is right for the first.** Dealing removes most of
  that noise.
- **It also shrinks the second, which dealing does not remove.** So the interval above misses a
  share m / B of the later-step noise, about 15%.
- **How much that matters on real runs.** With the tasks fixed, its expected variance is 0.91 to
  1.16 times the true variance on the 37 Verified agents, below 1 for 35 of them
  (`balanced_check.py`, on equal-chance groups). It cannot be proven valid, because in some
  cases it is not. In the extreme where every run starts, it estimates no later-step noise at all.
- **The fix.** The `balanced` design's interval adds, for each started run, the run's own
  unbiased estimate of its later-step noise (`later_step_table`). It is weighted by
  γ = f + (m − f) / ((m − 1)(B − 1)), a little more than the share f = m / B of runs started.
- **What that guarantees.** With equal chances, its expected variance is at least the true
  variance for any set of runs (THEORY_BALANCED.md, Proposition B3): 1.005 to 1.263 times on the
  37 agents.
- **Coverage.** The interval then reaches at least 95% as the number of tasks grows (Theorem B4).

**Results** (gold LightGBM held out, policies tuned on the other agents, 1,000 replicates). The
reported column is the compute needed to reach the interval width that is actually reported,
relative to full runs.

| SWE-bench Verified (37 agents) | Actual | Reported | Coverage, mean / worst | Worst agent (actual) |
| --- | --- | --- | --- | --- |
| Tracking, independent stopping, sd / sqrt(n) (the baseline) | 53.3% | 51.3% | 94.4% / 91.9% | 4.79x |
| Learned remaining cost, independent, sd / sqrt(n) (`runtime.py` standard design) | 48.9% | 47.7% | 94.8% / 92.5% | 3.18x |
| Tracking, balanced, design-based interval | 45.3% | 46.5% | 94.7% / 92.1% | 1.43x |
| Tracking, equal chances, interval with later steps | 44.7% | 47.7% | 95.1% / 93.3% | 1.32x |
| Learned remaining cost, balanced, sd / sqrt(n) | 43.3% | 47.8% | 95.9% / 93.7% | 1.11x |
| Learned remaining cost, balanced, design-based interval (`runtime.py` balanced-unproven design) | 43.3% | 43.2% | 94.6% / 92.9% | 1.11x |
| **Learned remaining cost, equal chances, interval with later steps (`runtime.py` balanced design, proven)** | **43.4%** | **45.4%** | **95.2% / 93.6%** | **1.11x** |

| SWE-rebench leaderboard (36 configurations) | Actual | Reported | Coverage, mean / worst |
| --- | --- | --- | --- |
| Tracking, independent, sd / sqrt(n) | 58.2% | 57.0% | 94.7% / 92.4% |
| Learned remaining cost, independent, sd / sqrt(n) (standard design) | 53.3% | 52.4% | 94.9% / 93.1% |
| Learned remaining cost, balanced, sd / sqrt(n) | 46.7% | 52.5% | 96.0% / 94.3% |
| Learned remaining cost, balanced, design-based interval (balanced-unproven design) | 46.7% | 48.0% | 94.8% / 93.5% |
| **Learned remaining cost, equal chances, interval with later steps (balanced design, proven)** | **47.7%** | **49.8%** | **95.3% / 93.8%** |

The two improvements stack. Against the baseline, the proven `balanced` design cuts the compute
for a given reported interval width by 12% on Verified and 13% on the leaderboard: per-agent
geometric-mean ratios 0.884 (32 of 37 agents cheaper) and 0.874 (35 of 36). The earlier interval
gave 16% (0.842, with 37 of 37 cheaper; 0.843, with 35 of 36), but it is slightly too narrow.
Coverage is at or above the baseline's.

**Without the design-based interval.** Balanced stopping makes the score more accurate under
either interval: actual compute drops from 48.9% to 43.3% on Verified and from 53.3% to 46.7% on
the leaderboard. Only the design-based interval shows it. With sd / sqrt(n) the reported width
stays where the independent design's is (47.8% and 52.5%), and coverage rises to 96%, so the
interval is wider than it needs to be. `runtime.py` offers three designs:

- **`standard`** (the default): independent stopping and Vedanti's proven interval. About 7 to
  8% cheaper than the baseline for the reported width (geometric-mean ratio 0.929 on Verified,
  36 of 37 agents cheaper; 0.921 on the leaderboard, 33 of 36).
- **`balanced`**: balanced stopping with equal chances and the proven interval with later
  steps. 12 to 13% cheaper than the baseline, and 5% cheaper than `standard` (geometric-mean
  ratios 0.952 and 0.950; cheaper for 23 of 37 Verified agents and 24 of 36 leaderboard
  configurations).
- **`balanced-unproven`**: the earlier balanced design and interval, 16% cheaper on paper, kept
  for comparison.

**Who gains** (Verified, final against baseline, geometric-mean ratio by resolve rate):

| Resolve rate | Up to 30% | 30-50% | 50-65% | Above 65% |
| --- | --- | --- | --- | --- |
| Ratio, `balanced` | 0.44 | 0.71 | 0.97 | 0.92 |
| Ratio, `balanced-unproven` | 0.40 | 0.68 | 0.94 | 0.87 |
| Agents | 2 | 3 | 14 | 18 |

For the reported width, the worst agent on Verified drops from 4.3x to 1.4x full-run compute
under `balanced` (1.3x under `balanced-unproven`; 4.8x to 1.1x in actual compute, the last
column of the table). On the leaderboard the 0.7%-resolve configuration improves from 42.9x to
18.7x (16.4x) but stays far above full runs.

**Caveats.**

- **Group size.** For `balanced-unproven`, B = 25 is within a point of B = 50 (44.1% against
  43.2% reported compute), so the choice matters little. The equal-chance design is simulated
  with B = 50 only.
- **The design needs f0 and P_1 for the whole batch of tasks before the first run starts.** Both
  use only task features and the learned step-0 cost, so this works in a live evaluation.
- **Batch size (not yet measured).** The design-based interval is estimated from the runs that
  start in each group: with 500 tasks about 7 to 8 per group of 50, and a t multiplier of about
  2.0. With 50 tasks there is one group with about 8 started runs and a multiplier of about 2.4,
  20% wider than the fixed 1.96 of the standard interval, which would more than cancel the gain.
  A rough estimate puts the break-even at 150 to 250 tasks.
- **θ̂ (1 − θ̂) / n targets the population of tasks** (Theorem 2a's setting). For a fixed task
  set the interval is conservative.

## A second benchmark: the SWE-rebench leaderboard

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
  are 1 to 3% cheaper than base in geometric mean and no cheaper on the median: within training
  noise.
- The geometric mean is higher because of two very weak configurations; the worst (DeepSeek
  V3.2 without demonstrations, 0.7% resolved) costs 44 times full runs, the weak-agent problem
  in its extreme form: when almost every run fails, Var(Y) is tiny and any error dominates.

**The deep dive replicates** (agents with 20 to 80% resolve rate): variance shares 15.1%
outcome, 66.9% step-0 term, 17.9% in-run terms (Verified, all 37 agents: 14.8 / 67.6 / 17.6);
compute 49.8%, 16.3% with a perfect step-0 prediction, 40.8% with perfect in-run predictions;
11% of the step-0 uncertainty resolved one step before the end (Verified, all 37 agents: 13%);
the last edit to a gold
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

## How we got here

The sections below are the detailed record behind the results above, roughly in the order the
work was done: how the metric behaves and how much training noise there is, the checks on the
predictor and the gate decision they led to, the agent-adaptive predictor, the deep dive into why
a better-trained predictor did not help, the tests of more training data, the literature survey,
and the theory note.

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
(step-0 Brier 0.445), so it costs 5.2 times full runs.

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
training run shifts all of them together: seed 2 against seed 0 gives p = 0.004 (tracking) and
0.001 (index) for the same model. **Those p-values ignore training randomness, so differences under about 3%
between single training runs are noise**, whatever p says. Below, such differences are
reported as "no difference".

## Checks before building the learned predictor

The checks keep their numbers from the original plan and appear in the order they were run. The
gate decision after them summarizes the answers.

### Check 0: the pipeline reproduces Vedanti's numbers (passed)

`evaluate.py` on Vedanti's stand-in predictions gives the same median, geometric mean and
worst case as `coverage/results/policy_loao.csv` for all five policies (largest per-agent
difference 0.001, identical tuned settings). The base features from `step_features.py` match
`coverage/features.py` on all 786,050 rows.

### Check 5: what the external datasets contain (passed, with caveats)

From the dataset cards and the converted files:

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
  held-out evaluation uses Vedanti's 37 agents. The OpenHands data was meant for extra
  training and a transfer test; so far it is used to measure luck.
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

### Other datasets (survey)

What the project lacks most is many agents on the same tasks (held-out-agent tests), repeated
runs (luck), and pairs that differ by one change (A/B). All of these include failing runs;
existence, size, row counts and licence were checked against the Hugging Face API on Nov 20, 2026.

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

- **LightGBM is about 5% cheaper than the stand-in**: 4 to 6% for every feature set and per-step
  policy except base features under the index policy (3.3%), more than the seed spread.
- **Richer features improve the prediction during the run but not the cost.** Gold features
  triple the improvement by halfway (9.6% against 2.9%), yet compute is the same as with
  base features. The improvement arrives after most of the cost is spent, while the cost is
  set mostly by the step-0 error, which no in-run feature can change.
- **The early signal is far below what Vedanti's controlled experiment needed.** In
  `coverage/midrun_signal.py`, with a signal 90% accurate a quarter of the way in, the index
  policy beat learned tracking by 7 points (47% against 54%); with a 75%-accurate signal it did
  not beat it (50% against 48%). Here the best prediction a quarter of the way in is
  5% better than at step 0.

### Agent calibration is the larger lever (oracle check)

Every LightGBM predictor gives a mean step-0 prediction of about 0.6 for every agent (0.59 to
0.64; the stand-in about 0.57), from 9% to 77% true resolve rate: at step 0 nothing about the agent is known. `oracle_shift.py` moves
each agent's predictions on the logit scale so they average to its true rate. It is an
oracle, not a method (it uses the answer), and shows how much a correct agent-level
calibration is worth. Gold LightGBM, held-out agents:

| Policy | As trained, median | Oracle shift, median | Paired ratio | Agents cheaper | Worst agent |
| --- | --- | --- | --- | --- | --- |
| One-shot | 52.5% | 44.5% | 0.840x | 36/37 | 5.2x → 0.87x |
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
  spread), removes every agent from above full-run cost under the per-step policies (under
  one-shot the weakest stays at 1.1x), and keeps coverage at 95%.
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

## Gate decision

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

**What came next.** Two levers that do not need the agent's strength turned out to work: the
learned remaining cost (7 to 8% less compute) and balanced stopping decisions (12 to 13% with
the proven interval). Both are described near the top of this README.

## Agent-adaptive predictor

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
  estimate has variance about 1.7 on average, agents' rates have a standard deviation of about
  0.15, and 1.72 / 0.146² ≈ 81),
  which closes 80% of the gap to the agent-known bound. Coverage drops to 93.3%, and the
  misses are one-sided: the interval lies entirely above the truth twice as often as below
  (4.6% against 2.2%), on weak, middle and strong agents alike.

### Why a uniform shift skews the estimate

Exact skewness of a task's estimate at floor 0.05 (no simulation), median over agents:
static −2.7, oracle uniform shift −4.4 (+11.6 for the weakest agent), agent-known model −2.7,
although the agent-known model has the most runs at the floor (23% against 11% static).
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
  1.7 on average, so after 100 tasks theta is still uncertain by about ±0.13, almost as large
  as the spread between agents (0.15).
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

## Why the learned predictor did not beat LightGBM (deep dive)

`deep_dive.py` and `probe_estimate.py`; tables in `results/deep_dive_*.csv`. Policy-dependent
numbers use tracking at kappa 0.25 with a 0.05 floor, except the probe table at the end; compute
is the geometric mean over the 37 held-out agents.

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
| Which tasks this agent solves, beyond its strength and the task's difficulty | at least 0.046 | at least 36% | the rest |
| Luck: the same agent on the same task passing or failing | at most 0.052 | at most 41% | below (an upper bound) |

Luck is measured two ways. Where one agent ran each task several times, the run-to-run
variance is: Nebius 0.051 (about 10 runs per task; 20% of the outcome variance), CoderForge
SWE-smith tasks 0.037 (4 runs; 15%), SWE-rebench tasks 0.070 (8 runs; 29%), R2E-Gym tasks
0.076 (8 runs; 33%). On SWE-bench, two near-identical configurations of one model disagree on
a task with probability d, which bounds the luck by d / 2: 0.054 for gpt-5.2 default against
high effort, 0.050 for opus-4.5 across a scaffold version (0.06 to 0.13 for less similar
pairs). No predictor can remove the luck, so even with the agent known a step-0 prediction
keeps a Brier score equal to it: at most about 0.05 on SWE-bench Verified by the bound above,
and 0.04 to 0.08 where luck is measured directly.

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
| the last edit of any file | 89% (76-94%) | 91% (79-95%) |
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
  outcome with a given accuracy. This table is `evaluate.py` output, with each policy tuned on
  the other agents.

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
- **What was done next:** the learned remaining cost and balanced stopping decisions, near the
  top of this README. Neither needs a better step-0 prediction: the first spends tokens where
  they buy the most certainty, and the second cancels the step-0 misjudgment within each group.

## Does more training data help? (tested: no)

`extra_data_check.py` and `leaderboard_rate_check.py`; tables in `results/lgbm_gold_loao_<run>_*.csv`.

The deep dive says the predictor is limited by what can be known about a run, not by training.
These tests check that directly. Each repeats the baseline exactly (gold LightGBM, leave one agent
out, the same seed, validation agents and cross-fitted task rate) with one change, and scores the
same 37 held-out Verified agents with `evaluate.py`:

- **OpenHands runs** (Nebius and CoderForge, all from one model): 250K or 786K step rows of whole
  runs added to training. Their tasks have no other agents, so the task solve rate is left
  missing for them, and they are never used for early stopping.
- **Leaderboard solve rate**: the share of 128 SWE-bench Verified leaderboard systems that solved
  each task (from github.com/SWE-bench/experiments), added as a feature. Every mini-swe-agent
  submission is left out, so the held-out agent's own result never enters; none of the rest
  matches one of our agents on more than 91.6% of tasks. It correlates 0.92 with our agents' own
  solve rate.
- **SWE-rebench runs** (36 configurations): 250K or 786K step rows added, with the solve rate
  cross-fitted across its configurations, exactly as for Verified.

Retraining the unchanged baseline with seeds 1 and 2 shows the noise: compute moves by up to 1.8%
per policy, and 11 to 26 of the 37 agents come out cheaper by chance.

| Training data added | Step-0 Brier | Compute vs baseline (range over the four policies) | Verdict |
| --- | --- | --- | --- |
| None (baseline, seeds 0 to 2) | 0.1260 to 0.1278 | 0.984x to 1.018x | The noise yardstick |
| OpenHands, 250K / 786K rows | 0.1274 / 0.1276 | 0.998x to 1.007x | No gain; the starting prediction is slightly worse for 29 of 37 agents |
| Leaderboard solve rate | 0.1265 | 0.998x to 1.002x | No gain |
| SWE-rebench, 250K / 786K rows | 0.1261 / 0.1263 | 0.989x to 0.995x | A small gain at seed 0, not confirmed below |

**Seed check for SWE-rebench** (786K rows, compute against the baseline trained with the same seed):

| Same-seed pair | Step-0 Brier, baseline → with SWE-rebench | One-shot | Tracking | Learned tracking | Index | Worst agent (tracking) |
| --- | --- | --- | --- | --- | --- | --- |
| Seed 0 | 0.1267 → 0.1263 | 0.993x | 0.993x | 0.989x | 0.990x | 4.71x → 4.70x |
| Seed 1 | 0.1260 → 0.1260 | 0.992x | 1.011x | 0.994x | 1.002x | 3.86x → 4.38x |
| Seed 2 | 0.1278 → 0.1267 | 0.989x | 0.983x | 0.986x | 0.985x | 4.89x → 4.92x |
| Average of the 3 seeds | 0.1260 → 0.1259 | 0.994x | 1.008x | 0.988x | 0.995x | 4.40x → 4.67x |

- **OpenHands adds nothing.** More runs from one other agent, on unrelated tasks with no solve
  rate, slightly blunt the starting prediction.
- **The leaderboard solve rate adds nothing**, although it comes from 128 systems instead of 36
  agents. Task difficulty is already measured well enough; what remains of the starting error is
  which tasks this agent solves, and luck.
- **SWE-rebench runs give no reliable gain.** One-shot and learned tracking are 0.6 to 1.4%
  cheaper at every seed, but tracking and index flip between seeds, the averaged models are 0.8%
  costlier under tracking, and the worst agent gets worse.
- **Averaging seeds is the one reliable training gain.** Averaging the predictions of models
  trained with seeds 0, 1 and 2 gives step-0 Brier 0.1260; for squared error the average can
  never be worse than the typical single run. Under Vedanti's policies it needs 0.6 to 1.7% less
  compute than the mean of the three single runs (one-shot 0.6%, learned tracking 0.9%, index
  1.4%, tracking 1.7%), and it removes the seed lottery.

**The seed average as a setup.** `train_lgbm.py --seeds 0,1,2` (offline) and
`export.py runtime --seeds 3` (runtime) make it an option next to the single-seed models, and it
works with any design. Only the success predictor is averaged; the remaining-cost model stays a
single seed. On SWE-bench Verified (37 held-out agents, 1,000 replicates) it lowers every setup by
about 1%, with coverage unchanged:

| Setup (compute for the reported width) | Single seed | 3-seed average | Agents cheaper | Worst agent |
| --- | --- | --- | --- | --- |
| Baseline: LightGBM + tracking | 51.3% | 50.8% | 27 of 37 | 4.30x → 4.04x |
| Standard design (learned remaining cost) | 47.7% | 47.1% | 25 of 37 | 2.95x → 2.73x |
| Balanced design (proven) | 45.4% | 44.7% | 27 of 37 | 1.39x → 1.26x |
| Balanced-unproven design | 43.2% | 42.7% | 28 of 37 | 1.27x → 1.15x |

The replays of two held-out agents (under "How to use it") agree. It has not been run on the
SWE-rebench leaderboard.

```bash
python predictor/extra_data_check.py --source openhands --rows 250000 --tag oh250k   # and 786000 / oh786k
python predictor/extra_data_check.py --source rebench --rows 786000 --tag rb786k     # seed check: --seed 1 or 2
python predictor/leaderboard_rate_check.py fetch
python predictor/leaderboard_rate_check.py train
python predictor/evaluate.py data/predictor/predictions_gold_loao_<tag>.pkl --name lgbm_gold_loao_<tag>
python predictor/train_lgbm.py --features gold --split loao --seeds 0,1,2 --tag ens3    # the seed average
python predictor/cost_aware.py --predictions data/predictor/predictions_gold_loao_ens3.pkl --name cost_aware_ens3
python predictor/balanced.py --inputs data/predictor/cost_aware_ens3_inputs.pkl \
    --kappas predictor/results/cost_aware_ens3.csv --name balanced_ens3 --blocks 50 --replicates 1000
```

## Literature survey: Monte Carlo methods and novelty

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
   Tillé 2004; stratified truncation levels, Vihola 2018). Now done: see "Balanced stopping
   decisions and the three designs".
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

This beats the oracle per-agent shift (one-shot on Verified: 47.5% geometric mean, 44.5% median)
without knowing the agent.
The two steps this check left open are now done (see "Balanced stopping decisions and the three
designs"): the per-step version deals each run's stopping number within groups of similar
tasks, and a design-based variance estimator gives the matching interval. Its proof
(THEORY_BALANCED.md) required one fix to the estimator, and still needs a second reader.

## Theory note: the variance objective aims at the same predictor as log loss

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

- **Offline only.** Every number comes from replaying logged runs; live runs are the real test.
- **The balanced design's proof needs a second reader.** THEORY_BALANCED.md proves the
  `balanced` interval from standard results and every identity is checked numerically, but
  nobody has yet read it step by step. `runtime.py` defaults to the standard design, whose
  interval Theorem 2a proves.
- **The balanced design needs a batch.** It needs the task list before the first run starts and,
  by a rough estimate not yet measured, at least 150 to 250 tasks for its interval to beat the
  standard one. Every design, as built, takes the task list up front (`standard` only to scale
  costs), so tasks that arrive one at a time are not supported yet.
- **Weak or misjudged agents are still expensive.** On Verified, 2 of 37 agents cost more than
  full runs under `standard` and 1 under either balanced design; on the leaderboard the
  0.7%-resolve configuration costs 40x full runs under `standard`, 19x under `balanced` and 16x
  under `balanced-unproven`.
- **Some token counts are estimated:** the OpenHands runs log none, and the leaderboard logs
  only a total per run.
- Policy tuning uses predictions for the training agents that came from models which saw
  the held-out agent (the same caveat as in `coverage/README.md`).
- Most comparisons are one training run. The seed spread (up to 3.4%) is measured for gold
  LightGBM, and the learned remaining cost was rerun with 3 seeds.
- Coverage numbers from `evaluate.py` and `cost_aware.py` use 500 replicates (about ±1 point per
  agent); `balanced.py` uses 1,000 and the agent-adaptive simulations 1,500. Vedanti's
  `experiments.py` uses 2,000.
- Some numbers come from one-off analyses with no script in the repo yet: the premise check for
  balanced decisions (in the literature survey), `results/rebench_lb_luck.csv`, the CoderForge
  luck values, the early-steps table under "Agent calibration is the larger lever (oracle check)", and the
  skewness figures under "Why a uniform shift skews the estimate". An independent recomputation
  gives the same conclusions (the early-steps errors differ by up to 0.03).
- Inside an evaluation, agent adaptation recovers about 10% (51.9% → 46.4% at floor 0.05)
  of the 12 to 20% the bounds promise (45.5% agent known, 41.6% ability-aware with the true
  rate), at a cost of about 1.5 points of coverage.
- The paired checks use step-0 predictions and the one-shot formula only, and the pairs are
  effort, scaffold and version changes, not tool changes; Rishi's runs are the real test.
- The leaderboard's multilingual subset (Go, Java, Rust, TypeScript) is not converted: the
  features' test-run patterns are Python-specific.
- Nebius and CoderForge were tested as extra training data and did not help (see "Does more
  training data help?"); otherwise they are used to measure luck.
- No result yet on Rishi's runs.
