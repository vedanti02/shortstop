# tool_ab (Rishi)

Tool A/B agent runs and the mini-swe-agent plug-in.

- Data: new paired runs for 3 tool changes
- Stand-in: plug-in with a placeholder stop policy (never stop)
- Done when: full-run verdicts are computed and the plug-in runs end to end

## What is here

| Path | What it does |
| --- | --- |
| `shortstop_plugin/` | The plug-in: `StoppingAgent` (continue-or-stop hook), tool variants, policies |
| `configs/base.yaml` | Baseline arm: mini-swe-agent's SWE-bench config with Qwen on Modal |
| `configs/{descriptions,search,compressed}.yaml` | One overlay per tool change |
| `configs/tasks.txt` | All 500 SWE-bench Verified tasks in a fixed shuffled order (seed 0) |
| `serving/serve_qwen.py` | Qwen3-Coder-30B-A3B (FP8) on vLLM on a Modal GPU, behind an API key |
| `serving/check_endpoint.py` | One request to confirm the endpoint returns parsed bash tool calls |
| `run_arm.py` | Runs one arm on a slice of `tasks.txt` with Modal sandboxes |
| `evaluate.py` | Scores predictions on SWE-bench Verified in Modal sandboxes |
| `convert.py` | Runs plus evaluation report to the shared JSONL format |
| `analyze.py` | Paired full-run verdict per tool change, written to `results/` |
| `tasks.py` | Regenerates `configs/tasks.txt` |
| `remote/full_run.py` | Runs every arm, scoring and analysis on Modal, so the laptop can be closed |
| `remote/watchdog.py` | Spending backstop: stops the run, its sandboxes and the GPU server at a deadline |
| `results/` | Verdicts from the full run (below) |

## Results

Full run, 2026-11-21: the first 100 tasks of `configs/tasks.txt`, Qwen3-Coder-30B-A3B (FP8), all
four arms, every patch scored. 10 of the 400 runs lost their sandbox connection partway
("Cannot connect to host", "Connection reset by peer"; 1 baseline, 2 descriptions, 4 search,
3 compressed). The run code at the time showed those failures to the agent instead of ending
the run, so they are now classed as infrastructure errors and left out until rerun; each
comparison uses the tasks both arms have.

| Change | Tasks | Baseline | With change | Difference (95% CI) | Wins / losses | McNemar p | Tokens | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| descriptions | 97 | 40.2% | 26.8% | -13.4 (-22.7 to -4.1) | 5 / 18 | 0.011 | 1.25x | worse |
| search | 95 | 41.1% | 35.8% | -5.3 (-13.7 to +3.2) | 6 / 11 | 0.332 | 1.15x | no detectable difference |
| compressed | 96 | 39.6% | 31.2% | -8.3 (-15.6 to -1.0) | 3 / 11 | 0.057 | 0.92x | borderline worse |

A verdict is firm when the bootstrap interval excludes zero and McNemar p < 0.05, and
borderline when only one of them does. None of the three tool changes helps this model;
longer tool descriptions clearly hurt it. Tokens include failed model calls (see
`shortstop_plugin/steps.py`). The whole run cost about $44 of Modal compute (about 15 h; the
GPU server is two thirds of it). Trajectories in the shared format (`qwen-*.jsonl`) stay out
of git, like all run data.

## The three tool changes

Every arm runs the same Qwen model, sampling settings, step limit and tasks. Each arm changes one
thing; `tests/test_configs.py` checks that no other setting differs from the baseline.

| Arm | Change | Where |
| --- | --- | --- |
| `baseline` | mini-swe-agent's SWE-bench prompt and bash tool, unchanged | `base.yaml` |
| `descriptions` | Detailed bash tool description: subshell semantics, timeout, output limits, how to view and edit files | `descriptions.yaml`, `shortstop_plugin/model.py` |
| `search` | A `search` command installed in the sandbox (grouped, capped matches; file-name search), documented in the prompt | `search.yaml`, `shortstop_plugin/scripts/search` |
| `compressed` | Every output rewritten, short ones too: ANSI codes stripped, progress bars kept at their final state, CRLF endings dropped, lines over 300 characters cut, runs of blank lines collapsed, repeated lines replaced with a marker; then anything still over 4,000 characters keeps its head and tail | `compressed.yaml`, `shortstop_plugin/compress.py` |

One baseline arm serves all three comparisons, so the experiment is 4 arms, not 6.

## The plug-in

`shortstop_plugin.agent.StoppingAgent` is a mini-swe-agent agent. Before step k it calls the
predictor on the steps so far to get f_{k-1} and the stop policy to get p_k. It draws one
uniform number u per task and runs step k while u < p_1 * ... * p_k, so each step still runs
with probability p_k given the earlier ones, and arms with the same seed stop at matching
points as often as possible. Each step is decided once, even if the model's reply fails to
parse and is retried. Policies and predictors are named by dotted path in the config:

```yaml
agent:
  stop_policy: shortstop_plugin.policies.never_stop    # the stand-in; also constant, tracking
  stop_policy_kwargs: {}
  predictor: shortstop.predictor.constant_predictor    # swap in Deepika's predictor here
  predictor_kwargs: {p: 0.5}
  seed: 0                                              # draws depend only on (seed, task)
```

Each trajectory saves `info.shortstop` with `predictions` (f_0, f_1, ...), `continue_probs`
(p_1, p_2, ...), `s`, `steps_run`, `stopped_early`, and the seed, agent name, policy and
predictor settings. For `rr_estimate(f, p, s)`:

```python
s = rec["s"]                                   # = len(continue_probs) - stopped_early
f = rec["predictions"][: s + 1] if rec["stopped_early"] else rec["predictions"][:s] + [Y]
```

`s` can be one more than `steps_run`: a run that ends inside a step after its draw (context
overflow, repeated format errors, a format error on the last allowed call) counts that
attempted step as the one that reveals Y.

- A predictor or policy that takes a `task_id` keyword gets the task ID; before step 1 the prefix
  is empty, so that is all the first prediction can go on. (Provisional until the shared
  `Predictor` type in `src/shortstop` takes it.)
- A policy gets `(prefix, predictions)` and returns p_k in (0, 1]; a predictor returns f in [0, 1].
  Anything else, or an exception, raises `PredictorError`: `run_arm.py` starts no new tasks and
  exits with an error, and the run is not retried as an infrastructure error.
- No draw is made for a step that cannot happen. When the step or cost limit is about to end the
  run, the agent ends it without drawing, so a full run of T steps has exactly T entries in
  `continue_probs`.
- Predictors and policies are built once per config and shared by all worker threads, so a learned
  predictor is loaded once, by `run_arm.py` before any sandbox starts. It must be safe to call from
  several threads.
- With `save_diffs: true` (on in `base.yaml`) the agent saves the repository diff after each step
  under `info.step_diffs`, built in a throwaway git index so neither the repository nor the model
  sees it. `convert.py` writes them to `qwen-<arm>.diffs.jsonl`. A diff equal to the previous one
  is saved as `"unchanged": true`, diffs are cut at 50,000 characters, and the submission step has
  no row (its diff is the submitted patch in `preds.json`).

## Running it

All commands from the repo root with the environment active. Install the extras once:

```bash
pip install "mini-swe-agent[modal]==2.4.6" "swe-rex==1.4.0" python-dotenv  # pinned: environment.py overrides mini-swe-agent internals
```

1. **Serve the model** (uses Modal GPU credits while running):

   ```bash
   modal secret create shortstop-vllm-key VLLM_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   modal deploy tool_ab/serving/serve_qwen.py
   cp tool_ab/.env.example tool_ab/.env    # paste the URL + /v1 and the same key
   python tool_ab/serving/check_endpoint.py
   ```

2. **Pilot: 5 tasks, baseline only.** Checks image builds, tool calls and timing before spending more.

   ```bash
   python tool_ab/run_arm.py --arm baseline --slice 0:5 -w 5
   ```

3. **Early batch for Deepika: 20 tasks.** Convert after evaluation (step 5) and send her
   `data/tool_ab/qwen-baseline.jsonl`.

   ```bash
   python tool_ab/run_arm.py --arm baseline --slice 0:20 -w 10
   ```

4. **All arms on the same tasks.** Raise the slice as the budget allows; finished tasks are skipped.

   ```bash
   for arm in baseline descriptions search compressed; do
     python tool_ab/run_arm.py --arm $arm --slice 0:100 -w 16
   done
   modal app stop shortstop-qwen
   ```

5. **Evaluate, convert, analyze.** `evaluate.py` scores predictions in Modal sandboxes (CPU only)
   with swebench's eval scripts and grading. swebench's own `--modal true` path is broken with
   modal 1.x in both 4.1.0 and 5.0.2, so use this instead (needs `pip install "swebench==4.1.0"`):

   ```bash
   python tool_ab/evaluate.py --preds data/tool_ab/runs/baseline/preds.json --run-id baseline -w 8
   python tool_ab/convert.py --arm baseline --report data/tool_ab/eval/baseline.json
   python tool_ab/analyze.py
   ```

   `convert.py` accepts evaluate.py's report, the SWE-bench harness report, or
   `{task: {"resolved": bool}}`. A task the report did not score (a grading error, or missing) is
   left out as `NotScored`, not counted as unresolved; `evaluate.py` exits with an error when that
   happens.

6. **Rerun infrastructure failures.** A run that ended because a sandbox failed to start, lost its
   connection, or the model server dropped says nothing about the tool change. A sandbox that never
   started leaves an entry in `preds.json` but no trajectory (`NoTrajectory`); a lost sandbox
   connection ends the run with `SandboxConnectionError`, and older runs in which any command lost
   the connection count the same way. `convert.py` leaves these out (listed in
   `qwen-<arm>.errors.json`) instead of scoring them as failures. Rerun those tasks with
   `python tool_ab/run_arm.py --arm <arm> --slice 0:100 --retry-errors`. Agent endings (submitted, step
   limit, repeated format errors, context overflow) are real results; see `shortstop_plugin/status.py`.

## Notes for the team

- **Sandbox commands run in bash with conda `testbed` active.** mini-swe-agent's `swerex_modal`
  environment runs commands under `/bin/sh` and replaces the sandbox's environment, so `BASH_ENV` is
  ignored and `python` is the system interpreter. `shortstop_plugin/environment.py` fixes this, and
  `run_arm.py` uses it for `environment_class: swerex_modal`. Plain `mini-extra swebench` with
  our configs does not get the fix.
- **Sandboxes are terminated after each run.** Upstream, neither the batch runner nor swe-rex 1.4.0
  terminates a live Modal sandbox, so each one would keep billing until `deployment_timeout` (the
  sandbox lifetime, 2 h; `runtime_timeout` is only the HTTP timeout of each call).
  `run_arm.py` and `environment.py` handle this. To check for leftovers after a crash:
  `python -c "import modal; a = modal.App.lookup('swe-rex'); print(list(modal.Sandbox.list(app_id=a.app_id)))"`.
- **Stop records live beside the trajectories.** The shared format has no field for predictions or
  p_k, so `qwen-<arm>.stops.json` next to each JSONL is the agreed place for them.
- **Outcome -1 means "stopped early, not evaluated".** A run the plug-in stopped early has no final
  result; `convert.py` writes its outcome as -1. Infrastructure errors are not -1: they are left out
  of the JSONL and listed in `qwen-<arm>.errors.json`. Full runs are always 0 or 1.
- **Same step code as `coverage/`, different logs.** `shortstop_plugin/steps.py` counts steps like
  `to_steps` in `coverage/download_swebench.py`: one model call plus its observations. But
  mini-swe-agent 1.17.5 (the downloaded runs) logged a reply that failed to parse as a step, and
  2.4.6 logs it only as an error message. Here such a call is not a step, and its tokens go to the
  next step, so per-step counts are not directly comparable between the two.
- **Departures from upstream mini-swe-agent:** step limit 100 instead of 250 (bounds compute), cost
  and wall-time limits off (a self-hosted model has no price, and run speed depends on server load,
  not on the tool change), and the runner swaps in the plug-in agent because mini-swe-agent's batch
  runner ignores `agent_class`.
- **`search` is checked on both platforms.** The tests run it with the macOS tools, and the same
  cases give identical output in `ubuntu:22.04` (GNU grep, mawk), which is what the SWE-bench
  images use.

## Tests

```bash
pytest tool_ab/tests        # no network or API calls
```

The agent tests run the plug-in end to end with mini-swe-agent's scripted test model in a local
shell, and `test_cli.py` drives `run_arm.py` with the SWE-bench runner mocked.
