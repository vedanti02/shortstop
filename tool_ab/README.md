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
four arms, no infrastructure errors left after one retry, every patch scored.

| Change | Baseline | With change | Difference (95% CI) | Wins / losses | McNemar p | Tokens | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| descriptions | 40% | 26% | -14 (-23 to -5) | 5 / 19 | 0.007 | 1.27x | worse |
| search | 40% | 34% | -6 (-14 to +2) | 6 / 12 | 0.238 | 1.12x | no detectable difference |
| compressed | 40% | 32% | -8 (-15 to -1) | 3 / 11 | 0.057 | 0.93x | worse by the CI, borderline by McNemar |

None of the three tool changes helps this model; longer tool descriptions clearly hurt it.
The whole run cost about $44 of Modal compute (about 15 h; the GPU server is two thirds of it).
Trajectories in the shared format (`qwen-*.jsonl`) stay out of git, like all run data.

## The three tool changes

Every arm runs the same Qwen model, sampling settings, step limit and tasks. Each arm changes one
thing; `tests/test_configs.py` checks that no other setting differs from the baseline.

| Arm | Change | Where |
| --- | --- | --- |
| `baseline` | mini-swe-agent's SWE-bench prompt and bash tool, unchanged | `base.yaml` |
| `descriptions` | Detailed bash tool description: subshell semantics, timeout, output limits, how to view and edit files | `descriptions.yaml`, `shortstop_plugin/model.py` |
| `search` | A `search` command installed in the sandbox (grouped, capped matches; file-name search), documented in the prompt | `search.yaml`, `shortstop_plugin/scripts/search` |
| `compressed` | Outputs cleaned (ANSI, progress bars, repeated lines) and cut to 4,000 characters, head and tail kept | `compressed.yaml`, `shortstop_plugin/compress.py` |

One baseline arm serves all three comparisons, so the experiment is 4 arms, not 6.

## The plug-in

`shortstop_plugin.agent.StoppingAgent` is a mini-swe-agent agent. Before step k it calls the
predictor on the steps so far to get f_{k-1}, calls the stop policy to get p_k, and runs step k
with probability p_k. Each step gets exactly one draw, even if the model's reply fails to parse
and is retried. Policies and predictors are named by dotted path in the config:

```yaml
agent:
  stop_policy: shortstop_plugin.policies.never_stop    # the stand-in; also constant, tracking
  stop_policy_kwargs: {}
  predictor: shortstop.predictor.constant_predictor    # swap in Deepika's predictor here
  predictor_kwargs: {p: 0.5}
  seed: 0                                              # draws depend only on (seed, task)
```

Each trajectory saves `info.shortstop` with `predictions` (f_0, f_1, ...), `continue_probs`
(p_1, p_2, ...), `steps_run` (s) and `stopped_early`. That is everything `rr_estimate(f, p, s)`
needs; for a full run, f_T is the evaluated outcome. A policy gets `(prefix, predictions)` and
returns p_k in (0, 1]; anything else stops the run with an error.

- No draw is made for a step that cannot happen. When the step or cost limit is about to end the
  run, the agent ends it without drawing, so a full run of T steps has exactly T entries in
  `continue_probs`.
- Predictors and policies are built once per config and shared by all worker threads, so a learned
  predictor is loaded once. It must be safe to call from several threads.

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

   `convert.py` accepts any report with a `resolved_ids` list, or `{task: {"resolved": bool}}`.

6. **Rerun infrastructure failures.** A run that ended because a sandbox failed to start or the model
   server dropped says nothing about the tool change. A sandbox that never started leaves an entry in
   `preds.json` but no trajectory; those runs count as errors too (`NoTrajectory`). `convert.py` leaves it out (listed in
   `qwen-<arm>.errors.json`) instead of scoring it as a failure. Rerun those tasks with
   `python tool_ab/run_arm.py --arm <arm> --slice 0:100 --retry-errors`. Agent endings (submitted, step
   limit, repeated format errors, context overflow) are real results; see `shortstop_plugin/status.py`.

## Notes for the team

- **Sandbox commands run in bash with conda `testbed` active.** mini-swe-agent's `swerex_modal`
  environment runs commands under `/bin/sh` and replaces the sandbox's environment, so `BASH_ENV` is
  ignored and `python` is the system interpreter. `shortstop_plugin/environment.py` fixes this, and
  `run_arm.py` uses it for `environment_class: swerex_modal`. Plain `mini-extra swebench` with
  our configs does not get the fix.
- **Sandboxes are terminated after each run.** Upstream, neither the batch runner nor swe-rex 1.4.0
  terminates a live Modal sandbox, so each one would keep billing until `runtime_timeout` (2 h).
  `run_arm.py` and `environment.py` handle this. To check for leftovers after a crash:
  `python -c "import modal; a = modal.App.lookup('swe-rex'); print(list(modal.Sandbox.list(app_id=a.app_id)))"`.
- **Stop records live beside the trajectories.** The shared format has no field for p_k, so
  `convert.py` writes `qwen-<arm>.stops.json` next to each JSONL. If others need p_k too, it could move
  into `src/shortstop/` (to discuss before changing the contract).
- **Outcome -1 means not evaluated.** A run the plug-in stopped early has no final result; `convert.py`
  writes its outcome as -1. Full runs (everything in the tool A/B experiment) are always 0 or 1.
- **Same step definition as `coverage/`.** `shortstop_plugin/steps.py` counts steps and tokens like
  `to_steps` in `coverage/download_swebench.py`: one model call plus its observations.
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
