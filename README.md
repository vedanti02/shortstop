# shortstop

Unbiased randomized truncation for cost-efficient agent evaluation and tool A/B testing.

Early-stopped agent evaluations give wrong confidence intervals. This project stops runs early with a randomized ("Russian roulette") rule and reweights what was observed, so the score estimate stays exactly unbiased whatever the quality of the step predictor.

## Setup

```bash
git clone <repo-url> && cd shortstop
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
```

Extras: `pip install -e ".[dev,predictor]"` adds LightGBM (on macOS, run `brew install libomp` first).

## Layout

| Path | Owner | What goes here |
| --- | --- | --- |
| `src/shortstop/` | shared | The day-one contract: trajectory format, predictor interface, reference estimator |
| `coverage/` | Vedanti | The finding, the estimator, confidence intervals, coverage and cost plots |
| `predictor/` | Deepika | The learned step predictor, compared against LightGBM |
| `tool_ab/` | Rishi | Tool A/B agent runs and the mini-swe-agent plug-in |
| `data/` | everyone | Local trajectories and outputs. Not committed |
| `tests/` | shared | Tests for the shared contract |

Each slice is end to end with its own data and its own stand-ins for the other pieces. Pieces are combined only at integration (Nov 28 to 30).

## Shared contract

**Trajectory format.** One JSON line per step:

```json
{"task_id": "django__django-11099", "agent": "mini-swe-agent", "step": 1, "tokens_in": 1840, "tokens_out": 212, "action": "...", "observation": "...", "outcome": 1}
```

`step` starts at 1. `outcome` is the run's final result (0 or 1), repeated on every line. Load with `shortstop.load_trajectories(path)`.

**Predictor interface.** A function that takes a trajectory prefix (the list of steps so far, empty before step 1) and returns a success probability between 0 and 1. See `shortstop.Predictor`.

**Estimator.** With predictions `f_0 ... f_{T-1}`, final outcome `f_T = Y`, and continue probability `p_k > 0` before step `k`, a run stopped after step `s` is scored as

```
Y_hat = f_0 + sum_{k=1..s} (f_k - f_{k-1}) / (p_1 * ... * p_k)
```

A reference implementation is in `src/shortstop/estimator.py`.

## Working together

- Branch off `main`, open a pull request, and keep changes inside your own folder where you can.
- Changes to `src/shortstop/` affect everyone, so flag them to the team first.
- Keep data out of git. Put it in `data/` and note in your folder's README where it came from.

## Timeline

- Week of Nov 16: learned predictor, cost-optimal stopping policy, start tool A/B runs
- Week of Nov 23: finish tool A/B experiments, package the plug-in
- Nov 28 to 30: integration
- Dec 1: poster and public repo
