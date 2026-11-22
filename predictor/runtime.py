"""Run-time interface for a live evaluation harness (e.g. Rishi's mini-swe-agent plug-in).

    rt = Runtime.load("data/predictor/runtime/all")       # made by predictor/export.py
    rt = rt.calibrate(earlier)    # optional: {task_id: 0 or 1} from earlier full runs of this agent
    plan = rt.plan(task_ids, seed=0)                       # before any run starts
    for task_id in task_ids:
        run = plan.start(task_id)
        while run.should_continue():                       # decided before every agent step
            step = ...                                     # the harness takes one agent step
            run.observe(step)                              # a shortstop.Step, or a dict of its fields
            if the run ended (the agent submitted, or a step or cost limit was reached):
                run.finish(outcome)                        # 0 or 1, from the benchmark's tests
                break
    result = plan.result()     # unbiased score, 95% interval, tokens spent, runs started

A run can also end during a step that never completes (a context overflow, a crash): call
run.finish(outcome) without observe(). The attempted step then counts as the one that reveals
the outcome; overwriting the last prediction instead would bias the estimate. result() refuses a
batch in which a started run was neither stopped nor finished.

Two interchangeable designs. They share the predictor, the cost model and the stopping policy,
and differ only in how each run's stopping uniform is drawn and which interval is reported:

    rt.plan(task_ids, design="standard")    the default
        Independent uniforms, and Vedanti's interval mean +/- 1.96 sd / sqrt(n)
        (coverage/experiments.py). Proven: coverage/THEORY.md, Theorem 2a.
    rt.plan(task_ids, design="balanced")
        Uniforms spread evenly within strata of runs with similar step-1 reach probability, and
        the design-based interval of predictor/balanced.py. A narrower interval for the same
        tokens; validated by simulation on two benchmarks, but not yet proven.

Either way every run's estimate is exactly unbiased (coverage/THEORY.md, Lemma 1), so the
score is unbiased under both designs; only the interval differs.

Either design runs with either set of models: `export.py runtime` saves one success model
(data/predictor/runtime/all), `export.py runtime --seeds 3` saves three trained with different
seeds (data/predictor/runtime/all_ens3), whose predictions are averaged. Both load the same way.

Agent calibration (optional). The models expect an agent like the training agents: their mean
step-0 prediction is about 0.6 whatever the agent. rt.calibrate(outcomes) returns a copy that
moves every success prediction by one logit shift, chosen so that the mean step-0 prediction on
those tasks equals the agent's solve rate on them. Use outcomes of earlier full runs of the same
agent (other tasks than the batch are fine) and calibrate before the batch starts: the shift is
then part of the predictor, so every run's estimate stays exactly unbiased. On Rishi's Qwen runs
it cut compute from 84-170% to 49-65% of full runs, with coverage at 94-96%
(predictor/tool_ab_check.py).
For a tool A/B test, give both arms the same shift (the baseline's) and the same seed, so their
step-0 predictions and stopping draws match.

What it does, per predictor/README.md:
  - f_k: the success prediction after k steps (gold LightGBM), moved by the calibration shift.
  - r_k: the expected tokens still to be spent (remaining-cost LightGBM), relative to the batch's
    mean at step 0.
  - The reach probability for the next step is min(previous, clip(kappa sqrt(f (1 - f) / r),
    floor, 1)): the learned-remaining-cost policy.
  - result() averages the randomized-truncation estimates (shortstop.rr_estimate) and gives the
    design's interval.

The models are trained on SWE-bench Verified agents. Task features (how often those agents solved
the task, the gold patch) exist for Verified tasks; for other tasks the predictions fall back to
average task difficulty and lose accuracy, but the estimate stays unbiased.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
from balanced import design_variance, stratified_uniforms
from step_features import RunFeatures, gold_info

from shortstop import rr_estimate

Z = 1.96  # the normal quantile of coverage/experiments.py
LO, HI = 1e-4, 1 - 1e-4  # success predictions are kept inside [LO, HI]


def as_dict(step) -> dict:
    return dataclasses.asdict(step) if dataclasses.is_dataclass(step) else dict(step)


def shift_probabilities(f, shift: float = 0.0):
    """Probabilities moved by `shift` on the logit scale, kept inside [LO, HI]."""
    f = np.clip(f, LO, HI)
    return np.clip(expit(logit(f) + shift), LO, HI) if shift else f


def logit_shift(f0, outcomes) -> float:
    """The logit shift that brings the mean of the step-0 predictions f0 to the mean of the 0/1
    outcomes of the same runs. A record of all 0s or all 1s is pulled half a run inwards, so the
    shift stays finite."""
    f0, y = np.clip(np.asarray(f0, float), LO, HI), np.asarray(outcomes, float)
    if len(y) == 0 or len(y) != len(f0) or np.any((y < 0) | (y > 1)):
        raise ValueError("need at least one outcome, each 0 or 1, one per step-0 prediction")
    rate = float(np.clip(y.mean(), 0.5 / len(y), 1 - 0.5 / len(y)))
    return float(brentq(lambda d: expit(logit(f0) + d).mean() - rate, -30, 30))


def gold_table(paths: list[str]) -> pd.DataFrame | None:
    """The gold patch tables named in meta.json; relative paths are read from the repo root, so
    the runtime loads the same way from any folder. A missing table is an error: without it every
    prediction would silently lose its task features."""
    files = [p if p.is_absolute() else REPO / p for p in map(Path, paths)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"gold patch table missing: {', '.join(missing)}. Make it with "
                                "step 1 of 'Run it' in predictor/README.md.")
    return pd.concat([pd.read_parquet(p) for p in files]).drop_duplicates("instance_id") if files else None


class Runtime:
    def __init__(self, success, cost: lgb.Booster, meta: dict, gold: dict, shift: float = 0.0):
        """success: one LightGBM model, or a list of them whose predictions are averaged.
        shift: added to the logit of every success prediction (see calibrate)."""
        self.success_models = list(success) if isinstance(success, (list, tuple)) else [success]
        self.cost_model, self.meta, self.gold = cost, meta, gold
        self.shift = float(shift)

    @classmethod
    def load(cls, path: str | Path) -> Runtime:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        table = gold_table(meta["gold_meta"])
        gold = gold_info(table) if table is not None else {}
        success = [lgb.Booster(model_file=str(path / f)) for f in meta.get("success_models", ["success.txt"])]
        return cls(success, lgb.Booster(model_file=str(path / "cost.txt")), meta, gold)

    def calibrate(self, outcomes: Mapping[str, float]) -> Runtime:
        """A copy whose success predictions all move by one logit shift, chosen so that the mean
        step-0 prediction on these tasks equals the agent's solve rate on them. outcomes: task id
        -> 0 or 1, from earlier full runs of the same agent. Calibrate before the batch starts and
        keep the shift fixed during it; see the module docstring."""
        plain = type(self)(self.success_models, self.cost_model, self.meta, self.gold)
        f0 = [plain.predict(t, plain.tracker(t))[0] for t in outcomes]
        shift = logit_shift(f0, list(outcomes.values()))
        return type(self)(self.success_models, self.cost_model, self.meta, self.gold, shift)

    def tracker(self, task_id: str) -> RunFeatures:
        return RunFeatures(self.gold.get(task_id.split("@")[0]))

    def _row(self, task_id: str, state: RunFeatures) -> pd.DataFrame:
        task = task_id.split("@")[0]
        row = state.row()
        row["task_rate"] = self.meta["task_rate"].get(task, self.meta["default_task_rate"])
        row["task_cost"] = self.meta["task_cost"].get(task, self.meta["default_task_cost"])
        return pd.DataFrame([row])

    def predict(self, task_id: str, state: RunFeatures) -> tuple[float, float]:
        """(success probability, expected remaining tokens) after the steps the tracker has seen."""
        x = self._row(task_id, state)
        xs = x[self.meta["success_features"]]
        f = float(np.mean([m.predict(xs)[0] for m in self.success_models]))
        r = float(np.expm1(self.cost_model.predict(x[self.meta["cost_features"]])[0]))
        return float(shift_probabilities(f, self.shift)), max(r, 1.0)

    def predictor(self, task_id: str):
        """The shared shortstop.Predictor interface: steps so far -> P(success)."""
        def f(prefix) -> float:
            state = self.tracker(task_id)
            for step in prefix:
                state.update(as_dict(step))
            return self.predict(task_id, state)[0]
        return f

    def target(self, f: float, r_rel: float) -> float:
        k, floor = self.meta["kappa"], self.meta["floor"]
        return float(np.clip(k * np.sqrt(f * (1 - f) / max(r_rel, 1e-6)), floor, 1.0))

    def plan(self, task_ids: list[str], seed: int = 0, design: str = "standard") -> Plan:
        """Plan a batch: design "standard" (independent stopping, proven interval) or "balanced"
        (balanced stopping, design-based interval, not yet proven). See the module docstring."""
        if design not in DESIGNS:
            raise ValueError(f"design must be one of {sorted(DESIGNS)}, not {design!r}")
        return DESIGNS[design](self, task_ids, np.random.default_rng(seed))


class Run:
    def __init__(self, plan: Plan, task_id: str, u: float):
        self.plan, self.task_id, self.u = plan, task_id, u
        self.state = plan.rt.tracker(task_id)
        f0, _ = plan.step0[task_id]
        self.f = [f0]          # f_0 .. f_k
        self.reach: list[float] = []
        self.tokens = 0
        self.done = False
        self.finished = False

    def should_continue(self) -> bool:
        """Ask before every agent step. Asking again before that step is observed (a retry)
        returns the same answer without a new draw."""
        if self.done:
            return False
        if len(self.reach) == len(self.f):  # the step decided last has not been observed yet
            return True
        if not self.reach:
            p = self.plan.p1[self.task_id]  # the step-1 probability the plan was built with
        else:
            f, r = self.plan.rt.predict(self.task_id, self.state)
            p = min(self.reach[-1], self.plan.rt.target(f, r / self.plan.scale))
        if self.u < p:
            self.reach.append(p)
            return True
        self.done = True
        return False

    def observe(self, step) -> None:
        """The step the harness just took, after should_continue() said True for it."""
        if self.done:
            raise RuntimeError(f"{self.task_id}: observe() after the run ended")
        if len(self.reach) != len(self.f):
            raise RuntimeError(f"{self.task_id}: call should_continue() before every step")
        step = as_dict(step)
        self.state.update(step)
        self.tokens += int(step["tokens_in"]) + int(step["tokens_out"])
        self.f.append(self.plan.rt.predict(self.task_id, self.state)[0])

    def finish(self, outcome: float, tokens: int = 0) -> None:
        """The benchmark's outcome for a run that ended by itself.

        After its last observed step (the agent submitted, or a limit was reached) the outcome
        replaces the prediction made after that step: f_T = Y. During a step that never completed
        (a context overflow, a crash) the attempted step reveals the outcome, so it is appended
        as one more step. tokens: any spent after the last observed step, such as a failed call.
        """
        if self.finished:
            raise RuntimeError(f"{self.task_id}: finish() was already called")
        if self.done:
            raise RuntimeError(f"{self.task_id}: the run was stopped early, so its outcome must not be used")
        if not self.reach:
            raise RuntimeError(f"{self.task_id}: call should_continue() before every step, including the first")
        if not 0 <= outcome <= 1:
            raise ValueError(f"{self.task_id}: the outcome must be between 0 and 1, not {outcome}")
        if len(self.reach) == len(self.f):  # the step decided last never completed
            self.f.append(float(outcome))
        else:
            self.f[-1] = float(outcome)
        self.tokens += int(tokens)
        self.done = self.finished = True

    def continue_probabilities(self) -> list[float]:
        """p_k = P_k / P_{k-1}: shortstop.rr_estimate takes the probability of continuing into
        step k given step k-1 was reached, while self.reach holds P_k = P(reach step k)."""
        P = self.reach
        return [P[0], *(P[k] / P[k - 1] for k in range(1, len(P)))] if P else []

    def estimate(self) -> float:
        return rr_estimate(self.f, self.continue_probabilities(), len(self.reach))


class Plan:
    """One evaluation batch. The two designs below differ only in uniforms() and half_width()."""

    design = ""

    def __init__(self, rt: Runtime, task_ids: list[str], rng):
        self.rt, self.task_ids = rt, list(task_ids)
        self.step0 = {t: rt.predict(t, rt.tracker(t)) for t in self.task_ids}
        self.scale = float(np.mean([r for _, r in self.step0.values()]))
        self.p1 = {t: rt.target(f, r / self.scale) for t, (f, r) in self.step0.items()}
        self.u = dict(zip(self.task_ids, self.uniforms(rng)))
        self.runs: dict[str, Run] = {}

    def uniforms(self, rng) -> np.ndarray:
        """Each run's stopping uniform, in task order."""
        raise NotImplementedError

    def half_width(self, ids: list[str], runs: list[Run], est: np.ndarray) -> float:
        """Half-width of the 95% interval around the mean of the per-run estimates est."""
        raise NotImplementedError

    def start(self, task_id: str) -> Run:
        self.runs[task_id] = Run(self, task_id, float(self.u[task_id]))
        return self.runs[task_id]

    def result(self) -> dict:
        ids = [t for t in self.task_ids if t in self.runs]
        runs = [self.runs[t] for t in ids]
        unended = [r.task_id for r in runs if not r.done]
        if unended:
            raise RuntimeError(f"{len(unended)} run(s) neither stopped nor finished, e.g. "
                               f"{', '.join(unended[:3])}: call finish(outcome) when a run ends")
        est = np.array([r.estimate() for r in runs])
        half = self.half_width(ids, runs, est)
        score = float(est.mean())
        return {"design": self.design, "score": score, "interval": (score - half, score + half),
                "half_width": half, "tasks": len(runs), "runs_started": int(sum(len(r.reach) > 0 for r in runs)),
                "runs_finished": int(sum(r.finished for r in runs)), "tokens": int(sum(r.tokens for r in runs)),
                "shift": self.rt.shift}


class StandardPlan(Plan):
    """Independent stopping uniforms and Vedanti's interval, mean +/- 1.96 sd / sqrt(n)
    (coverage/experiments.py, interval_stats), proven in coverage/THEORY.md, Theorem 2a."""

    design = "standard"

    def uniforms(self, rng) -> np.ndarray:
        return rng.random(len(self.task_ids))

    def half_width(self, ids: list[str], runs: list[Run], est: np.ndarray) -> float:
        return float(Z * est.std(ddof=1) / np.sqrt(len(est)))


class BalancedPlan(Plan):
    """Stopping uniforms spread evenly within strata of meta["block"] runs sorted by (P_1, f0),
    and the design-based interval of predictor/balanced.py (validated by simulation on two
    benchmarks; not yet proven)."""

    design = "balanced"

    def uniforms(self, rng) -> np.ndarray:
        p1 = np.array([self.p1[t] for t in self.task_ids])[None, :]
        f0 = np.array([self.step0[t][0] for t in self.task_ids])[None, :]
        u, strata = stratified_uniforms(p1, f0, self.rt.meta["block"], rng)
        self.stratum = dict(zip(self.task_ids, strata[0]))
        return u[0]

    def half_width(self, ids: list[str], runs: list[Run], est: np.ndarray) -> float:
        def row(values) -> np.ndarray:
            return np.asarray(values)[None, :]
        _, var, df = design_variance(row(est), row([r.f[0] for r in runs]), row([self.p1[t] for t in ids]),
                                     row([len(r.reach) > 0 for r in runs]), row([self.stratum[t] for t in ids]),
                                     len(runs))
        return float(student_t.ppf(0.975, df[0]) * np.sqrt(var[0]))


DESIGNS = {"standard": StandardPlan, "balanced": BalancedPlan}
