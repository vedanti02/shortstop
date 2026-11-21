"""Run-time interface for a live evaluation harness (e.g. Rishi's mini-swe-agent plug-in).

    rt = Runtime.load("data/predictor/runtime/all")       # made by predictor/export.py
    plan = rt.plan(task_ids, seed=0)                       # before any run starts
    for task_id in task_ids:
        run = plan.start(task_id)
        while run.should_continue():                       # decided before every agent step
            step = ...                                     # the harness takes one agent step
            run.observe(step)                              # a shortstop.Step, or a dict of its fields
            if the agent submitted:
                run.finish(outcome)                        # 0 or 1, from the benchmark's tests
                break
    result = plan.result()     # unbiased score, 95% interval, tokens spent, runs started

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

What it does, per predictor/README.md:
  - f_k: the success prediction after k steps (gold LightGBM).
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
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from balanced import design_variance, stratified_uniforms
from step_features import RunFeatures, gold_info

from shortstop import rr_estimate

Z = 1.96  # the normal quantile of coverage/experiments.py


def as_dict(step) -> dict:
    return dataclasses.asdict(step) if dataclasses.is_dataclass(step) else dict(step)


class Runtime:
    def __init__(self, success: lgb.Booster, cost: lgb.Booster, meta: dict, gold: dict):
        self.success_model, self.cost_model, self.meta, self.gold = success, cost, meta, gold

    @classmethod
    def load(cls, path: str | Path) -> Runtime:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        frames = [pd.read_parquet(p) for p in meta["gold_meta"] if Path(p).exists()]
        gold = gold_info(pd.concat(frames).drop_duplicates("instance_id")) if frames else {}
        return cls(lgb.Booster(model_file=str(path / "success.txt")),
                   lgb.Booster(model_file=str(path / "cost.txt")), meta, gold)

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
        f = float(self.success_model.predict(x[self.meta["success_features"]])[0])
        r = float(np.expm1(self.cost_model.predict(x[self.meta["cost_features"]])[0]))
        return min(max(f, 1e-4), 1 - 1e-4), max(r, 1.0)

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
        if self.done:
            return False
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
        step = as_dict(step)
        self.state.update(step)
        self.tokens += int(step["tokens_in"]) + int(step["tokens_out"])
        self.f.append(self.plan.rt.predict(self.task_id, self.state)[0])

    def finish(self, outcome: int) -> None:
        self.f[-1] = float(outcome)  # f_T = Y
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
        est = np.array([r.estimate() for r in runs])
        half = self.half_width(ids, runs, est)
        score = float(est.mean())
        return {"design": self.design, "score": score, "interval": (score - half, score + half),
                "half_width": half, "tasks": len(runs), "runs_started": int(sum(len(r.reach) > 0 for r in runs)),
                "runs_finished": int(sum(r.finished for r in runs)), "tokens": int(sum(r.tokens for r in runs))}


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
