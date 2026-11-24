"""Connects Rishi's mini-swe-agent plug-in (tool_ab/shortstop_plugin, PR #3) to runtime.py.

The plug-in names a predictor and a stop policy in its agent config by dotted path plus keyword
arguments, builds each once (policies.load_shared), and before every step k calls

    f   = predictor(prefix, task_id=...)                       the prediction f_{k-1}
    p_k = stop_policy(prefix, (f_0, ..., f_{k-1}), task_id=...)  P_k / P_{k-1}

It then runs step k while one uniform per task stays below p_1 * ... * p_k, and saves the
predictions and continue probabilities in each run's record. This module provides both
callables. They share one LiveSession, which loads the runtime models once, calibrates them once,
plans the batch, and answers exactly as runtime.py's standard design would: P_1 is the plan's
step-1 reach probability, and later P_k = min(P_{k-1}, kappa sqrt(f (1 - f) / r)) with the floor,
r being the expected remaining tokens relative to the batch's step-0 mean.

In the plug-in's agent config (YAML), for each arm:

    predictor: plugin_adapter.predictor
    predictor_kwargs:
      runtime: data/predictor/runtime/all
      tasks_file: tool_ab/configs/tasks.txt
      slice: "0:100"                             # the same tasks, in the same order, as the run
      calibrate_from: data/tool_ab/qwen-baseline.jsonl   # optional: earlier full runs
      arm: baseline
      log: data/predictor/live/baseline.decisions.jsonl
    stop_policy: plugin_adapter.policy
    stop_policy_kwargs: {}                       # uses the predictor's session
    seed: 0                                      # the same seed in every arm

and run the plug-in with this folder on the import path, e.g.
PYTHONPATH=predictor python tool_ab/run_arm.py --arm baseline --slice 0:100.
Relative paths are read from the repo root. The settings go to the predictor only, so they cannot
drift apart between the two; a policy given settings must be given exactly the predictor's.

Only the standard design is offered: the plug-in draws each task's uniform itself, from the seed
and the task, which is right for independent stopping. The balanced designs deal the uniforms
across a group of tasks, so they need the plug-in to take each task's uniform from the plan.

Calibration (optional) uses earlier full runs of the same agent, as Runtime.calibrate: a JSON
file {task_id: 0 or 1}, or a trajectory file in the shared format (one outcome per task; -1 rows
are skipped). For a tool A/B test give every arm the baseline's file and the same seed.

Logging. Every event is one JSON line in `log`, flushed as it is written: a "session" line at
start (models, shift, tasks, kappa, cost scale), "run_start" when a task's run begins (also for a
rerun), one "decision" line per step (f, expected remaining tokens, P_k, p_k, tokens so far, and
the time each call took), and "error" when a call fails. The same events go to the Python logger
"shortstop.adapter" (session at INFO, reruns and slow calls at WARNING, failures at ERROR).
predictor/live_monitor.py reads the log, the plug-in's run folder and the grading report.

Any problem raises: a missing task ID, a task outside the plan, a policy call without the
predictor call for the same step, or predictions that differ from this session's. The plug-in
turns that into a PredictorError and stops starting new tasks.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import REPO, Runtime, as_dict
from step_features import RunFeatures

logger = logging.getLogger("shortstop.adapter")

KEYS = {"runtime", "tasks", "tasks_file", "slice", "calibrate_from", "arm", "log", "design", "latency_warn_ms"}


def resolve(path: str | Path) -> Path:
    """A path from the config: relative paths are read from the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def slice_tasks(tasks: list[str], spec: str | None) -> list[str]:
    """tasks[a:b] for spec "a:b", as tool_ab/run_arm.py --slice does."""
    if not spec:
        return tasks
    bounds = [int(x) if x else None for x in str(spec).split(":")]
    if len(bounds) > 3:
        raise ValueError(f"slice must look like 'a:b', not {spec!r}")
    return tasks[slice(*bounds)]


def read_tasks(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def read_outcomes(path: Path) -> dict[str, int]:
    """Earlier full-run outcomes: a JSON object {task_id: 0 or 1}, or a shared-format trajectory
    file (one JSON object per step), keeping each task's outcome and skipping -1."""
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        out = {str(t): int(y) for t, y in data.items()}
    else:
        out = {}
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("outcome") in (0, 1):
                    out.setdefault(row["task_id"], int(row["outcome"]))
    if not out or any(y not in (0, 1) for y in out.values()):
        raise ValueError(f"{path}: need at least one outcome, each 0 or 1")
    return out


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TaskRun:
    """What the session knows about the run in progress for one task."""

    tracker: RunFeatures
    attempt: int
    seen: int = 0                        # steps given to the tracker
    f: float | None = None               # prediction after `seen` steps
    r: float | None = None               # expected remaining tokens after `seen` steps
    predict_ms: float = 0.0
    reach: list[float] = field(default_factory=list)   # P_1 .. P_k decided so far
    tokens: int = 0


class LiveSession:
    """One live batch: the runtime models, the plan, and every task's run in progress."""

    def __init__(self, rt: Runtime, task_ids: Sequence[str], *, log: str | Path | None = None,
                 arm: str = "live", latency_warn_ms: float = 1000.0, settings: dict | None = None,
                 calibrated_on: int = 0):
        if len(set(task_ids)) != len(task_ids) or not task_ids:
            raise ValueError("the batch needs at least one task, each listed once")
        self.rt, self.arm, self.latency_warn_ms = rt, arm, float(latency_warn_ms)
        self.plan = rt.plan(list(task_ids), design="standard")   # step-0 predictions, cost scale, P_1
        self.id = uuid.uuid4().hex[:8]
        self._runs: dict[str, TaskRun] = {}
        self._attempts: dict[str, int] = {}
        self._lock = threading.RLock()
        self._log_path = resolve(log) if log else None
        self._log = None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self._log_path.open("a", buffering=1)  # line-buffered: each event is flushed
        f0 = [f for f, _ in self.plan.step0.values()]
        self.summary = {
            "arm": arm, "tasks": len(task_ids), "design": "standard", "shift": rt.shift,
            "calibrated_on": calibrated_on, "success_models": len(rt.success_models),
            "kappa": rt.meta["kappa"], "floor": rt.meta["floor"], "cost_scale": self.plan.scale,
            "mean_f0": sum(f0) / len(f0), "predicted_full_tokens": sum(r for _, r in self.plan.step0.values()),
            "task_ids": list(task_ids), "settings": settings or {},
        }
        self._write({"event": "session", **self.summary})
        logger.info("session %s (%s): %d tasks, shift %+.3f, mean step-0 prediction %.3f, log %s",
                    self.id, arm, len(task_ids), rt.shift, self.summary["mean_f0"], self._log_path)

    @classmethod
    def from_config(cls, settings: dict[str, Any]) -> LiveSession:
        unknown = set(settings) - KEYS
        if unknown:
            raise TypeError(f"unknown setting(s) {sorted(unknown)}; expected some of {sorted(KEYS)}")
        if settings.get("design", "standard") != "standard":
            raise ValueError("only design 'standard' works with the plug-in, which draws each task's "
                             "uniform itself; the balanced designs need the uniform from the plan")
        if "runtime" not in settings:
            raise TypeError("missing setting 'runtime' (a folder made by predictor/export.py runtime)")
        if ("tasks" in settings) == ("tasks_file" in settings):
            raise TypeError("give exactly one of 'tasks' (a list) and 'tasks_file'")
        tasks = list(settings["tasks"]) if "tasks" in settings else read_tasks(resolve(settings["tasks_file"]))
        tasks = slice_tasks(tasks, settings.get("slice"))
        rt = Runtime.load(resolve(settings["runtime"]))
        outcomes = {}
        if settings.get("calibrate_from"):
            outcomes = read_outcomes(resolve(settings["calibrate_from"]))
            rt = rt.calibrate(outcomes)
        arm = str(settings.get("arm", "live"))
        log = settings.get("log", f"data/predictor/live/{arm}.decisions.jsonl")
        return cls(rt, tasks, log=log, arm=arm, latency_warn_ms=settings.get("latency_warn_ms", 1000.0),
                   settings=settings, calibrated_on=len(outcomes))

    # -- the two callables the plug-in uses ------------------------------------------------------

    def predict(self, prefix: Sequence, task_id: str | None = None) -> float:
        """f_{k-1}: the success prediction after the steps so far (empty before step 1)."""
        start = time.perf_counter()
        n = len(prefix)
        with self._lock:
            try:
                task = self._task(task_id)
                run = self._runs.get(task)
                if run is None or n == 0 or n < run.seen:
                    run = self._start(task, run, n)
                for step in prefix[run.seen:]:
                    d = as_dict(step)
                    run.tracker.update(d)
                    run.tokens += int(d["tokens_in"]) + int(d["tokens_out"])
                run.seen = n
                run.f, run.r = self.plan.step0[task] if n == 0 else self.rt.predict(task, run.tracker)
                run.predict_ms = 1000 * (time.perf_counter() - start)
                return run.f
            except Exception as e:
                self._error("predictor", task_id, n + 1, e)
                raise

    def continue_probability(self, prefix: Sequence, predictions: Sequence[float],
                             task_id: str | None = None) -> float:
        """p_k = P_k / P_{k-1} for step k = len(prefix) + 1, after predict() for the same prefix."""
        start = time.perf_counter()
        n = len(prefix)
        k = n + 1
        with self._lock:
            try:
                task = self._task(task_id)
                run = self._runs.get(task)
                if run is None or run.seen != n or run.f is None:
                    raise RuntimeError(f"{task}: the stop policy was called for step {k} before the predictor, "
                                       "or with other settings: give the policy no settings (it then uses the "
                                       "predictor's session) or exactly the predictor's")
                if len(predictions) != k or abs(float(predictions[-1]) - run.f) > 1e-9:
                    raise RuntimeError(f"{task}: predictions {list(predictions)[-3:]} (of {len(predictions)}) do "
                                       f"not end with this session's f = {run.f:.6f} for step {k}; are the "
                                       "predictor and the stop policy from the same settings?")
                if len(run.reach) < k:
                    if len(run.reach) != n:
                        raise RuntimeError(f"{task}: step {k} asked before step {len(run.reach) + 1}")
                    target = self.plan.p1[task] if n == 0 else self.rt.target(run.f, run.r / self.plan.scale)
                    run.reach.append(min(run.reach[-1], target) if run.reach else target)
                reach, before = run.reach[k - 1], (run.reach[k - 2] if k > 1 else 1.0)
                p = reach / before
                total_ms = run.predict_ms + 1000 * (time.perf_counter() - start)
                self._write({"event": "decision", "task_id": task, "attempt": run.attempt, "k": k, "f": run.f,
                             "r_hat": run.r, "r_rel": run.r / self.plan.scale, "reach": reach, "p": p,
                             "at_floor": reach <= self.rt.meta["floor"] + 1e-12, "tokens_so_far": run.tokens,
                             "predict_ms": round(run.predict_ms, 3), "total_ms": round(total_ms, 3)})
                if total_ms > self.latency_warn_ms:
                    logger.warning("%s step %d: predictor and policy took %.0f ms", task, k, total_ms)
                return p
            except Exception as e:
                self._error("stop_policy", task_id, k, e)
                raise

    # -- helpers -----------------------------------------------------------------------------------

    def _task(self, task_id: str | None) -> str:
        if not task_id:
            raise ValueError("the adapter needs the task ID: use the plug-in from PR #3 (9abe23c or "
                             "later), which passes task_id to predictors and policies that accept it")
        if task_id not in self.plan.step0:
            raise KeyError(f"{task_id} is not in this batch's {len(self.plan.step0)} planned tasks; "
                           "check tasks_file and slice against the run's")
        return task_id

    def _start(self, task: str, previous: TaskRun | None, n: int) -> TaskRun:
        attempt = self._attempts.get(task, 0) + 1
        self._attempts[task] = attempt
        run = TaskRun(self.rt.tracker(task), attempt)
        self._runs[task] = run
        f0, r0 = self.plan.step0[task]
        self._write({"event": "run_start", "task_id": task, "attempt": attempt, "f0": f0, "r0": r0,
                     "p1": self.plan.p1[task], "prefix_steps": n})
        if previous is not None:
            logger.warning("%s: a new run (attempt %d) replaces the one at step %d", task, attempt, previous.seen + 1)
        return run

    def _error(self, where: str, task: str | None, k: int, e: Exception) -> None:
        self._write({"event": "error", "where": where, "task_id": task, "k": k, "error": repr(e)})
        logger.error("%s failed for %s at step %d: %r", where, task, k, e)

    def _write(self, event: dict) -> None:
        if self._log is not None:
            with self._lock:
                self._log.write(json.dumps({"time": now(), "session": self.id, **event}) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._log is not None:
                self._log.close()
                self._log = None


_SESSIONS: dict[str, LiveSession] = {}
_SESSIONS_LOCK = threading.Lock()


def session_for(settings: dict[str, Any]) -> LiveSession:
    """The session for these settings, built once per process and shared across threads."""
    key = json.dumps(settings, sort_keys=True, default=str)
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            _SESSIONS[key] = LiveSession.from_config(settings)
        return _SESSIONS[key]


def predictor(**settings) -> Callable[..., float]:
    """Factory for the plug-in's `predictor` setting: (prefix, task_id) -> f."""
    return session_for(settings).predict


def policy(**settings) -> Callable[..., float]:
    """Factory for the plug-in's `stop_policy` setting: (prefix, predictions, task_id) -> p_k.

    Without settings it uses the predictor's session, found at the first call, since the plug-in
    may build the policy before the predictor; a process with several sessions must say which."""
    if settings:
        return session_for(settings).continue_probability

    def continue_probability(prefix: Sequence, predictions: Sequence[float], task_id: str | None = None) -> float:
        with _SESSIONS_LOCK:
            sessions = list(_SESSIONS.values())
        if len(sessions) != 1:
            raise RuntimeError(f"the stop policy without settings uses the predictor's session, but this process "
                               f"has {len(sessions)}: give the policy exactly the predictor's settings")
        return sessions[0].continue_probability(prefix, predictions, task_id=task_id)

    return continue_probability
