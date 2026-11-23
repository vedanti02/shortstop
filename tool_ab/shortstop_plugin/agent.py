"""mini-swe-agent agents for the tool A/B experiments.

ToolAgent      installs helper scripts in the sandbox, optionally compresses outputs, and can
               save the repository diff after each step.
StoppingAgent  adds the continue-or-stop hook: before each step it asks the predictor for
               f_{k-1} and the policy for p_k, and runs the step while one uniform draw per
               task stays below p_1 * ... * p_k. Everything the estimator needs is saved under
               info.shortstop.
"""

from __future__ import annotations

import inspect
import random
import time
from collections.abc import Callable
from typing import Any

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow
from pydantic import Field

from .compress import CompressionConfig, compress_result
from .policies import load_shared
from .scripts import install_command
from .steps import messages_to_steps, to_step_objects

# Diff of the working tree (untracked files included) against HEAD, built in a throwaway index
# so the repository's own index, and so what the agent sees from git, never changes.
DIFF_COMMAND = (
    'tmp=$(mktemp -u) && trap \'rm -f "$tmp"\' EXIT && git_dir=$(git rev-parse --git-dir) && '
    '{ cp "$git_dir/index" "$tmp" 2>/dev/null || true; } && '
    'GIT_INDEX_FILE="$tmp" git add -A . >/dev/null 2>&1 && '
    'GIT_INDEX_FILE="$tmp" git diff --cached --no-color HEAD 2>/dev/null'
)


class EarlyStopped(InterruptAgentFlow):
    """Raised when the stop policy decides not to run the next step."""


class PredictorError(RuntimeError):
    """The predictor or stop policy failed or returned an invalid value. Not retried: fix it first."""


class ToolAgentConfig(AgentConfig):
    setup_scripts: list[str] = Field(default_factory=list)
    """Bundled helper scripts to install in the sandbox before the run (e.g. ["search"])."""
    script_install_dir: str = "/usr/local/bin"
    compression: CompressionConfig | None = None
    """If set, command outputs are compressed before the model sees them."""
    extra_instructions: str = ""
    """Extra text the instance template can include, e.g. documentation for a new tool."""
    save_diffs: bool = False
    """Save `git diff` of the repository after each step under info.step_diffs (not shown to the model).

    A diff equal to the previous step's is saved as {"unchanged": true}. The submission step ends the
    run before its diff is taken; its diff is the submitted patch in preds.json."""
    diff_command: str = DIFF_COMMAND
    max_diff_chars: int = 50_000


class StoppingAgentConfig(ToolAgentConfig):
    stop_policy: str = "shortstop_plugin.policies.never_stop"
    stop_policy_kwargs: dict[str, Any] = Field(default_factory=dict)
    predictor: str = "shortstop.predictor.constant_predictor"
    predictor_kwargs: dict[str, Any] = Field(default_factory=lambda: {"p": 0.5})
    seed: int | None = None
    """With a seed, the stop draw depends only on (seed, instance_id), so every arm draws the same."""
    agent_name: str = "mini-swe-agent"
    """Value of the `agent` field in the steps handed to the predictor (run_arm.py sets qwen-<arm>)."""


class ToolAgent(DefaultAgent):
    def __init__(self, model, env, *, config_class: type = ToolAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._step_diffs: list[dict] = []
        self._last_diff = ""

    def run(self, task: str = "", **kwargs) -> dict:
        for name in self.config.setup_scripts:
            result = self.env.execute({"command": install_command(name, self.config.script_install_dir)})
            if result.get("returncode") != 0:
                raise RuntimeError(f"Installing {name!r} failed: {result.get('output', '')[:500]}")
        return super().run(task, **kwargs)

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        if self.config.compression is not None:
            outputs = [compress_result(o, self.config.compression) for o in outputs]
        added = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )
        if self.config.save_diffs:
            self._save_diff()
        return added

    def _save_diff(self) -> None:
        result = self.env.execute({"command": self.config.diff_command})
        diff = result.get("output", "") if result.get("returncode") == 0 else ""
        limit = self.config.max_diff_chars
        if len(diff) > limit:
            diff = f"{diff[:limit]}\n[... diff cut at {limit} of {len(diff)} characters ...]\n"
        step = len(messages_to_steps(self.messages))
        row = {"step": step, "returncode": result.get("returncode")}
        if diff and diff == self._last_diff:
            row["unchanged"] = True
        else:
            row["diff"] = diff
            self._last_diff = diff
        self._step_diffs.append(row)

    def serialize(self, *extra_dicts) -> dict:
        diffs = {"info": {"step_diffs": list(self._step_diffs)}} if self.config.save_diffs else {}
        return super().serialize(diffs, *extra_dicts)


def _accepts_task_id(fn: Callable) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "task_id" or p.kind is p.VAR_KEYWORD for p in params)


class StoppingAgent(ToolAgent):
    def __init__(self, model, env, *, config_class: type = StoppingAgentConfig,
                 instance_id: str = "", **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.instance_id = instance_id
        # Shared across agents (and worker threads) with the same config, so a learned
        # predictor is loaded once. Shared predictors must be safe to call from several threads.
        self._policy = load_shared(self.config.stop_policy, self.config.stop_policy_kwargs)
        self._predictor = load_shared(self.config.predictor, self.config.predictor_kwargs)
        seed = self.config.seed
        self._rng = random.Random(f"{seed}:{instance_id}") if seed is not None else random.Random()
        # One uniform draw per task: step k runs while u < p_1 * ... * p_k. Each step still runs
        # with probability p_k given the earlier ones, but arms with the same seed stop at
        # matching points as often as possible, which makes paired comparisons cheaper.
        self._u = self._rng.random()
        self._reach = 1.0
        self._predictions: list[float] = []
        self._continue_probs: list[float] = []
        self._stopped_early = False

    def _call(self, fn: Callable, *args) -> float:
        """Call the predictor or policy, passing the task ID to those that take `task_id`.

        The prefix is empty before step 1, so the task ID is the only thing the first
        prediction can go on. The keyword is provisional until the shared Predictor type
        in src/shortstop takes it.
        """
        kwargs = {"task_id": self.instance_id} if _accepts_task_id(fn) else {}
        return float(fn(*args, **kwargs))

    def _steps_so_far(self) -> list[dict]:
        return messages_to_steps(self.messages)

    def _decide(self, k: int, steps: list[dict]) -> bool:
        """Draw whether to run step k. Called once per step."""
        prefix = to_step_objects(steps, task_id=self.instance_id, agent=self.config.agent_name)
        try:
            f = self._call(self._predictor, prefix)
        except Exception as e:
            raise PredictorError(f"Predictor failed before step {k}: {e!r}") from e
        if not 0 <= f <= 1:
            raise PredictorError(f"Predictor returned {f} before step {k}; expected [0, 1]")
        try:
            p = self._call(self._policy, prefix, (*self._predictions, f))
        except Exception as e:
            raise PredictorError(f"Stop policy failed before step {k}: {e!r}") from e
        if not 0 < p <= 1:
            raise PredictorError(f"Stop policy returned {p} before step {k}; expected (0, 1]")
        self._predictions.append(f)
        self._continue_probs.append(p)
        self._reach *= p
        return self._u < self._reach

    def _limit_reached(self) -> bool:
        """Mirrors the checks in DefaultAgent.query: the next call would end the run anyway.

        Such a run has T = steps so far and needs no draw for a step that cannot happen.
        """
        cfg = self.config
        elapsed = int(time.time() - self._start_time)
        return (
            0 < cfg.step_limit <= self.n_calls
            or 0 < cfg.cost_limit <= self.cost
            or 0 < cfg.wall_time_limit_seconds <= elapsed
        )

    def query(self) -> dict:
        steps = self._steps_so_far()
        k = len(steps) + 1
        # A reply that fails to parse uses a model call but adds no step, so step k is queried
        # again after a format error: draw only once for it.
        needs_draw = len(self._continue_probs) < k and not self._limit_reached()
        if needs_draw and not self._decide(k, steps):
            self._stopped_early = True
            raise EarlyStopped({
                "role": "exit",
                "content": "EarlyStopped",
                "extra": {"exit_status": "EarlyStopped", "submission": ""},
            })
        return super().query()

    def estimator_steps(self) -> int:
        """s for rr_estimate: steps drawn and run, counting a step the run ended during.

        A run can end inside step k after its draw (context overflow, repeated format errors,
        a format error on the last allowed call), so it has one more draw than steps. That
        attempted step is the one that reveals the outcome Y.
        """
        return len(self._continue_probs) - int(self._stopped_early)

    def shortstop_record(self) -> dict:
        return {
            "s": self.estimator_steps(),
            "steps_run": len(self._steps_so_far()),
            "stopped_early": self._stopped_early,
            "predictions": list(self._predictions),
            "continue_probs": list(self._continue_probs),
            "seed": self.config.seed,
            "agent_name": self.config.agent_name,
            "stop_policy": self.config.stop_policy,
            "stop_policy_kwargs": dict(self.config.stop_policy_kwargs),
            "predictor": self.config.predictor,
            "predictor_kwargs": dict(self.config.predictor_kwargs),
        }

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"shortstop": self.shortstop_record()}}, *extra_dicts)
