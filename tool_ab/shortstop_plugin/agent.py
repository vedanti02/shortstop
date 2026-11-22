"""mini-swe-agent agents for the tool A/B experiments.

ToolAgent      installs helper scripts in the sandbox and optionally compresses outputs.
StoppingAgent  adds the continue-or-stop hook: before each step it asks the predictor for
               f_{k-1}, asks the policy for p_k, and runs the step with probability p_k.
               Everything the estimator needs is saved under info.shortstop.
"""

from __future__ import annotations

import random
import time
from typing import Any

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow
from pydantic import Field

from .compress import CompressionConfig, compress_result
from .policies import load_shared
from .scripts import install_command
from .steps import messages_to_steps, to_step_objects


class EarlyStopped(InterruptAgentFlow):
    """Raised when the stop policy decides not to run the next step."""


class ToolAgentConfig(AgentConfig):
    setup_scripts: list[str] = Field(default_factory=list)
    """Bundled helper scripts to install in the sandbox before the run (e.g. ["search"])."""
    script_install_dir: str = "/usr/local/bin"
    compression: CompressionConfig | None = None
    """If set, command outputs are compressed before the model sees them."""
    extra_instructions: str = ""
    """Extra text the instance template can include, e.g. documentation for a new tool."""


class StoppingAgentConfig(ToolAgentConfig):
    stop_policy: str = "shortstop_plugin.policies.never_stop"
    stop_policy_kwargs: dict[str, Any] = Field(default_factory=dict)
    predictor: str = "shortstop.predictor.constant_predictor"
    predictor_kwargs: dict[str, Any] = Field(default_factory=lambda: {"p": 0.5})
    seed: int | None = None
    """With a seed, the stop draws depend only on (seed, instance_id)."""
    agent_name: str = "mini-swe-agent"
    """Value of the `agent` field in the steps handed to the predictor."""


class ToolAgent(DefaultAgent):
    def __init__(self, model, env, *, config_class: type = ToolAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

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
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )


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
        self._predictions: list[float] = []
        self._continue_probs: list[float] = []
        self._stopped_early = False

    def _steps_so_far(self) -> list[dict]:
        return messages_to_steps(self.messages)

    def _decide(self, k: int, steps: list[dict]) -> bool:
        """Draw whether to run step k. Called once per step."""
        prefix = to_step_objects(steps, task_id=self.instance_id, agent=self.config.agent_name)
        f = float(self._predictor(prefix))
        if not 0 <= f <= 1:
            raise ValueError(f"Predictor returned {f} before step {k}; expected [0, 1]")
        p = float(self._policy(prefix, (*self._predictions, f)))
        if not 0 < p <= 1:
            raise ValueError(f"Stop policy returned {p} before step {k}; expected (0, 1]")
        self._predictions.append(f)
        self._continue_probs.append(p)
        return self._rng.random() < p

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
        # A format error adds no model call, so the same step can be queried twice: draw once.
        needs_draw = len(self._continue_probs) < k and not self._limit_reached()
        if needs_draw and not self._decide(k, steps):
            self._stopped_early = True
            raise EarlyStopped({
                "role": "exit",
                "content": "EarlyStopped",
                "extra": {"exit_status": "EarlyStopped", "submission": ""},
            })
        return super().query()

    def shortstop_record(self) -> dict:
        return {
            "steps_run": len(self._steps_so_far()),
            "stopped_early": self._stopped_early,
            "predictions": list(self._predictions),
            "continue_probs": list(self._continue_probs),
            "seed": self.config.seed,
            "stop_policy": self.config.stop_policy,
            "stop_policy_kwargs": dict(self.config.stop_policy_kwargs),
            "predictor": self.config.predictor,
        }

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"shortstop": self.shortstop_record()}}, *extra_dicts)
