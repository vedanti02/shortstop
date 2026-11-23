"""Run one arm of the tool A/B experiment on SWE-bench Verified with the shortstop plug-in.

    python tool_ab/run_arm.py --arm baseline --slice 0:20 -w 8
    python tool_ab/run_arm.py --arm search --slice 0:100 -w 16

This is `mini-extra swebench` with three differences. The agent is the plug-in
(StoppingAgent; mini-swe-agent's batch runner ignores `agent_class`, so its agent is
swapped here). `swerex_modal` runs commands in bash with the image's conda environment
(see shortstop_plugin/environment.py). The config is configs/base.yaml plus the arm's
overlay, with `agent.agent_name` set to qwen-<arm> as in the converted data. Tasks come from
configs/tasks.txt in a fixed order, so `--slice 0:20` is the same 20 tasks for every arm.

The predictor and stop policy are loaded once before any sandbox starts, so a predictor that
fails to load stops the run at once. If the predictor or policy fails during a run, no new
tasks start and the script exits with an error; fix it, then rerun with --retry-errors.

Outputs go to data/tool_ab/runs/<arm>/: one <task>/<task>.traj.json per run and preds.json
for evaluation. Finished tasks are skipped on a rerun unless --redo-existing is given;
--retry-errors reruns only the tasks that ended in an infrastructure error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path

TOOL_AB = Path(__file__).resolve().parent
REPO = TOOL_AB.parent
CONFIGS = TOOL_AB / "configs"
ARMS: dict[str, list[str]] = {
    "baseline": [],
    "descriptions": ["descriptions.yaml"],
    "search": ["search.yaml"],
    "compressed": ["compressed.yaml"],
}


def config_specs(arm: str, overrides: tuple[str, ...] = ()) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return [str(CONFIGS / "base.yaml"), *(str(CONFIGS / f) for f in ARMS[arm]),
            f"agent.agent_name=qwen-{arm}", *overrides]


def arm_config(arm: str, overrides: tuple[str, ...] = ()) -> dict:
    """The merged config the run will use."""
    from minisweagent.config import get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    return recursive_merge(*(get_config_from_spec(s) for s in config_specs(arm, overrides)))


def read_tasks(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def slice_tasks(tasks: list[str], spec: str) -> list[str]:
    if not spec:
        return tasks
    bounds = [int(x) if x else None for x in spec.split(":")]
    return tasks[slice(*bounds)]


def filter_regex(task_ids: list[str]) -> str:
    return "^(" + "|".join(re.escape(t) for t in task_ids) + ")$"


def forget_errored_runs(output_dir: Path) -> list[str]:
    """Drop tasks that ended in an infrastructure error from preds.json so they run again."""
    from shortstop_plugin.status import infra_error_of, tasks_without_trajectory

    preds_path = Path(output_dir) / "preds.json"
    if not preds_path.exists():
        return []
    errored = tasks_without_trajectory(output_dir)
    for traj_path in sorted(Path(output_dir).glob("*/*.traj.json")):
        if infra_error_of(json.loads(traj_path.read_text())) is not None:
            errored.append(traj_path.parent.name)
    preds = json.loads(preds_path.read_text())
    kept = {task: pred for task, pred in preds.items() if task not in errored}
    preds_path.write_text(json.dumps(kept, indent=2))
    return errored


class RunAborted(RuntimeError):
    """A task was skipped because an earlier one hit a predictor error."""


def preload_stop_components(config: dict) -> None:
    """Build the predictor and stop policy before any sandbox starts.

    Agents get the same shared objects afterwards (policies.load_shared), so a predictor that
    fails to load stops the run here instead of failing once per task with a live sandbox.
    """
    from shortstop_plugin.agent import StoppingAgentConfig
    from shortstop_plugin.policies import load_shared

    agent = StoppingAgentConfig(**{k: v for k, v in config.get("agent", {}).items()
                                   if k in StoppingAgentConfig.model_fields})
    load_shared(agent.stop_policy, agent.stop_policy_kwargs)
    load_shared(agent.predictor, agent.predictor_kwargs)


def make_progress_agent(abort: threading.Event | None = None) -> type:
    """StoppingAgent with the progress reporting the batch runner expects.

    It stops its environment, also when the agent cannot be built, and sets `abort` on a
    predictor error so the remaining tasks are skipped.
    """
    from shortstop_plugin.agent import PredictorError, StoppingAgent

    abort = abort or threading.Event()

    class ProgressStoppingAgent(StoppingAgent):
        def __init__(self, model, env, *, progress_manager, instance_id: str = "", **kwargs):
            try:
                super().__init__(model, env, instance_id=instance_id, **kwargs)
            except Exception:
                _stop_env(env)  # the sandbox already exists and nothing else would stop it
                raise
            self.progress_manager = progress_manager

        def step(self) -> list[dict]:
            self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d}")
            return super().step()

        def run(self, *args, **kwargs) -> dict:
            # The batch runner never stops environments, so each Modal sandbox would outlive its run.
            try:
                if abort.is_set():
                    raise RunAborted("skipped: an earlier task hit a predictor error")
                return super().run(*args, **kwargs)
            except PredictorError:
                abort.set()
                raise
            finally:
                _stop_env(self.env)

    return ProgressStoppingAgent


def _stop_env(env) -> None:
    """Stop the environment without masking the exception that ended the run."""
    stop = getattr(env, "stop", None)
    if stop is None:
        return
    try:
        stop()
    except Exception as e:  # noqa: BLE001 -- the sandbox's own timeout still ends it
        print(f"warning: stopping the environment failed: {e!r}", file=sys.stderr)


def guard_environment_start(start, abort: threading.Event):
    """Wrap the runner's sandbox factory so no new sandbox starts once the run is aborted."""
    def guarded(*args, **kwargs):
        if abort.is_set():
            raise RunAborted("skipped: an earlier task hit a predictor error")
        return start(*args, **kwargs)

    return guarded


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--tasks", type=Path, default=CONFIGS / "tasks.txt")
    parser.add_argument("--slice", default="", help="Python-style slice of the task list, e.g. 0:20")
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-c", "--config", action="append", default=[],
                        help="Extra config file or key=value override, applied last")
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Rerun tasks whose last run ended in an infrastructure error")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(TOOL_AB / ".env")
    if str(TOOL_AB) not in sys.path:
        sys.path.insert(0, str(TOOL_AB))  # so configs can name shortstop_plugin classes

    from minisweagent.run.benchmarks import swebench
    from shortstop_plugin import environment

    tasks = slice_tasks(read_tasks(args.tasks), args.slice)
    if not tasks:
        raise SystemExit("No tasks selected")
    output = args.output or REPO / "data" / "tool_ab" / "runs" / args.arm
    specs = config_specs(args.arm, tuple(args.config))
    preload_stop_components(arm_config(args.arm, tuple(args.config)))
    if args.retry_errors:
        print(f"retrying {len(forget_errored_runs(output))} errored tasks")
    abort = threading.Event()
    swebench.ProgressTrackingAgent = make_progress_agent(abort)
    swebench.get_sb_environment = guard_environment_start(swebench.get_sb_environment, abort)
    environment.register()
    swebench.main(
        subset="verified",
        split="test",
        slice_spec="",
        filter_spec=filter_regex(tasks),
        shuffle=False,
        output=str(output),
        workers=args.workers,
        model=None,
        model_class=None,
        redo_existing=args.redo_existing,
        config_spec=specs,
        environment_class=None,
    )
    if abort.is_set():
        raise SystemExit("Stopped: the predictor or stop policy failed (PredictorError in the "
                         "trajectories). Fix it, then rerun with --retry-errors.")


if __name__ == "__main__":
    main()
