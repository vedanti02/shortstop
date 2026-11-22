"""Run one arm of the tool A/B experiment on SWE-bench Verified with the shortstop plug-in.

    python tool_ab/run_arm.py --arm baseline --slice 0:20 -w 8
    python tool_ab/run_arm.py --arm search --slice 0:100 -w 16

This is `mini-extra swebench` with three differences. The agent is the plug-in
(StoppingAgent; mini-swe-agent's batch runner ignores `agent_class`, so its agent is
swapped here). `swerex_modal` runs commands in bash with the image's conda environment
(see shortstop_plugin/environment.py). The config is configs/base.yaml plus the arm's overlay. Tasks come from
configs/tasks.txt in a fixed order, so `--slice 0:20` is the same 20 tasks for every arm.

Outputs go to data/tool_ab/runs/<arm>/: one <task>/<task>.traj.json per run and preds.json
for evaluation. Finished tasks are skipped on a rerun unless --redo-existing is given;
--retry-errors reruns only the tasks that ended in an infrastructure error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
    return [str(CONFIGS / "base.yaml"), *(str(CONFIGS / f) for f in ARMS[arm]), *overrides]


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
    from shortstop_plugin.status import is_infra_error, tasks_without_trajectory

    preds_path = Path(output_dir) / "preds.json"
    if not preds_path.exists():
        return []
    errored = tasks_without_trajectory(output_dir)
    for traj_path in sorted(Path(output_dir).glob("*/*.traj.json")):
        info = json.loads(traj_path.read_text()).get("info") or {}
        if is_infra_error(info.get("exit_status")):
            errored.append(traj_path.parent.name)
    preds = json.loads(preds_path.read_text())
    kept = {task: pred for task, pred in preds.items() if task not in errored}
    preds_path.write_text(json.dumps(kept, indent=2))
    return errored


def make_progress_agent() -> type:
    """StoppingAgent with the progress reporting the batch runner expects; stops its environment."""
    from shortstop_plugin.agent import StoppingAgent

    class ProgressStoppingAgent(StoppingAgent):
        def __init__(self, *args, progress_manager, instance_id: str = "", **kwargs):
            super().__init__(*args, instance_id=instance_id, **kwargs)
            self.progress_manager = progress_manager

        def step(self) -> list[dict]:
            self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d}")
            return super().step()

        def run(self, *args, **kwargs) -> dict:
            # The batch runner never stops environments, so each Modal sandbox would outlive its run.
            try:
                return super().run(*args, **kwargs)
            finally:
                stop = getattr(self.env, "stop", None)
                if stop is not None:
                    stop()

    return ProgressStoppingAgent


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
    if args.retry_errors:
        print(f"retrying {len(forget_errored_runs(output))} errored tasks")
    swebench.ProgressTrackingAgent = make_progress_agent()
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
        config_spec=config_specs(args.arm, tuple(args.config)),
        environment_class=None,
    )


if __name__ == "__main__":
    main()
