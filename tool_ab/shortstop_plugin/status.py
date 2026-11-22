"""Which run endings are the agent's own outcome and which are infrastructure failures.

An agent outcome (it submitted, ran out of steps, kept breaking the tool format, overflowed
its context, or was stopped by the plug-in) is a legitimate result for that arm. Anything
else, such as a sandbox that failed to start or a dropped connection to the model server,
says nothing about the tool change: those runs are excluded and should be rerun.
"""

from __future__ import annotations

import json
from pathlib import Path

AGENT_OUTCOMES = frozenset({
    "Submitted",
    "LimitsExceeded",
    "TimeExceeded",
    "RepeatedFormatError",
    "ContextWindowExceededError",
    "EarlyStopped",
})


def is_infra_error(exit_status: str | None) -> bool:
    return exit_status not in AGENT_OUTCOMES


NO_TRAJECTORY = "NoTrajectory"


def tasks_without_trajectory(runs_dir: Path | str) -> list[str]:
    """Tasks the runner recorded in preds.json but never saved a trajectory for.

    mini-swe-agent writes an empty prediction and no .traj.json when the sandbox fails
    to start, so these runs are infrastructure errors too.
    """
    preds_path = Path(runs_dir) / "preds.json"
    if not preds_path.exists():
        return []
    preds = json.loads(preds_path.read_text())
    return sorted(task for task in preds if not (Path(runs_dir) / task / f"{task}.traj.json").exists())
