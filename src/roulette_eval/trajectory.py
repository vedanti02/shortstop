"""Trajectory format: one JSON line per step."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    task_id: str
    agent: str
    step: int  # starts at 1
    tokens_in: int
    tokens_out: int
    action: str
    observation: str
    outcome: int  # the run's final result (0 or 1), repeated on every step


def load_trajectories(path: str | Path) -> dict[tuple[str, str], list[Step]]:
    """Read a JSONL file into {(task_id, agent): [steps in order]}."""
    runs: dict[tuple[str, str], list[Step]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            if line.strip():
                step = Step(**json.loads(line))
                runs[(step.task_id, step.agent)].append(step)
    for steps in runs.values():
        steps.sort(key=lambda s: s.step)
    return dict(runs)
