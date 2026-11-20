"""Turn converted trajectories into one small feature row per step prefix.

    python coverage/features.py

For a run of T steps there are T rows, k = 0..T-1. Row k describes what is known after k
steps (so k = 0 uses nothing from the run) and carries the cost of taking step k + 1.
Output: data/swebench_verified/features.pkl
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

TEST_RUN = re.compile(r"pytest|runtests|manage\.py test|unittest|\btox\b|bin/test\b")
EDIT = re.compile(r"sed -i|cat\s*<<|cat\s*>|\btee\b|git apply|\bpatch\b|str_replace|write_text|\.write\(")
ERROR = re.compile(r"Traceback \(most recent call last\)|\bError\b|\bFAILED\b|\bfailed\b")
RETURNCODE = re.compile(r"<returncode>(-?\d+)</returncode>")
PASSED = re.compile(r"\b\d+ passed\b|\bOK\b")

COUNTS = ["test_runs", "edits", "errors", "nonzero_exits", "tests_passed"]


def run_rows(steps: list[dict]) -> list[dict]:
    """Feature rows for one run, given its step records in order."""
    rows = []
    counts = dict.fromkeys(COUNTS, 0)
    tokens = 0
    last_nonzero = 0
    for k, step in enumerate(steps):
        cost = step["tokens_in"] + step["tokens_out"]
        rows.append(
            {"task_id": step["task_id"], "agent": step["agent"], "k": k, "tokens_so_far": tokens,
             "last_nonzero_exit": last_nonzero, **counts, "next_cost": cost,
             "outcome": step["outcome"]}
        )
        action, observation = step["action"], step["observation"]
        is_test = bool(TEST_RUN.search(action))
        code = RETURNCODE.search(observation)
        last_nonzero = int(bool(code) and code.group(1) != "0")
        counts["test_runs"] += is_test
        counts["edits"] += bool(EDIT.search(action))
        counts["errors"] += bool(ERROR.search(observation))
        counts["nonzero_exits"] += last_nonzero
        counts["tests_passed"] += is_test and bool(PASSED.search(observation)) and not last_nonzero
        tokens += cost
    return rows


def agent_rows(path: Path) -> list[dict]:
    rows, steps, key = [], [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            step = json.loads(line)
            if step["task_id"] != key and steps:
                rows += run_rows(steps)
                steps = []
            key = step["task_id"]
            steps.append(step)
    return rows + (run_rows(steps) if steps else [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    args = parser.parse_args()
    frames = []
    for path in sorted(args.data.glob("*.jsonl.gz")):
        frames.append(pd.DataFrame(agent_rows(path)))
        print(f"{path.name}: {len(frames[-1])} rows", flush=True)
    features = pd.concat(frames, ignore_index=True)
    features.to_pickle(args.data / "features.pkl")
    print(f"wrote {len(features)} rows for {features['agent'].nunique()} agents")


if __name__ == "__main__":
    main()
