"""Convert one arm's runs into the shared trajectory format.

    python tool_ab/convert.py --arm baseline --report path/to/eval_report.json

Reads data/tool_ab/runs/<arm>/*/*.traj.json and an evaluation report, and writes
data/tool_ab/qwen-<arm>.jsonl (one line per step) plus qwen-<arm>.stops.json with what the
plug-in recorded for each run (predictions, continue probabilities, steps run).

The report can be the SWE-bench harness or sb-cli report (a "resolved_ids" list) or a
per-task mapping {task_id: {"resolved": bool}}. A task missing from the report counts as
unresolved. A run that the plug-in stopped early was never evaluated, so its outcome is
written as -1 (unknown); the estimator does not need it. Runs that ended in an
infrastructure error (see shortstop_plugin/status.py) are left out and listed in
qwen-<arm>.errors.json; rerun them with `run_arm.py --retry-errors`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shortstop_plugin.status import NO_TRAJECTORY, is_infra_error, tasks_without_trajectory
from shortstop_plugin.steps import UNKNOWN_OUTCOME, messages_to_steps, to_records

REPO = Path(__file__).resolve().parent.parent
EMPTY_STEP = {"tokens_in": 0, "tokens_out": 0, "action": "", "observation": ""}


def load_resolved(report_path: Path) -> set[str]:
    data = json.loads(Path(report_path).read_text())
    if "resolved_ids" in data:
        return set(data["resolved_ids"])
    return {task for task, result in data.items() if isinstance(result, dict) and result.get("resolved")}


def convert_arm(runs_dir: Path, resolved: set[str], agent: str) -> tuple[list[dict], dict, dict]:
    """Returns (step records, stop records by task, infrastructure errors by task)."""
    records: list[dict] = []
    stops: dict[str, dict] = {}
    errors: dict[str, str] = {task: NO_TRAJECTORY for task in tasks_without_trajectory(runs_dir)}
    for traj_path in sorted(Path(runs_dir).glob("*/*.traj.json")):
        traj = json.loads(traj_path.read_text())
        info = traj.get("info") or {}
        task_id = traj.get("instance_id") or traj_path.parent.name
        if is_infra_error(info.get("exit_status")):
            errors[task_id] = str(info.get("exit_status"))
            continue
        record = info.get("shortstop")
        if record and record.get("stopped_early"):
            outcome = UNKNOWN_OUTCOME
        else:
            outcome = int(task_id in resolved)
        steps = messages_to_steps(traj.get("messages") or []) or [EMPTY_STEP]  # keep empty runs
        records += to_records(steps, task_id=task_id, agent=agent, outcome=outcome)
        if record:
            stops[task_id] = {**record, "exit_status": info.get("exit_status")}
    return records, stops, errors


def write_arm(records: list[dict], stops: dict, out_dir: Path, agent: str,
              errors: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = agent.replace("/", "_")
    jsonl = out_dir / f"{name}.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
    (out_dir / f"{name}.stops.json").write_text(json.dumps(stops, indent=1))
    (out_dir / f"{name}.errors.json").write_text(json.dumps(errors or {}, indent=1))
    return jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=None)
    parser.add_argument("--agent", default=None, help="Default: qwen-<arm>")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "tool_ab")
    args = parser.parse_args()

    runs = args.runs or REPO / "data" / "tool_ab" / "runs" / args.arm
    agent = args.agent or f"qwen-{args.arm}"
    records, stops, errors = convert_arm(runs, load_resolved(args.report), agent)
    path = write_arm(records, stops, args.out, agent, errors)
    n_runs = len({r["task_id"] for r in records})
    n_resolved = len({r["task_id"] for r in records if r["outcome"] == 1})
    print(f"{agent}: {n_runs} runs, {n_resolved} resolved, {len(records)} steps -> {path}")
    if errors:
        print(f"  {len(errors)} runs left out for infrastructure errors: {sorted(set(errors.values()))}")


if __name__ == "__main__":
    main()
