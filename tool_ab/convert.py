"""Convert one arm's runs into the shared trajectory format.

    python tool_ab/convert.py --arm baseline --report path/to/eval_report.json

Reads data/tool_ab/runs/<arm>/*/*.traj.json and an evaluation report, and writes
data/tool_ab/qwen-<arm>.jsonl (one line per step), qwen-<arm>.stops.json with what the
plug-in recorded for each run (predictions, continue probabilities, s), and, for runs saved
with `save_diffs`, qwen-<arm>.diffs.jsonl (the repository diff after each step).

The report can be evaluate.py's or the SWE-bench harness report ("resolved_ids" plus
"unresolved_ids"), or a per-task mapping {task_id: {"resolved": bool}}. A task the report
did not score (a grading error, or missing from the report) is left out, not counted as
unresolved; only a report with nothing but "resolved_ids" cannot say which tasks it scored.

Outcome -1 means "stopped early, not evaluated": the plug-in stopped the run, so it has no
final result and the estimator does not need one. Runs that ended in an infrastructure error
(see shortstop_plugin/status.py, including runs that lost the sandbox connection) and unscored
tasks are left out and listed in qwen-<arm>.errors.json; rerun the first with
`run_arm.py --retry-errors`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shortstop_plugin.status import NO_TRAJECTORY, infra_error_of, tasks_without_trajectory
from shortstop_plugin.steps import UNKNOWN_OUTCOME, messages_to_steps, to_records

REPO = Path(__file__).resolve().parent.parent
EMPTY_STEP = {"tokens_in": 0, "tokens_out": 0, "action": "", "observation": ""}
NOT_SCORED = "NotScored"


def load_report(report_path: Path) -> tuple[set[str], set[str] | None]:
    """(resolved tasks, scored tasks); scored is None when the report cannot tell."""
    data = json.loads(Path(report_path).read_text())
    if "resolved_ids" in data:
        resolved = set(data["resolved_ids"])
        if "unresolved_ids" not in data:
            return resolved, None
        return resolved, resolved | set(data["unresolved_ids"]) | set(data.get("empty_patch_ids") or [])
    scored = {task for task, result in data.items() if isinstance(result, dict) and "resolved" in result}
    return {task for task in scored if data[task]["resolved"]}, scored


def load_resolved(report_path: Path) -> set[str]:
    return load_report(report_path)[0]


def convert_arm(runs_dir: Path, resolved: set[str], agent: str,
                scored: set[str] | None = None) -> tuple[list[dict], dict, dict]:
    """Returns (step records, stop records by task, left-out tasks with the reason)."""
    records: list[dict] = []
    stops: dict[str, dict] = {}
    errors: dict[str, str] = {task: NO_TRAJECTORY for task in tasks_without_trajectory(runs_dir)}
    for traj_path in sorted(Path(runs_dir).glob("*/*.traj.json")):
        traj = json.loads(traj_path.read_text())
        info = traj.get("info") or {}
        task_id = traj.get("instance_id") or traj_path.parent.name
        error = infra_error_of(traj)
        if error is not None:
            errors[task_id] = error
            continue
        record = info.get("shortstop")
        if record and record.get("stopped_early"):
            outcome = UNKNOWN_OUTCOME
        elif scored is not None and task_id not in scored:
            errors[task_id] = NOT_SCORED
            continue
        else:
            outcome = int(task_id in resolved)
        steps = messages_to_steps(traj.get("messages") or []) or [EMPTY_STEP]  # keep empty runs
        records += to_records(steps, task_id=task_id, agent=agent, outcome=outcome)
        if record:
            stops[task_id] = {**record, "exit_status": info.get("exit_status")}
    return records, stops, errors


def collect_diffs(runs_dir: Path, agent: str, tasks: set[str]) -> list[dict]:
    """One record per step for the given tasks' runs saved with save_diffs."""
    rows = []
    for traj_path in sorted(Path(runs_dir).glob("*/*.traj.json")):
        traj = json.loads(traj_path.read_text())
        task_id = traj.get("instance_id") or traj_path.parent.name
        if task_id in tasks:
            rows += [{"task_id": task_id, "agent": agent, **d} for d in (traj.get("info") or {}).get("step_diffs") or []]
    return rows


def write_arm(records: list[dict], stops: dict, out_dir: Path, agent: str,
              errors: dict | None = None, diffs: list[dict] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = agent.replace("/", "_")
    jsonl = out_dir / f"{name}.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
    (out_dir / f"{name}.stops.json").write_text(json.dumps(stops, indent=1))
    (out_dir / f"{name}.errors.json").write_text(json.dumps(errors or {}, indent=1))
    if diffs:
        (out_dir / f"{name}.diffs.jsonl").write_text("".join(json.dumps(d) + "\n" for d in diffs))
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
    resolved, scored = load_report(args.report)
    records, stops, errors = convert_arm(runs, resolved, agent, scored)
    diffs = collect_diffs(runs, agent, {r["task_id"] for r in records})
    path = write_arm(records, stops, args.out, agent, errors, diffs)
    n_runs = len({r["task_id"] for r in records})
    n_resolved = len({r["task_id"] for r in records if r["outcome"] == 1})
    print(f"{agent}: {n_runs} runs, {n_resolved} resolved, {len(records)} steps -> {path}")
    if errors:
        print(f"  {len(errors)} runs left out (infrastructure errors or not scored): {sorted(set(errors.values()))}")


if __name__ == "__main__":
    main()
