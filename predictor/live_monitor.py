"""Watch a live batch run through plugin_adapter.py, and score it once the runs are graded.

    python -I predictor/live_monitor.py status --log data/predictor/live/baseline.decisions.jsonl \
        --runs data/tool_ab/runs/baseline [--reference data/tool_ab/qwen-baseline.jsonl] \
        [--grades data/tool_ab/eval/baseline.json] [--watch 60] [--json] [--strict]
    python -I predictor/live_monitor.py report --data data/tool_ab --arms baseline descriptions \
        [--baseline baseline] [--truth data/tool_ab_full] [--logs data/predictor/live] [--json]

status   During the batch, from the adapter's decision log and the plug-in's run folder (a run's
         <task>/<task>.traj.json appears when it ends): progress, tokens spent, a running score on
         the runs that have ended, and alerts. --reference gives earlier full runs of the agent, so
         tokens spent can be compared with what full runs of the same tasks cost; without it the
         cost model's step-0 estimate is shown, which can be far off for a new agent. --grades
         gives the grading report, so finished runs count in the running score. --watch N
         refreshes every N seconds.
report   After grading, from tool_ab/convert.py's output (<agent>.jsonl, .stops.json and
         .errors.json in --data): each arm's score with its 95% interval (standard design: mean
         +/- 1.96 sd / sqrt(n) of the per-task estimates), and each arm minus the baseline arm,
         paired by task. --truth points at full runs of the same arms, to show the true values
         and the share of full-run tokens spent.

Every stop record is checked: as many predictions as continue probabilities, every prediction in
[0, 1] and probability in (0, 1], s = len(continue_probs) - stopped_early, a reach probability
never below the floor, and outcome -1 exactly for the runs stopped early. With the decision log,
each record must also hold exactly the values the adapter logged.

Alerts are critical (failed predictor or policy calls, records that fail a check or differ from
the log), warnings (slow calls, stalled runs, predictions that look miscalibrated for this agent,
settings that changed between sessions, seeds or shifts that differ between A/B arms) or info.
Exit status: 2 if any alert is critical, 1 with --strict if any is a warning, otherwise 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortstop import rr_estimate

Z = 1.96
FLOOR = 0.05
# how a run can end by itself; any other exit status is an infrastructure error
# (tool_ab/shortstop_plugin/status.py, AGENT_OUTCOMES)
AGENT_ENDINGS = {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError",
                 "ContextWindowExceededError", "EarlyStopped"}


# -- stop records ----------------------------------------------------------------------------------

def estimator_steps(record: dict) -> int:
    return len(record["continue_probs"]) - int(bool(record["stopped_early"]))


def check_record(record: dict, outcome: int | None = None, floor: float = FLOOR) -> list[str]:
    """What is wrong with one run's stop record (empty if nothing)."""
    f, p = record.get("predictions", []), record.get("continue_probs", [])
    problems = []
    if len(f) != len(p):
        problems.append(f"{len(f)} predictions but {len(p)} continue probabilities")
    if not p:
        problems.append("no draw was recorded, so there is no step-0 prediction")
    if any(not 0 <= x <= 1 for x in f):
        problems.append("a prediction outside [0, 1]")
    if any(not 0 < x <= 1 for x in p):
        problems.append("a continue probability outside (0, 1]")
    elif p and float(np.prod(p)) < floor * (1 - 1e-9):
        problems.append(f"the reach probability fell to {np.prod(p):.4f}, below the floor {floor}")
    if "s" in record and record["s"] != estimator_steps(record):
        problems.append(f"s = {record['s']}, but len(continue_probs) - stopped_early = {estimator_steps(record)}")
    if outcome is not None and (outcome == -1) != bool(record["stopped_early"]):
        problems.append(f"outcome {outcome} for a run {'stopped early' if record['stopped_early'] else 'not stopped'}")
    return problems


def record_estimate(record: dict, outcome: float | None) -> float:
    """The run's randomized-truncation estimate: f_0..f_s are the predictions for a stopped run;
    for a run that ended by itself f_s is the outcome, and an attempted step that never completed
    counts as step s (predictor/README.md, "When a step never completes")."""
    s, f = estimator_steps(record), list(record["predictions"])
    if record["stopped_early"]:
        values = f[: s + 1]
    else:
        if outcome not in (0, 1):
            raise ValueError("a run that was not stopped needs its 0/1 outcome")
        values = f[:s] + [float(outcome)]
    return rr_estimate(values, record["continue_probs"], s)


def interval(values) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan")), float("nan")
    return float(values.mean()), float(Z * values.std(ddof=1) / np.sqrt(len(values)))


# -- reading ---------------------------------------------------------------------------------------

def read_log(path: Path) -> list[dict]:
    """The adapter's events; a half-written last line (the batch is still running) is skipped."""
    events = []
    lines = Path(path).read_text().splitlines() if Path(path).exists() else []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return events


def usage_tokens(messages: list[dict]) -> int:
    """Tokens logged for every model call in a mini-swe-agent trajectory, failed calls included."""
    total = 0
    for msg in messages:
        response = (msg.get("extra") or {}).get("response")
        usage = msg.get("usage") or (response.get("usage") if isinstance(response, dict) else None)
        if isinstance(usage, dict):
            total += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            total += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return total


def read_runs(folder: Path | None) -> dict[str, dict]:
    """Ended runs in the plug-in's output folder: exit status, stop record, tokens."""
    out = {}
    if folder is None or not Path(folder).exists():
        return out
    for path in sorted(Path(folder).glob("*/*.traj.json")):
        try:
            traj = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # being written
        info = traj.get("info") or {}
        task = traj.get("instance_id") or path.parent.name
        out[task] = {"exit_status": info.get("exit_status"), "record": info.get("shortstop"),
                     "tokens": usage_tokens(traj.get("messages") or [])}
    return out


def read_grades(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if "resolved_ids" in data:
        resolved = set(data["resolved_ids"])
        return {t: int(t in resolved) for t in resolved | set(data.get("unresolved_ids") or [])}
    return {t: int(bool(v.get("resolved"))) for t, v in data.items() if isinstance(v, dict) and "resolved" in v}


def read_arm(folder: Path, agent: str) -> dict[str, dict]:
    """tool_ab/convert.py's output for one agent: per task, the stop record (None without a
    .stops.json, as for plain full runs), outcome and tokens. Tasks listed in <agent>.errors.json
    (infrastructure errors, unscored) are left out."""
    folder = Path(folder)
    stops_path = folder / f"{agent}.stops.json"
    stops = json.loads(stops_path.read_text()) if stops_path.exists() else None
    errors_path = folder / f"{agent}.errors.json"
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else {}
    runs: dict[str, dict] = {}
    with open(folder / f"{agent}.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            run = runs.setdefault(row["task_id"], {"outcome": row["outcome"], "tokens": 0, "rows": 0})
            run["tokens"] += int(row["tokens_in"]) + int(row["tokens_out"])
            run["rows"] += 1
    return {t: {**runs[t], "record": stops.get(t) if stops is not None else None}
            for t in sorted(runs) if t not in errors}


def read_reference(path: Path | None) -> dict[str, int]:
    """Full-run tokens per task from a shared-format trajectory file of earlier full runs."""
    out: dict[str, int] = defaultdict(int)
    if path is None:
        return {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            out[row["task_id"]] += int(row["tokens_in"]) + int(row["tokens_out"])
    return dict(out)


# -- status ------------------------------------------------------------------------------------------

def alert(level: str, check: str, message: str) -> dict:
    return {"level": level, "check": check, "message": message}


def latest_attempts(events: list[dict]) -> dict[str, list[dict]]:
    """Per task, the decisions of its latest run (after its last run_start), in step order."""
    out: dict[str, list[dict]] = {}
    for e in events:
        if e["event"] == "run_start":
            out[e["task_id"]] = []
        elif e["event"] == "decision":
            out.setdefault(e["task_id"], []).append(e)
    return out


def status(log: Path, runs_dir: Path | None = None, reference: Path | None = None, grades: Path | None = None,
           latency_ms: float = 1000.0, stall_minutes: float = 30.0, now: datetime | None = None) -> dict:
    events = read_log(log)
    sessions = [e for e in events if e["event"] == "session"]
    if not sessions:
        return {"alerts": [alert("critical", "log", f"no session in {log}: is the adapter configured to log there?")]}
    session = sessions[-1]
    planned = session["task_ids"]
    decisions = [e for e in events if e["event"] == "decision"]
    errors = [e for e in events if e["event"] == "error"]
    attempts: dict[str, int] = defaultdict(int)
    for e in events:
        if e["event"] == "run_start":
            attempts[e["task_id"]] += 1
    latest = latest_attempts(events)
    f0 = {e["task_id"]: e["f0"] for e in events if e["event"] == "run_start"}
    ended = read_runs(runs_dir)
    outcomes = read_grades(grades)
    now = now or datetime.now(timezone.utc)
    alerts = []

    # failures and settings
    if errors:
        shown = "; ".join(f"{e['task_id']} step {e['k']} ({e['where']}): {e['error']}" for e in errors[:3])
        alerts.append(alert("critical", "errors", f"{len(errors)} failed call(s): {shown}"))
    keys = ("shift", "tasks", "kappa", "floor", "success_models", "cost_scale")
    changed = [k for k in keys if len({json.dumps(s.get(k)) for s in sessions}) > 1]
    if changed:
        alerts.append(alert("warning", "settings", f"{len(sessions)} sessions differ in {', '.join(changed)}: "
                                                   "keep the calibration and the task list fixed for a batch"))

    # ended runs: records checked against the rules and against the log
    bad, mismatched, unlogged, est, ended_f0 = [], [], [], [], []
    endings: dict[str, int] = defaultdict(int)
    for task, run in ended.items():
        endings[run["exit_status"] or "unknown"] += 1
        record = run["record"]
        if record is None or run["exit_status"] not in AGENT_ENDINGS:
            continue
        problems = check_record(record, floor=session["floor"])
        if problems:
            bad.append(f"{task}: {problems[0]}")
            continue
        logged = latest.get(task)
        if not logged:
            unlogged.append(task)
        else:
            same = (len(logged) == len(record["continue_probs"])
                    and all(abs(e["f"] - f) <= 1e-9 and abs(e["p"] - p) <= 1e-9
                            for e, f, p in zip(logged, record["predictions"], record["continue_probs"])))
            if not same:
                mismatched.append(task)
        if record["stopped_early"]:
            est.append(record_estimate(record, None))
        elif task in outcomes:
            est.append(record_estimate(record, outcomes[task]))
        else:
            continue
        ended_f0.append(record["predictions"][0])
    if bad:
        alerts.append(alert("critical", "records", f"{len(bad)} stop record(s) fail a check, e.g. {bad[0]}"))
    if mismatched:
        alerts.append(alert("critical", "records vs log", f"{len(mismatched)} record(s) differ from the decisions "
                            f"the adapter logged, e.g. {', '.join(mismatched[:3])}: another predictor or policy?"))
    if unlogged:
        alerts.append(alert("warning", "records vs log", f"{len(unlogged)} ended run(s) have no decisions in this "
                            f"log, e.g. {', '.join(unlogged[:3])}"))
    infra = {k: v for k, v in endings.items() if k not in AGENT_ENDINGS}
    if infra:
        alerts.append(alert("warning", "endings", "runs ended in errors, rerun with --retry-errors: "
                            + ", ".join(f"{k} {v}" for k, v in sorted(infra.items()))))
    outside = sorted(set(ended) - set(planned))
    if outside:
        alerts.append(alert("warning", "tasks", f"{len(outside)} run(s) for tasks outside the plan, e.g. {outside[0]}"))

    # running runs: stalls
    stalled = []
    for task, logged in latest.items():
        if task not in ended and logged:
            age = (now - datetime.fromisoformat(logged[-1]["time"])).total_seconds() / 60
            if age > stall_minutes:
                stalled.append(f"{task} ({age:.0f} min at step {logged[-1]['k']})")
    if stalled:
        alerts.append(alert("warning", "stalled", f"{len(stalled)} run(s) without a decision for over "
                            f"{stall_minutes:.0f} min and not ended: {', '.join(stalled[:3])}"))

    # speed
    ms = np.array([e["total_ms"] for e in decisions]) if decisions else np.array([0.0])
    p95 = float(np.percentile(ms, 95))
    if p95 > latency_ms:
        alerts.append(alert("warning", "latency", f"95% of decisions take up to {p95:.0f} ms (limit {latency_ms:.0f})"))

    # a running score, and whether the predictions look calibrated on the same runs
    score, half = interval(est)
    if len(est) >= 20 and abs(score - float(np.mean(ended_f0))) > half:
        alerts.append(alert("warning", "calibration", f"on the {len(est)} runs scored so far the mean step-0 "
                            f"prediction is {np.mean(ended_f0):.2f}, outside the estimate {score:.2f} +/- {half:.2f}: "
                            "calibrate on earlier full runs of this agent"))

    if mismatched == [] and bad == [] and ended:
        alerts.append(alert("info", "records", f"{len(ended)} ended run(s) checked against the rules and the log"))
    floor_share = float(np.mean([e["at_floor"] for e in decisions])) if decisions else 0.0
    reruns = sorted(t for t, n in attempts.items() if n > 1)
    if reruns:
        alerts.append(alert("info", "reruns", f"{len(reruns)} task(s) started more than once, e.g. {reruns[0]}"))

    ref = read_reference(reference)
    spent = sum(r["tokens"] for r in ended.values()) + sum(
        logged[-1]["tokens_so_far"] for t, logged in latest.items() if t not in ended and logged)
    finished = [t for t, r in ended.items()
                if r["record"] and not r["record"]["stopped_early"] and r["exit_status"] in AGENT_ENDINGS]
    return {
        "arm": session["arm"], "sessions": len(sessions), "shift": session["shift"], "design": session["design"],
        "planned": len(planned), "started": len(latest), "running": len([t for t in latest if t not in ended]),
        "ended": len(ended), "endings": dict(endings), "decisions": len(decisions), "at_floor": floor_share,
        "latency_ms": {"median": float(np.median(ms)), "p95": p95, "max": float(ms.max())},
        "tokens_spent": spent,
        "full_run_tokens": sum(ref.get(t, 0) for t in planned) if ref else None,
        "predicted_full_run_tokens": session["predicted_full_tokens"],
        "mean_f0": session["mean_f0"], "mean_f0_started": float(np.mean(list(f0.values()))) if f0 else None,
        "scored": len(est), "awaiting_grades": sum(t not in outcomes for t in finished),
        "score": score, "half_width": half, "alerts": alerts,
    }


def render_status(s: dict) -> str:
    if "planned" not in s:
        return "\n".join(f"{a['level'].upper():9s}{a['message']}" for a in s["alerts"])
    full = s["full_run_tokens"]
    share = (f"{s['tokens_spent'] / full:.0%} of the earlier full runs' {full / 1e6:.1f}M" if full else
             f"cost model's step-0 estimate of full runs: {s['predicted_full_run_tokens'] / 1e6:.1f}M (rough)")
    endings = ", ".join(f"{k} {v}" for k, v in sorted(s["endings"].items())) or "none yet"
    ms = s["latency_ms"]
    lines = [
        (f"{s['arm']}: {s['planned']} tasks planned, standard design, shift {s['shift']:+.2f}, "
         f"mean step-0 prediction {s['mean_f0']:.2f} ({s['sessions']} session(s))"),
        f"runs: {s['started']} started, {s['running']} running, {s['ended']} ended ({endings})",
        f"tokens spent: {s['tokens_spent'] / 1e6:.2f}M ({share})",
        (f"decisions: {s['decisions']}, {s['at_floor']:.0%} at the floor; time per decision median "
         f"{ms['median']:.1f} ms, 95% {ms['p95']:.1f} ms, max {ms['max']:.1f} ms"),
    ]
    if s["scored"]:
        lines.append(f"running score on the {s['scored']} runs scored so far: {s['score']:.3f} +/- {s['half_width']:.3f}"
                     + (f" ({s['awaiting_grades']} finished runs await grading)" if s["awaiting_grades"] else ""))
    lines += [f"{a['level'].upper():9s}{a['check']}: {a['message']}" for a in s["alerts"]] or ["OK       no alerts"]
    return "\n".join(lines)


# -- report ------------------------------------------------------------------------------------------

def report(data: Path, arms: list[str], baseline: str | None = None, truth: Path | None = None,
           logs: Path | None = None, prefix: str = "qwen-") -> dict:
    out, alerts, per_task = {"arms": {}, "paired": {}}, [], {}
    seeds, shifts = {}, {}
    for arm in arms:
        agent = f"{prefix}{arm}"
        runs = read_arm(data, agent)
        log_path = Path(logs) / f"{arm}.decisions.jsonl" if logs else None
        if log_path and not log_path.exists():
            alerts.append(alert("warning", f"{arm} log", f"no decision log at {log_path}, so the records were not "
                                                         "compared with the adapter's decisions"))
        latest = latest_attempts(read_log(log_path)) if log_path else {}
        est, bad, mismatched = {}, [], []
        for task, run in runs.items():
            if run["record"] is None:
                bad.append(f"{task}: no stop record")
                continue
            problems = check_record(run["record"], run["outcome"])
            if problems:
                bad.append(f"{task}: {problems[0]}")
                continue
            if latest:
                logged = latest.get(task, [])
                rec = run["record"]
                if len(logged) != len(rec["continue_probs"]) or any(
                        abs(e["f"] - f) > 1e-9 or abs(e["p"] - p) > 1e-9
                        for e, f, p in zip(logged, rec["predictions"], rec["continue_probs"])):
                    mismatched.append(task)
            est[task] = record_estimate(run["record"], None if run["outcome"] == -1 else run["outcome"])
        if bad:
            alerts.append(alert("critical", f"{arm} records", f"{len(bad)} record(s) fail a check, e.g. {bad[0]}"))
        if mismatched:
            alerts.append(alert("critical", f"{arm} records vs log",
                                f"{len(mismatched)} record(s) differ from the adapter's log, e.g. {mismatched[0]}"))
        seeds[arm] = {r["record"].get("seed") for r in runs.values() if r["record"]}
        score, half = interval(list(est.values()))
        row = {"tasks": len(est), "score": score, "half_width": half, "interval": [score - half, score + half],
               "started": sum(estimator_steps(runs[t]["record"]) > 0 for t in est),
               "stopped_early": sum(bool(runs[t]["record"]["stopped_early"]) for t in est),
               "tokens": sum(runs[t]["tokens"] for t in est)}
        if logs:
            session = [e for e in read_log(Path(logs) / f"{arm}.decisions.jsonl") if e["event"] == "session"]
            shifts[arm] = {s["shift"] for s in session}
        if truth:
            full = read_arm(truth, agent)
            common = [t for t in est if t in full and full[t]["outcome"] in (0, 1)]
            true_rate = float(np.mean([full[t]["outcome"] for t in common])) if common else float("nan")
            row.update({"true_rate": true_rate, "covered": bool(abs(score - true_rate) <= half),
                        "full_run_tokens": sum(full[t]["tokens"] for t in common),
                        "tokens_share": row["tokens"] / max(sum(full[t]["tokens"] for t in common), 1)})
            per_task[arm + "/truth"] = {t: full[t]["outcome"] for t in common}
        out["arms"][arm] = row
        per_task[arm] = est
    if len({json.dumps(sorted(map(str, v))) for v in seeds.values()}) > 1 or any(len(v) > 1 for v in seeds.values()):
        alerts.append(alert("warning", "seeds", f"the arms' stop records use different seeds {seeds}: "
                                                "an A/B test needs the same seed in every arm"))
    if shifts and (len({json.dumps(sorted(v)) for v in shifts.values()}) > 1 or any(len(v) > 1 for v in shifts.values())):
        alerts.append(alert("warning", "shifts", f"the arms' calibration shifts differ {shifts}: an A/B test "
                                                 "needs the same shift (the baseline's) in every arm"))
    if baseline:
        for arm in arms:
            if arm == baseline:
                continue
            common = sorted(set(per_task[arm]) & set(per_task[baseline]))
            d = [per_task[arm][t] - per_task[baseline][t] for t in common]
            diff, half = interval(d)
            verdict = "better" if diff - half > 0 else "worse" if diff + half < 0 else "no detectable difference"
            row = {"tasks": len(common), "difference": diff, "half_width": half, "verdict": verdict}
            if truth and f"{arm}/truth" in per_task:
                ta, tb = per_task[f"{arm}/truth"], per_task[f"{baseline}/truth"]
                both = [t for t in common if t in ta and t in tb]
                row["true_difference"] = float(np.mean([ta[t] - tb[t] for t in both])) if both else float("nan")
            out["paired"][arm] = row
    if not any(a["level"] == "critical" for a in alerts):
        alerts.append(alert("info", "records", "every stop record passes the checks"))
    out["alerts"] = alerts
    return out


def render_report(r: dict) -> str:
    lines = ["| Arm | Tasks | Score (95% interval) | Started | Stopped early | Tokens | True rate | Share of full-run tokens |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, x in r["arms"].items():
        true = f"{x['true_rate']:.3f} ({'covered' if x['covered'] else 'missed'})" if "true_rate" in x else ""
        share = f"{x['tokens_share']:.1%}" if "tokens_share" in x else ""
        lines.append(f"| {arm} | {x['tasks']} | {x['score']:.3f} ({x['interval'][0]:.3f} to {x['interval'][1]:.3f}) | "
                     f"{x['started']} | {x['stopped_early']} | {x['tokens'] / 1e6:.2f}M | {true} | {share} |")
    if r["paired"]:
        lines += ["", "| Change | Tasks | Difference (95% interval) | Verdict | True difference |", "| --- | --- | --- | --- | --- |"]
        for arm, x in r["paired"].items():
            true = f"{x['true_difference']:+.3f}" if "true_difference" in x else ""
            lines.append(f"| {arm} | {x['tasks']} | {x['difference']:+.3f} ({x['difference'] - x['half_width']:+.3f} to "
                         f"{x['difference'] + x['half_width']:+.3f}) | {x['verdict']} | {true} |")
    lines += [""] + [f"{a['level'].upper():9s}{a['check']}: {a['message']}" for a in r["alerts"]]
    return "\n".join(lines)


def exit_code(alerts: list[dict], strict: bool) -> int:
    levels = {a["level"] for a in alerts}
    return 2 if "critical" in levels else 1 if strict and "warning" in levels else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("status", help="progress, tokens, running score and alerts during a batch")
    st.add_argument("--log", type=Path, required=True)
    st.add_argument("--runs", type=Path)
    st.add_argument("--reference", type=Path)
    st.add_argument("--grades", type=Path)
    st.add_argument("--latency-ms", type=float, default=1000.0)
    st.add_argument("--stall-minutes", type=float, default=30.0)
    st.add_argument("--watch", type=float, default=0.0)
    rp = sub.add_parser("report", help="scores and A/B differences after grading")
    rp.add_argument("--data", type=Path, required=True)
    rp.add_argument("--arms", nargs="+", required=True)
    rp.add_argument("--baseline")
    rp.add_argument("--truth", type=Path)
    rp.add_argument("--logs", type=Path)
    rp.add_argument("--prefix", default="qwen-")
    for p in (st, rp):
        p.add_argument("--json", action="store_true")
        p.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "report":
        r = report(args.data, args.arms, args.baseline, args.truth, args.logs, args.prefix)
        print(json.dumps(r, indent=1) if args.json else render_report(r))
        return exit_code(r["alerts"], args.strict)
    while True:
        s = status(args.log, args.runs, args.reference, args.grades, args.latency_ms, args.stall_minutes)
        print(json.dumps(s, indent=1) if args.json else render_status(s), flush=True)
        if not args.watch:
            return exit_code(s["alerts"], args.strict)
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    sys.exit(main())
