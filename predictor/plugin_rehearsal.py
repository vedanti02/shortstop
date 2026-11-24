"""Rehearse a live batch offline: Rishi's real plug-in and plugin_adapter.py, on logged runs.

    <python with mini-swe-agent 2.4.6> -I predictor/plugin_rehearsal.py --tool-ab <tool_ab folder> \
        --arm baseline --data data/tool_ab --out data/predictor/rehearsal \
        [--calibrate-from data/tool_ab/qwen-baseline.jsonl] [--seed 0] [--workers 8] [--limit N]

For each run logged in <data>/qwen-<arm>.jsonl, StoppingAgent (tool_ab/shortstop_plugin) runs
with a scripted model that replays the run's actions with their logged token counts, and an
environment that returns the logged outputs. Nothing is executed: the commands ran in another
machine's sandbox and must not run here. The plug-in loads the adapter by dotted path through its
own policies.load_shared, as tool_ab/run_arm.py does, and decides before every step whether to go
on, exactly as it would live. A run it lets go on ends as the logged run did: submitted, at the step
limit, or with a context overflow on the next call (the step that never completes).

Writes, under <out>/<arm>/:
    runs/<task>/<task>.traj.json   what the plug-in saves: exit status, stop record, messages
    decisions.jsonl                the adapter's log
    qwen-<arm>.jsonl, .stops.json  tool_ab/convert.py's output, graded with <data>/eval/<arm>.json

and checks that the plug-in rebuilt every logged step exactly, so that every prediction it recorded
equals the runtime's offline prediction for that step. Then check the output with
predictor/live_monitor.py: `status --log <out>/<arm>/decisions.jsonl --runs <out>/<arm>/runs` and
`report --data <out>/<arm> --arms <arm> --truth <data>`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class ContextWindowExceededError(Exception):
    """Stands in for the model call that overflowed the context in the logged run."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tool-ab", type=Path, required=True, help="Rishi's tool_ab folder (PR #3)")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "tool_ab")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "predictor" / "rehearsal")
    parser.add_argument("--runtime", default="data/predictor/runtime/all")
    parser.add_argument("--calibrate-from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    args = parser.parse_args()

    sys.path.insert(0, str(args.tool_ab.resolve()))       # shortstop_plugin and convert
    sys.path.insert(0, str(REPO / "src"))                 # shortstop
    sys.path.insert(0, str(REPO / "predictor"))           # plugin_adapter, as the plug-in imports it
    import convert
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel
    from plugin_adapter import read_outcomes
    from runtime import Runtime
    from shortstop_plugin.agent import StoppingAgent
    from shortstop_plugin.policies import load_shared

    from shortstop import load_trajectories

    class ReplayModel(DeterministicModel):
        """The logged actions in order, each as one command with the logged token counts."""

        def __init__(self, steps, overflow: bool):
            outputs = [{"role": "assistant", "content": "", "extra": {"actions": [{"command": s.action}], "cost": 0.0},
                        "usage": {"prompt_tokens": s.tokens_in, "completion_tokens": s.tokens_out}} for s in steps]
            super().__init__(outputs=outputs, observation_template="{{output.output}}", cost_per_call=0.0)
            self.overflow = overflow

        def query(self, messages, **kwargs) -> dict:
            self.current_index += 1
            if self.current_index < len(self.config.outputs):
                return json.loads(json.dumps(self.config.outputs[self.current_index]))
            if self.overflow:
                raise ContextWindowExceededError("the logged run overflowed the context on this call")
            raise RuntimeError("the replay went past the logged steps")

    class ReplayEnvironment(LocalEnvironment):
        """The logged outputs in order. Never runs a command."""

        def __init__(self, steps, submits: bool):
            super().__init__()
            self.outputs, self.submits, self.i = [s.observation for s in steps], submits, -1

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            self.i += 1
            last = self.i == len(self.outputs) - 1
            text = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" if self.submits and last else self.outputs[self.i]
            output = {"output": text, "returncode": 0, "exception_info": ""}
            self._check_finished(output)  # mini-swe-agent's own test for a submission
            return output

    agent_name = f"qwen-{args.arm}"
    runs = {t: s for (t, _), s in load_trajectories(args.data / f"{agent_name}.jsonl").items()}
    logged = json.loads((args.data / f"{agent_name}.stops.json").read_text())
    tasks = sorted(runs)[: args.limit or None]
    out = args.out / args.arm
    if out.exists():
        shutil.rmtree(out)
    (out / "runs").mkdir(parents=True)
    settings = {"runtime": args.runtime, "tasks": tasks, "arm": args.arm, "log": str(out / "decisions.jsonl")}
    if args.calibrate_from:
        settings["calibrate_from"] = args.calibrate_from
    # as tool_ab/run_arm.py's preload_stop_components: build both, policy first, before any run
    # starts; the settings go to the predictor only, as recommended in plugin_adapter.py
    load_shared("plugin_adapter.policy", {})
    load_shared("plugin_adapter.predictor", settings)

    def run_one(task: str) -> dict:
        steps, ending = runs[task], logged[task]["exit_status"]
        agent = StoppingAgent(
            ReplayModel(steps, overflow=ending == "ContextWindowExceededError"),
            ReplayEnvironment(steps, submits=ending == "Submitted"),
            system_template="Replay of a logged run.", instance_template="{{task}}", instance_id=task,
            step_limit=len(steps) if ending == "LimitsExceeded" else 0, cost_limit=0,
            predictor="plugin_adapter.predictor", predictor_kwargs=settings,
            stop_policy="plugin_adapter.policy", stop_policy_kwargs={},
            seed=args.seed, agent_name=agent_name)
        try:
            agent.run(task)
        except ContextWindowExceededError:
            pass  # recorded as the run's exit status, as the batch runner does
        traj = {"instance_id": task, **agent.serialize()}
        (out / "runs" / task).mkdir()
        (out / "runs" / task / f"{task}.traj.json").write_text(json.dumps(traj))
        return {"task": task, "exit_status": traj["info"]["exit_status"], "logged": ending,
                "record": traj["info"]["shortstop"], "prefix": agent._steps_so_far()}

    with ThreadPoolExecutor(args.workers) as pool:
        done = list(pool.map(run_one, tasks))

    # the plug-in must have rebuilt the logged steps exactly, so its predictions are the offline ones
    rt = Runtime.load(REPO / args.runtime)
    if args.calibrate_from:
        rt = rt.calibrate(read_outcomes(REPO / args.calibrate_from))
    worst, rebuilt = 0.0, 0
    for d in done:
        steps, record = runs[d["task"]], d["record"]
        for k, f in enumerate(record["predictions"]):
            state = rt.tracker(d["task"])
            for step in steps[:k]:
                state.update(vars(step))
            worst = max(worst, abs(rt.predict(d["task"], state)[0] - f))
        same = all(a["action"] == b.action and a["observation"] == b.observation and a["tokens_in"] == b.tokens_in
                   and a["tokens_out"] == b.tokens_out for a, b in zip(d["prefix"], steps))
        rebuilt += same

    resolved, scored = convert.load_report(args.data / "eval" / f"{args.arm}.json")
    records, stops, errors = convert.convert_arm(out / "runs", resolved, agent_name, scored)
    convert.write_arm(records, stops, out, agent_name, errors)

    stopped = [d for d in done if d["record"]["stopped_early"]]
    ended_as_logged = sum(d["exit_status"] == d["logged"] for d in done if not d["record"]["stopped_early"])
    spent = sum(r["tokens_in"] + r["tokens_out"] for r in records)
    full = sum(s.tokens_in + s.tokens_out for t in tasks for s in runs[t])
    print(f"{args.arm}: {len(done)} runs through the plug-in, {len(stopped)} stopped early; "
          f"{ended_as_logged} of {len(done) - len(stopped)} others ended as logged")
    print(f"steps rebuilt exactly in {rebuilt} of {len(done)} runs; largest gap between a recorded "
          f"prediction and the offline one: {worst:.1e}")
    print(f"tokens spent {spent / 1e6:.2f}M of the logged runs' {full / 1e6:.2f}M ({spent / full:.1%}); "
          f"converted: {len({r['task_id'] for r in records})} runs, {len(errors)} left out")
    print(f"exit statuses: {dict(sorted(Counter(d['exit_status'] for d in done).items()))}")
    if worst > 1e-9 or rebuilt != len(done):
        raise SystemExit("the plug-in's steps or predictions differ from the logged ones")


if __name__ == "__main__":
    main()
