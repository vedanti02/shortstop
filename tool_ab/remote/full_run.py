"""Run the tool A/B experiment entirely on Modal, so it keeps going with the laptop closed.

    modal deploy tool_ab/serving/serve_qwen.py        # the model server must be deployed
    modal deploy tool_ab/remote/full_run.py
    python tool_ab/remote/full_run.py launch --run-name full-100 --slice 0:100
    python tool_ab/remote/full_run.py launch --run-name smoke --slice 0:1 --arms baseline

`launch` spawns the job on the deployed app and returns at once. One Modal function per arm
runs run_arm.py, retries infrastructure errors once, scores with evaluate.py and converts
with convert.py; then analyze.py writes the verdicts. Everything lands in the Modal volume
`shortstop-runs` under <run-name>/ (STATUS.txt says how each arm ended). Fetch it with

    modal volume get shortstop-runs <run-name> data/tool_ab/remote/

The GPU server shuts itself down 10 minutes after the last request.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[2] if modal.is_local() else Path("/repo")  # Modal runs this file as /root/full_run.py
VOLUME_ROOT = Path("/vol")
APP_NAME = "shortstop-full-run"
ARMS = ["baseline", "descriptions", "search", "compressed"]
DAY_SECONDS = 24 * 60 * 60
COMMIT_EVERY_SECONDS = 10 * 60
EVAL_WORKERS = 8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "mini-swe-agent[modal]==2.4.6", "swe-rex==1.4.0", "swebench==4.1.0", "python-dotenv",
        "numpy", "pandas", "scikit-learn", "matplotlib",
    )
    .env({"PYTHONPATH": "/repo/src:/repo/tool_ab", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO / "src", "/repo/src", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "tool_ab", "/repo/tool_ab",
                   ignore=["**/__pycache__", ".env", "results", "tests", "remote"])
)
runs_volume = modal.Volume.from_name("shortstop-runs", create_if_missing=True)
app = modal.App(APP_NAME)


def arm_commands(arm: str, root: Path, slice_spec: str, workers: int) -> list[list[str]]:
    """The scripts one arm runs, in order: run, retry errors, score, convert."""
    tool_ab = "/repo/tool_ab"
    runs = root / "runs" / arm
    run = ["python", f"{tool_ab}/run_arm.py", "--arm", arm, "--slice", slice_spec,
           "-w", str(workers), "-o", str(runs)]
    return [
        run,
        [*run, "--retry-errors"],
        ["python", f"{tool_ab}/evaluate.py", "--preds", str(runs / "preds.json"), "--run-id", arm,
         "--out", str(root / "eval"), "-w", str(EVAL_WORKERS)],
        ["python", f"{tool_ab}/convert.py", "--arm", arm, "--report", str(root / "eval" / f"{arm}.json"),
         "--runs", str(runs), "--out", str(root)],
    ]


def analyze_command(root: Path, arms: list[str]) -> list[str] | None:
    others = [f"qwen-{a}" for a in arms if a != "baseline"]
    if "baseline" not in arms or not others:
        return None
    return ["python", "/repo/tool_ab/analyze.py", "--data", str(root), "--arms", *others,
            "--out", str(root / "results")]


def run_and_commit(command: list[str], log_path: Path) -> int:
    """Run a command, appending its output to log_path and committing the volume as it goes."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        last_commit = time.monotonic()
        while process.poll() is None:
            time.sleep(15)
            if time.monotonic() - last_commit > COMMIT_EVERY_SECONDS:
                runs_volume.commit()
                last_commit = time.monotonic()
    runs_volume.commit()
    return process.returncode


def point_at_model_server() -> None:
    url = modal.Function.from_name("shortstop-qwen", "serve").get_web_url()
    os.environ["HOSTED_VLLM_API_BASE"] = f"{url}/v1"
    os.environ["HOSTED_VLLM_API_KEY"] = os.environ["VLLM_API_KEY"]  # from the shortstop-vllm-key secret


@app.function(image=image, volumes={str(VOLUME_ROOT): runs_volume}, timeout=DAY_SECONDS,
              secrets=[modal.Secret.from_name("shortstop-vllm-key")], cpu=2, memory=4096)
def run_arm_job(arm: str, run_name: str, slice_spec: str, workers: int) -> str:
    point_at_model_server()
    root = VOLUME_ROOT / run_name
    notes = []
    report = root / "eval" / f"{arm}.json"
    for command in arm_commands(arm, root, slice_spec, workers):
        is_evaluate = command[1].endswith("evaluate.py")
        if is_evaluate:
            report.unlink(missing_ok=True)  # so convert can never grade against an earlier attempt's report
        code = run_and_commit(command, root / "logs" / f"{arm}.log")
        if code != 0 and is_evaluate and report.exists():
            notes.append(f"some tasks not graded (exit {code}); convert.py leaves them out")
        elif code != 0:
            return f"{arm}: FAILED (exit {code}) at: {' '.join(command[1:3])}"
    return f"{arm}: done" + "".join(f" ({n})" for n in notes)


@app.function(image=image, volumes={str(VOLUME_ROOT): runs_volume}, timeout=DAY_SECONDS)
def orchestrate(run_name: str, slice_spec: str, arms: list[str], workers: int) -> str:
    root = VOLUME_ROOT / run_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "STATUS.txt").write_text(f"running: slice {slice_spec}, arms {arms}\n")
    runs_volume.commit()
    results = run_arm_job.starmap([(arm, run_name, slice_spec, workers) for arm in arms],
                                  return_exceptions=True)
    lines = [str(r) if isinstance(r, str) else f"error: {r!r}" for r in results]
    runs_volume.reload()
    command = analyze_command(root, arms)
    if command is not None:
        code = run_and_commit(command, root / "logs" / "analyze.log")
        lines.append(f"analyze: {'done' if code == 0 else f'FAILED (exit {code})'}")
    (root / "STATUS.txt").write_text("finished\n" + "\n".join(lines) + "\n")
    runs_volume.commit()
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("launch", help="Spawn the job on the deployed app and return")
    launch.add_argument("--run-name", required=True)
    launch.add_argument("--slice", required=True, help="e.g. 0:100")
    launch.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    launch.add_argument("-w", "--workers", type=int, default=6, help="Parallel tasks per arm")
    args = parser.parse_args(argv)

    job = modal.Function.from_name(APP_NAME, "orchestrate")
    call = job.spawn(args.run_name, args.slice, args.arms, args.workers)
    print(f"spawned {call.object_id}: {args.run_name}, slice {args.slice}, arms {args.arms}, "
          f"{args.workers} workers per arm")


if __name__ == "__main__":
    main()
