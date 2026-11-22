"""Spending backstop: stop the tool A/B run, its sandboxes and the GPU server after a deadline.

    modal deploy tool_ab/remote/watchdog.py
    python tool_ab/remote/watchdog.py arm --hours 11 --run-name full-100

It runs as its own Modal app, so deploying it does not touch a run in progress. At the
deadline it stops the apps that bill (the run, the model server), terminates every live
sandbox the run started, and notes it in <run-name>/STATUS.txt on the shortstop-runs
volume. Finished tasks stay on the volume; relaunching the same run name skips them.
After it fires, redeploy serve_qwen.py and full_run.py before the next run.

This caps time, not dollars. Modal's workspace spending limit is the only hard cap.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "shortstop-watchdog"
RUN_APPS = ["shortstop-full-run", "shortstop-qwen"]
SANDBOX_APPS = ["swe-rex", "shortstop-eval"]
MAX_HOURS = 23  # Modal functions run at most 24 h

image = modal.Image.debian_slim(python_version="3.12").pip_install("modal==1.6.4")
runs_volume = modal.Volume.from_name("shortstop-runs", create_if_missing=True)
app = modal.App(APP_NAME)


def stop_everything(apps: list[str], sandbox_apps: list[str]) -> list[str]:
    notes = []
    for name in apps:
        result = subprocess.run(["modal", "app", "stop", "-y", name], capture_output=True, text=True, check=False)
        notes.append(f"stop {name}: {'ok' if result.returncode == 0 else result.stderr.strip()[:200]}")
    for name in sandbox_apps:
        try:
            sandboxes = list(modal.Sandbox.list(app_id=modal.App.lookup(name).app_id))
        except Exception as e:  # noqa: BLE001 -- keep going with the other apps
            notes.append(f"sandboxes {name}: lookup failed: {e}")
            continue
        for sandbox in sandboxes:
            sandbox.terminate()
        notes.append(f"sandboxes {name}: terminated {len(sandboxes)}")
    return notes


@app.function(image=image, volumes={"/vol": runs_volume}, timeout=24 * 60 * 60)
def stop_after(hours: float, run_name: str, apps: list[str], sandbox_apps: list[str]) -> list[str]:
    time.sleep(hours * 3600)
    notes = stop_everything(apps, sandbox_apps)
    runs_volume.reload()
    status = Path("/vol") / run_name / "STATUS.txt"
    if status.parent.exists():
        previous = status.read_text() if status.exists() else ""
        stamp = datetime.now(timezone.utc).isoformat(timespec="minutes")
        status.write_text(previous + f"watchdog fired at {stamp} after {hours} h:\n  " + "\n  ".join(notes) + "\n")
        runs_volume.commit()
    return notes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    arm = sub.add_parser("arm", help="Start the countdown")
    arm.add_argument("--hours", type=float, required=True)
    arm.add_argument("--run-name", required=True)
    arm.add_argument("--apps", nargs="+", default=RUN_APPS)
    arm.add_argument("--sandbox-apps", nargs="+", default=SANDBOX_APPS)
    args = parser.parse_args(argv)
    if not 0 < args.hours <= MAX_HOURS:
        raise SystemExit(f"--hours must be in (0, {MAX_HOURS}]")

    call = modal.Function.from_name(APP_NAME, "stop_after").spawn(
        args.hours, args.run_name, args.apps, args.sandbox_apps)
    print(f"watchdog {call.object_id}: stops {args.apps} and sandboxes in {args.sandbox_apps} "
          f"in {args.hours} h (cancel: modal app stop -y {APP_NAME})")


if __name__ == "__main__":
    main()
