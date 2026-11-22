"""Score one arm's predictions on SWE-bench Verified in Modal sandboxes.

    python tool_ab/evaluate.py --preds data/tool_ab/runs/baseline/preds.json --run-id baseline -w 8
    python tool_ab/convert.py --arm baseline --report data/tool_ab/eval/baseline.json

swebench's own Modal path does not work with modal 1.x (4.1.0 calls the removed
Sandbox.open; 5.0.2 references a missing TestSpec field), so this does the same steps from
here: start the prebuilt task image, apply the patch (git apply, then patch --fuzz=5),
run swebench's eval script, and grade the log with swebench's get_eval_report. Each
sandbox is terminated when its task is done. Logs go to <out>/logs/<run-id>/<task>/.
"""

from __future__ import annotations

import argparse
import json
import shlex
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

from swebench.harness.grading import get_eval_report

REPO = Path(__file__).resolve().parent.parent
DATASET = "princeton-nlp/SWE-bench_Verified"
SANDBOX_TIMEOUT_SECONDS = 30 * 60
UNRESOLVED = {"resolved": False, "patch_successfully_applied": False}


class Sandbox(Protocol):
    def run(self, command: str, stdin: str | None = None) -> tuple[str, int]: ...

    def terminate(self) -> None: ...


class ModalSandbox:
    """A Modal sandbox started from the prebuilt SWE-bench image of one task."""

    def __init__(self, instance_id: str):
        import modal

        name = instance_id.replace("__", "_1776_").lower()
        image = modal.Image.from_registry(f"docker.io/swebench/sweb.eval.x86_64.{name}:latest")
        app = modal.App.lookup("shortstop-eval", create_if_missing=True)
        self.sandbox = modal.Sandbox.create(image=image, app=app, timeout=SANDBOX_TIMEOUT_SECONDS, cpu=2)

    def run(self, command: str, stdin: str | None = None) -> tuple[str, int]:
        process = self.sandbox.exec("bash", "-c", f"{command} 2>&1")
        if stdin is not None:
            process.stdin.write(stdin.encode())
            process.stdin.write_eof()
            process.stdin.drain()
        output = process.stdout.read()
        return output, process.wait()

    def terminate(self) -> None:
        self.sandbox.terminate()


def write_file(box: Sandbox, path: str, content: str) -> None:
    box.run(f"cat > {shlex.quote(path)}", stdin=content)


def apply_patch(box: Sandbox, patch: str) -> tuple[bool, str]:
    write_file(box, "/tmp/patch.diff", patch)
    output, code = box.run("cd /testbed && git apply -v /tmp/patch.diff")
    if code == 0:
        return True, output
    fallback, code = box.run("cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff")
    return code == 0, output + "\n" + fallback


def evaluate_one(test_spec, pred: dict, log_dir: Path,
                 make_sandbox: Callable[[str], Sandbox]) -> dict:
    """The task's grading report (resolved, tests_status, ...); unresolved if nothing to test."""
    patch = pred.get("model_patch") or ""
    if not patch.strip():
        return dict(UNRESOLVED)
    task_dir = Path(log_dir) / test_spec.instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    box = make_sandbox(test_spec.instance_id)
    try:
        applied, apply_output = apply_patch(box, patch)
        (task_dir / "apply_patch.txt").write_text(apply_output)
        if not applied:
            return dict(UNRESOLVED)
        write_file(box, "/root/eval.sh", test_spec.eval_script.replace("locale-gen", "locale-gen en_US.UTF-8"))
        test_output, _ = box.run("cd /testbed && /bin/bash /root/eval.sh")
    finally:
        box.terminate()
    log_path = task_dir / "test_output.txt"
    log_path.write_text(test_output)
    report = get_eval_report(test_spec=test_spec, prediction=pred, test_log_path=str(log_path),
                             include_tests_status=True)
    result = report[test_spec.instance_id]
    (task_dir / "report.json").write_text(json.dumps(result, indent=1))
    return result


def evaluate_all(specs: list, predictions: dict, log_dir: Path, workers: int,
                 make_sandbox: Callable[[str], Sandbox] | None) -> dict:
    make_sandbox = make_sandbox or ModalSandbox

    def score(test_spec) -> tuple[str, dict | None, str]:
        try:
            result = evaluate_one(test_spec, predictions[test_spec.instance_id], log_dir, make_sandbox)
            return test_spec.instance_id, result, ""
        except Exception:  # noqa: BLE001 -- one broken task must not stop the others
            return test_spec.instance_id, None, traceback.format_exc()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = sorted(pool.map(score, specs))
    return {
        "resolved_ids": [t for t, r, _ in outcomes if r and r.get("resolved")],
        "unresolved_ids": [t for t, r, _ in outcomes if r and not r.get("resolved")],
        "error_ids": [t for t, r, _ in outcomes if r is None],
        "errors": {t: e for t, r, e in outcomes if r is None},
    }


def load_specs(dataset: str, instance_ids: list[str]) -> list:
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import load_swebench_dataset

    return [make_test_spec(inst) for inst in load_swebench_dataset(dataset, "test", instance_ids)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "tool_ab" / "eval")
    parser.add_argument("--instance-ids", nargs="*", default=None, help="Default: every task in --preds")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("-w", "--workers", type=int, default=4)
    args = parser.parse_args(argv)

    predictions = json.loads(args.preds.read_text())
    ids = sorted(args.instance_ids or predictions)
    specs = load_specs(args.dataset, ids)
    report = evaluate_all(specs, predictions, args.out / "logs" / args.run_id, args.workers, make_sandbox=None)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.run_id}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"{args.run_id}: {len(report['resolved_ids'])}/{len(ids)} resolved, "
          f"{len(report['error_ids'])} errors -> {path}")


if __name__ == "__main__":
    main()
