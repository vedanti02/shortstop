"""evaluate.py: score predictions in Modal sandboxes with swebench's eval scripts and grading."""

import json
from pathlib import Path
from types import SimpleNamespace

import evaluate
import pytest

PATCH = "diff --git a/x.py b/x.py\n"


class FakeSandbox:
    """Answers commands from a script and records what it was asked to do."""

    def __init__(self, replies=None):
        self.replies = replies or {}
        self.commands = []
        self.files = {}
        self.terminated = False

    def run(self, command, stdin=None):
        self.commands.append(command)
        if command.startswith("cat > "):
            self.files[command.removeprefix("cat > ")] = stdin
            return "", 0
        for prefix, reply in self.replies.items():
            if prefix in command:
                return reply
        return "", 0

    def terminate(self):
        self.terminated = True


def spec(task="t__t-1"):
    return SimpleNamespace(instance_id=task, eval_script="run tests\nlocale-gen\n")


@pytest.fixture
def graded(monkeypatch):
    calls = []

    def fake_report(test_spec, prediction, test_log_path, include_tests_status):
        calls.append(test_log_path)
        resolved = "PASSED" in Path(test_log_path).read_text()
        return {test_spec.instance_id: {"resolved": resolved, "patch_successfully_applied": True}}

    monkeypatch.setattr(evaluate, "get_eval_report", fake_report)
    return calls


def test_applies_the_patch_runs_the_eval_script_and_grades_the_log(tmp_path, graded):
    box = FakeSandbox({"/root/eval.sh": ("test_a PASSED\n", 0)})
    result = evaluate.evaluate_one(spec(), {"model_patch": PATCH}, tmp_path, lambda s: box)

    assert result == {"resolved": True, "patch_successfully_applied": True}
    assert box.files["/tmp/patch.diff"] == PATCH
    assert box.files["/root/eval.sh"] == "run tests\nlocale-gen en_US.UTF-8\n"  # as upstream does
    assert any("git apply" in c for c in box.commands)
    assert (tmp_path / "t__t-1" / "test_output.txt").read_text() == "test_a PASSED\n"
    assert box.terminated


def test_falls_back_to_patch_when_git_apply_fails(tmp_path, graded):
    box = FakeSandbox({"git apply": ("conflict", 1), "/root/eval.sh": ("PASSED", 0)})
    assert evaluate.evaluate_one(spec(), {"model_patch": PATCH}, tmp_path, lambda s: box)["resolved"]
    assert any(c.startswith("cd /testbed && patch --batch") for c in box.commands)


def test_a_patch_that_does_not_apply_is_unresolved(tmp_path, graded):
    box = FakeSandbox({"git apply": ("conflict", 1), "patch --batch": ("rejected", 1)})
    result = evaluate.evaluate_one(spec(), {"model_patch": PATCH}, tmp_path, lambda s: box)
    assert result == {"resolved": False, "patch_successfully_applied": False}
    assert "rejected" in (tmp_path / "t__t-1" / "apply_patch.txt").read_text()
    assert not graded and box.terminated


def test_an_empty_patch_is_unresolved_without_a_sandbox(tmp_path, graded):
    def no_sandbox(_):
        raise AssertionError("no sandbox needed")

    result = evaluate.evaluate_one(spec(), {"model_patch": ""}, tmp_path, no_sandbox)
    assert result == {"resolved": False, "patch_successfully_applied": False}


def test_the_sandbox_is_terminated_when_a_command_fails(tmp_path, graded):
    class Broken(FakeSandbox):
        def run(self, command, stdin=None):
            raise ConnectionError("sandbox lost")

    box = Broken()
    with pytest.raises(ConnectionError):
        evaluate.evaluate_one(spec(), {"model_patch": PATCH}, tmp_path, lambda s: box)
    assert box.terminated


def test_evaluate_all_reports_resolved_unresolved_and_errored_tasks(tmp_path, monkeypatch):
    outcomes = {"a": {"resolved": True}, "b": {"resolved": False}}

    def fake_one(test_spec, pred, log_dir, make_sandbox):
        if test_spec.instance_id == "c":
            raise RuntimeError("image build failed")
        return outcomes[test_spec.instance_id]

    monkeypatch.setattr(evaluate, "evaluate_one", fake_one)
    preds = {t: {"instance_id": t, "model_patch": PATCH} for t in "abc"}
    report = evaluate.evaluate_all([spec(t) for t in "abc"], preds, tmp_path, workers=2,
                                   make_sandbox=None)
    assert report["resolved_ids"] == ["a"]
    assert report["unresolved_ids"] == ["b"]
    assert report["error_ids"] == ["c"]
    assert "image build failed" in report["errors"]["c"]


def test_main_writes_a_report_convert_can_read(tmp_path, monkeypatch):
    import convert

    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps({"a": {"instance_id": "a", "model_patch": PATCH},
                                 "b": {"instance_id": "b", "model_patch": PATCH}}))
    seen = {}

    def fake_specs(dataset, ids):
        seen["ids"] = ids
        return [spec(t) for t in ids]

    def fake_all(specs, predictions, log_dir, workers, make_sandbox):
        seen["workers"] = workers
        return {"resolved_ids": ["a"], "unresolved_ids": [], "error_ids": [], "errors": {}}

    monkeypatch.setattr(evaluate, "load_specs", fake_specs)
    monkeypatch.setattr(evaluate, "evaluate_all", fake_all)
    evaluate.main(["--preds", str(preds), "--run-id", "smoke", "--out", str(tmp_path / "eval"),
                   "--instance-ids", "a", "-w", "3"])

    assert seen == {"ids": ["a"], "workers": 3}
    assert convert.load_resolved(tmp_path / "eval" / "smoke.json") == {"a"}


def test_main_writes_the_report_and_fails_when_tasks_could_not_be_graded(tmp_path, monkeypatch):
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps({"a": {"instance_id": "a", "model_patch": PATCH}}))
    monkeypatch.setattr(evaluate, "load_specs", lambda dataset, ids: [spec(t) for t in ids])
    monkeypatch.setattr(evaluate, "evaluate_all", lambda *a, **k: {
        "resolved_ids": [], "unresolved_ids": [], "error_ids": ["a"], "errors": {"a": "boom"}})
    with pytest.raises(SystemExit, match="could not be graded"):
        evaluate.main(["--preds", str(preds), "--run-id", "r", "--out", str(tmp_path / "eval")])
    assert json.loads((tmp_path / "eval" / "r.json").read_text())["error_ids"] == ["a"]
