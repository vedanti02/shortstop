"""The Modal sandbox environment runs every command in bash with the image's own environment."""

import shlex
import subprocess

import pytest
from shortstop_plugin import environment
from shortstop_plugin.environment import BashModalEnvironment, wrap_in_bash


def test_wrapped_command_runs_in_bash_with_the_extra_variables():
    wrapped = wrap_in_bash("echo \"$GREETING, it's $0\"; echo $KEEP", {"GREETING": "hi there"})
    out = subprocess.run(wrapped, shell=True, capture_output=True, text=True, check=True,
                         env={"PATH": "/usr/bin:/bin", "KEEP": "inherited"})
    assert out.stdout == "hi there, it's bash\ninherited\n"


def test_wrapping_without_variables_still_uses_bash():
    assert wrap_in_bash("ls", {}) == "bash -c ls"
    assert wrap_in_bash("a; b", {}) == "bash -c " + shlex.quote("a; b")


class FakeRuntime:
    def __init__(self, stdout="ok"):
        self.commands = []
        self.stdout = stdout

    async def execute(self, command):
        self.commands.append(command)
        return type("Response", (), {"stdout": self.stdout, "exit_code": 0})()


class FakeSandbox:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class FakeDeployment:
    fail_start = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.runtime = FakeRuntime()
        self._sandbox = None

    async def start(self):
        self._sandbox = FakeSandbox()
        if self.fail_start:
            raise RuntimeError("runtime never came up")

    async def stop(self):
        # Like swe-rex 1.4.0: forgets a sandbox that is still running instead of terminating it.
        self._sandbox = None


@pytest.fixture
def env(monkeypatch):
    from minisweagent.environments.extra import swerex_modal

    monkeypatch.setattr(swerex_modal, "ModalDeployment", FakeDeployment)
    return BashModalEnvironment(image="img", cwd="/testbed", env={"BASH_ENV": "/root/.bashrc"})


def test_execute_inherits_the_sandbox_environment(env):
    out = env.execute({"command": "python -V"})
    sent = env.deployment.runtime.commands[0]
    assert sent.env is None  # an explicit env would replace PATH and the rest of the image's environment
    assert sent.command == wrap_in_bash("python -V", {"BASH_ENV": "/root/.bashrc"})
    assert sent.cwd == "/testbed"
    assert out == {"output": "ok", "returncode": 0, "exception_info": ""}


def test_extra_variables_are_still_recorded_in_the_trajectory(env):
    assert env.serialize()["info"]["config"]["environment"]["env"] == {"BASH_ENV": "/root/.bashrc"}
    assert env.get_template_vars()["env"] == {"BASH_ENV": "/root/.bashrc"}


def test_registered_under_the_swerex_modal_name(monkeypatch):
    from minisweagent import environments

    monkeypatch.setitem(environments._ENVIRONMENT_MAPPING, "swerex_modal", "unchanged")
    environment.register()
    assert environments.get_environment_class("swerex_modal") is BashModalEnvironment


def test_a_runtime_failure_becomes_an_observation(env):
    async def boom(command):
        raise TimeoutError("sandbox gone")

    env.deployment.runtime.execute = boom
    out = env.execute({"command": "ls"})
    assert out["returncode"] == -1
    assert "sandbox gone" in out["exception_info"]
    assert out["extra"]["exception_type"] == "TimeoutError"


def test_stop_terminates_a_sandbox_that_is_still_running(env):
    sandbox = env.deployment._sandbox
    env.stop()
    assert sandbox.terminated
    env.stop()  # a second stop is harmless


def test_a_failed_start_terminates_the_sandbox(monkeypatch):
    from minisweagent.environments.extra import swerex_modal

    created = []

    class FailingDeployment(FakeDeployment):
        fail_start = True

        async def start(self):
            created.append(self)
            await super().start()

    monkeypatch.setattr(swerex_modal, "ModalDeployment", FailingDeployment)
    with pytest.raises(RuntimeError, match="never came up"):
        BashModalEnvironment(image="img")
    assert created[0]._sandbox.terminated


def test_stop_survives_a_runtime_and_sandbox_that_will_not_close(env, caplog):
    class StuckSandbox:
        object_id = "sb-stuck"

        def terminate(self):
            raise ConnectionError("modal unreachable")

    async def fail():
        raise TimeoutError("runtime hung")

    env.deployment._sandbox = StuckSandbox()
    env.deployment.stop = fail
    env.stop()
    assert "runtime hung" in caplog.text and "sb-stuck" in caplog.text
