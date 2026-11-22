"""Modal sandbox environment that runs every command in bash with the image's own environment.

mini-swe-agent's swerex_modal environment hands each command to subprocess with shell=True,
so it runs under /bin/sh (dash on the SWE-bench images), and it passes `env` as the whole
process environment. Dash ignores BASH_ENV and the replaced environment drops PATH, so the
conda `testbed` environment is never activated and `python` is the system interpreter.

This subclass leaves the sandbox environment alone and runs `env K=V ... bash -c <command>`
instead, so the extra variables (BASH_ENV among them) are added on top of the image's PATH.

It also terminates its sandbox on stop. swe-rex 1.4.0's ModalDeployment.stop() only calls
terminate() on a sandbox that has already exited, so a live one keeps running (and billing)
until its runtime_timeout. A sandbox whose runtime fails to start is terminated as well.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
from swerex.runtime.abstract import Command as RexCommand

CLASS_PATH = f"{__name__}.BashModalEnvironment"
STOP_TIMEOUT_SECONDS = 10

logger = logging.getLogger(__name__)


def wrap_in_bash(command: str, env: dict[str, str]) -> str:
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    prefix = f"env {assignments} " if assignments else ""
    return f"{prefix}bash -c {shlex.quote(command)}"


class BashModalEnvironment(SwerexModalEnvironment):
    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
        except Exception:
            if hasattr(self, "deployment"):  # the sandbox may exist even though its runtime failed
                self._terminate_sandbox(getattr(self.deployment, "_sandbox", None))
            raise

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "") if isinstance(action, dict) else action
        rex_command = RexCommand(
            command=wrap_in_bash(command, self.config.env),
            shell=True,
            check=False,
            cwd=cwd or self.config.cwd,
            timeout=timeout or self.config.timeout,
            merge_output_streams=True,
            env=None,
        )
        try:
            result = asyncio.run(self.deployment.runtime.execute(rex_command))
            output = {"output": result.stdout, "returncode": result.exit_code, "exception_info": ""}
        except Exception as e:  # noqa: BLE001 -- the agent sees any runtime failure, as in the parent class
            output = {
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def stop(self) -> None:
        sandbox = getattr(self.deployment, "_sandbox", None)  # deployment.stop() forgets it
        try:
            asyncio.run(asyncio.wait_for(self.deployment.stop(), timeout=STOP_TIMEOUT_SECONDS))
        except Exception as e:  # noqa: BLE001 -- the sandbox is terminated below either way
            logger.warning("Closing the swe-rex runtime failed: %s", e)
        self._terminate_sandbox(sandbox)

    @staticmethod
    def _terminate_sandbox(sandbox) -> None:
        if sandbox is None:
            return
        try:
            sandbox.terminate()
        except Exception as e:  # noqa: BLE001 -- runtime_timeout still ends it eventually
            logger.warning("Terminating Modal sandbox %s failed: %s", getattr(sandbox, "object_id", "?"), e)


def register() -> None:
    """Serve `environment_class: swerex_modal` with this class.

    The SWE-bench runner only sets the task image for the built-in names, so the config keeps
    `swerex_modal` and the name is pointed here instead.
    """
    from minisweagent import environments

    environments._ENVIRONMENT_MAPPING["swerex_modal"] = CLASS_PATH
