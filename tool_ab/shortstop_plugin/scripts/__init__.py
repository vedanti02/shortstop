"""Helper programs that a tool variant installs into the agent's sandbox."""

from __future__ import annotations

import shlex
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_EOF = "SHORTSTOP_SCRIPT_EOF"


def script_path(name: str) -> Path:
    path = _DIR / name
    if not path.is_file() or path.suffix == ".py":
        raise ValueError(f"Unknown sandbox script: {name!r}")
    return path


def install_command(name: str, install_dir: str = "/usr/local/bin") -> str:
    """A single shell command that writes the script into install_dir and makes it executable."""
    body = script_path(name).read_text()
    if _EOF in body:
        raise ValueError(f"Script {name!r} contains the heredoc terminator")
    target = shlex.quote(f"{install_dir.rstrip('/')}/{name}")
    return (
        f"mkdir -p {shlex.quote(install_dir)} && cat > {target} <<'{_EOF}'\n"
        f"{body}{'' if body.endswith(chr(10)) else chr(10)}{_EOF}\n"
        f"chmod +x {target}"
    )
