"""Compressed command outputs: the third tool change.

The agent sees a cleaned, shorter version of each command's output. Order of operations:
strip ANSI codes, keep only the final state of carriage-return progress bars, truncate long
lines, collapse repeated and blank lines, and finally keep the head and tail of whatever is
still over the character budget.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class CompressionConfig(BaseModel):
    max_chars: int = 4000
    """Character budget for the whole output after line-level cleanup."""
    max_line_chars: int = 300
    """Lines longer than this are cut."""
    head_fraction: float = 0.6
    """Share of the budget given to the start of the output; the rest goes to the end."""


def _clean_line(line: str, max_line_chars: int) -> str:
    line = line.rstrip("\r").rsplit("\r", 1)[-1]  # CRLF endings stay; progress redraws collapse
    if len(line) > max_line_chars:
        return f"{line[:max_line_chars]} ...[{len(line) - max_line_chars} more chars]"
    return line


def _collapse(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        repeats = j - i
        if lines[i] == "":
            out.append("")
        elif repeats:
            out += [lines[i], f"[previous line repeated {repeats} more times]"]
        else:
            out.append(lines[i])
        i = j + 1
    return out


def _head_and_tail(lines: list[str], cfg: CompressionConfig) -> list[str]:
    head_budget = int(cfg.max_chars * cfg.head_fraction)
    tail_budget = cfg.max_chars - head_budget
    head, used = [], 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        head.append(line)
        used += len(line) + 1
    tail, used = [], 0
    for line in reversed(lines[len(head):]):
        if used + len(line) + 1 > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    omitted = len(lines) - len(head) - len(tail)
    return [*head, f"[... {omitted} lines omitted ...]", *reversed(tail)]


def compress_text(text: str, cfg: CompressionConfig) -> str:
    text = _ANSI.sub("", text)
    lines = _collapse([_clean_line(line, cfg.max_line_chars) for line in text.split("\n")])
    if sum(len(line) + 1 for line in lines) - 1 > cfg.max_chars:
        lines = _head_and_tail(lines, cfg)
    return "\n".join(lines)


def compress_result(output: dict, cfg: CompressionConfig) -> dict:
    """Return a copy of an environment result with its output compressed."""
    return {**output, "output": compress_text(output.get("output", ""), cfg)}
