"""Convert the SWE-rebench leaderboard trajectories into the shared step format.

    python predictor/convert_rebench_lb.py              # Python subset: 36 configs, 110 tasks

nebius/SWE-rebench-leaderboard-trajectories has 36 agent configurations (25 models; a common
scaffold plus Claude Code, Codex, Cursor and Junie) on the same 110 Python tasks, 5 runs per
task (2 for the Claude Code effort panel). It is a second benchmark with many agents on the
same tasks, repeated runs, and reasoning-effort pairs.

One step is one action: a tool call (with the assistant text before it) and its result, a
Codex file change, a Junie agent step, or the final message. Every action is rewritten as the
shell command with the same effect, so predictor/step_features.py works unchanged:

    bash / bash_terminal / shell {command}   -> the command
    Read / read {file_path | path}           -> cat <path>
    Edit / edit / MultiEdit, file_change     -> str_replace <path>
    Write {file_path, content}               -> cat > <path> <<'EOF' ... EOF
    Grep / grep {pattern, path}              -> grep -rn "<pattern>" <path>
    Glob / glob {pattern}                    -> find . -name "<pattern>"
    Junie "Opened file" / "Edited files"     -> cat / str_replace on the paths in its output
    final message                            -> echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Results get <returncode> from exit_code when logged, else 0 for status "success" and 1 for
"error". Token usage is logged per run only, so each step's tokens are the characters/4
estimate (context so far, and the action) scaled so the run's steps add up to its logged
total. task_id is "<instance_id>@<run_index>" and agent is the configuration id.

Output: data/rebench_lb/<file>.jsonl.gz and data/meta/rebench_lb_gold.parquet
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SOURCE = "datasets/nebius/SWE-rebench-leaderboard-trajectories/data/{subset}/*.parquet"
GOLD = "datasets/nebius/SWE-rebench-leaderboard"
KEEP = 3000
PATH = re.compile(r"[\w./-]+\.\w{1,6}\b")


def clip(text: str) -> str:
    return text if len(text) <= 2 * KEEP else text[:KEEP] + "\n[...]\n" + text[-KEEP:]


def as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def shell_command(tool: str, args: dict) -> str:
    t = (tool or "").lower()
    path = args.get("file_path") or args.get("path") or ""
    if t in {"bash", "bash_terminal", "shell", "execute_bash", "run_terminal_cmd", "exec_command"}:
        cmd = args.get("command") or args.get("cmd") or ""
        return " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    if t in {"read", "view", "open", "read_file"}:
        return f"cat {path}"
    if t in {"write", "create", "write_file"}:
        body = as_text(args.get("content") or args.get("file_text") or "")[:KEEP]
        return f"cat > {path} <<'EOF'\n{body}\nEOF"
    if t in {"edit", "multiedit", "str_replace", "apply_patch", "edit_file", "notebookedit"}:
        return f"str_replace {path}"
    if t in {"grep", "search"}:
        return f'grep -rn "{args.get("pattern", "")}" {path}'
    if t == "glob":
        return f'find {path or "."} -name "{args.get("pattern", "")}"'
    if t == "delete":
        return f"rm {path}"
    return ""


def junie_command(name: str, content: str) -> str:
    paths = PATH.findall(content[:500])
    if name.startswith("Opened file"):
        return f"cat {paths[0]}" if paths else "cat"
    if name.startswith("Edited file"):
        return "str_replace " + " ".join(dict.fromkeys(paths))
    if name.startswith(("Found ", "Searched ")):
        return f"grep {name}"
    if name.startswith("Fetch URL"):
        return ""
    return name  # Junie names a shell step by its command line


def to_steps(traj: dict) -> list[dict]:
    steps: list[dict] = []
    by_call: dict[str, int] = {}
    pending, chars = "", 0

    def new_step(cmd: str, text: str = "") -> None:
        nonlocal pending, chars
        action = (pending + text) + (f"\n```bash\n{cmd}\n```" if cmd else "")
        steps.append({"action": action, "observation": "", "_in": chars, "_out": len(action)})
        chars += len(action)
        pending = ""

    for e in traj.get("events", []):
        kind = e.get("type")
        if kind == "prompt":
            chars += sum(len(as_text(m.get("content", ""))) for m in e.get("messages") or [])
        elif kind in {"assistant_message", "assistant_analysis"}:
            pending += as_text(e.get("content", "")) + "\n"
        elif kind == "tool_call":
            new_step(shell_command(e.get("tool"), e.get("input") or {}))
            by_call[e.get("tool_call_id")] = len(steps) - 1
        elif kind == "tool_result" and steps:
            code = e.get("exit_code")
            if code is None:
                code = 0 if e.get("status") == "success" else 1
            text = as_text(e.get("content", ""))
            chars += len(text)
            i = by_call.get(e.get("tool_call_id"), len(steps) - 1)
            sep = "\n" if steps[i]["observation"] else ""
            steps[i]["observation"] += f"{sep}<returncode>{code}</returncode>\n{clip(text)}"
        elif kind == "file_change":
            paths = " ".join(c.get("path", "") for c in e.get("changes") or [])
            new_step(f"str_replace {paths}")
        elif kind == "agent_step":
            content = as_text(e.get("content", ""))
            new_step(junie_command(e.get("name") or "", content), text=(e.get("name") or "") + "\n")
            steps[-1]["observation"] = clip(content)
            chars += len(content)
        elif kind == "final_message":
            new_step("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", text=as_text(e.get("content", "")))
    return steps


def with_tokens(steps: list[dict], usage: dict) -> list[dict]:
    est = [s["_in"] // 4 + s["_out"] // 4 for s in steps]
    total = ((usage or {}).get("tokens") or {}).get("total")
    scale = total / sum(est) if total and sum(est) > 0 else 1.0
    out = []
    for s in steps:
        out.append({"tokens_in": int(s["_in"] // 4 * scale), "tokens_out": int(s["_out"] // 4 * scale),
                    "action": s["action"], "observation": s["observation"]})
    return out


def convert_file(job: tuple[str, Path]) -> dict:
    path, out_dir = job
    out = out_dir / (path.split("/")[-1].removesuffix(".parquet") + ".jsonl.gz")
    if out.exists():
        return {"existing": 1}
    fs = HfFileSystem()
    with fs.open(path) as fh:
        table = pq.read_table(fh).to_pylist()
    stats = {"runs": 0, "steps": 0, "resolved": 0, "empty": 0}
    tmp = out.with_name(out.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in table:
            traj = json.loads(r["trajectory"])
            steps = with_tokens(to_steps(traj), traj.get("usage"))
            if not steps:
                stats["empty"] += 1
                steps = [{"tokens_in": 0, "tokens_out": 0, "action": "", "observation": ""}]
            agent = r["agent_config_id"].replace("/", "__")
            for i, step in enumerate(steps, start=1):
                f.write(json.dumps({"task_id": f"{r['instance_id']}@{r['run_index']}", "agent": agent,
                                    "step": i, **step, "outcome": int(r["resolved"])}) + "\n")
            stats["runs"] += 1
            stats["steps"] += len(steps)
            stats["resolved"] += int(r["resolved"])
    tmp.rename(out)
    return stats


def fetch_gold(fs: HfFileSystem, out: Path) -> None:
    frames = []
    for p in fs.glob(f"{GOLD}/**/*.parquet"):
        with fs.open(p) as fh:
            t = pq.read_table(fh, columns=["instance_id", "repo", "patch", "test_patch", "FAIL_TO_PASS"]).to_pylist()
        frames.append(pd.DataFrame(t))
    gold = pd.concat(frames, ignore_index=True)
    gold["FAIL_TO_PASS"] = gold["FAIL_TO_PASS"].map(
        lambda v: v if isinstance(v, str) else json.dumps(list(v) if v is not None else []))
    gold.drop_duplicates("instance_id").to_parquet(out)
    print(f"wrote {out}: {gold['instance_id'].nunique()} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subset", default="python_spring2026")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    fs = HfFileSystem()
    gold = Path("data/meta/rebench_lb_gold.parquet")
    if not gold.exists():
        fetch_gold(fs, gold)
    out_dir = Path("data/rebench_lb")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(fs.glob(SOURCE.format(subset=args.subset)))
    total: dict = {}
    with Pool(args.workers) as pool:
        for stats in pool.imap_unordered(convert_file, [(p, out_dir) for p in files]):
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v
    print(f"done {args.subset}: {len(files)} files, {total}")


if __name__ == "__main__":
    main()
