"""Download mini-swe-agent runs on SWE-bench Verified and convert them to the shared format.

Trajectories come from the public SWE-bench submissions bucket and pass/fail results from
the swe-bench/experiments repo. Each raw trajectory is converted as it is downloaded and
then dropped, so the full set (about 13 GB raw) is never on disk.

    python coverage/download_swebench.py                 # every agent with results
    python coverage/download_swebench.py --agents gpt-5  # only agents whose name matches

Output: data/swebench_verified/<agent>.jsonl.gz, one line per step, plus manifest.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://swe-bench-submissions.s3.amazonaws.com/"
RESULTS = (
    "https://raw.githubusercontent.com/swe-bench/experiments/main/"
    "evaluation/verified/{agent}/per_instance_details.json"
)
PREFIX = "bash-only/"


def fetch(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:  # timeouts, dropped connections
            last = e
        time.sleep(2**attempt)
    raise last


def list_bucket(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    """Return (keys, sub-prefixes) under a bucket prefix."""
    keys, prefixes, token = [], [], None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        xml = fetch(BUCKET + "?" + urllib.parse.urlencode(query)).decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", xml)
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", xml)
        more = re.search(r"<NextContinuationToken>([^<]*)<", xml)
        if not more:
            return keys, prefixes
        token = more.group(1)


# --- converting one raw trajectory into steps ---------------------------------------------


def _text(content) -> str:
    """Flatten a message's content, which is a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(p.get("text") if isinstance(p, dict) else p) for p in content)
    return str(content)


def _is_model_call(msg: dict) -> bool:
    return msg.get("role") == "assistant" or msg.get("object") == "response"


def _action(msg: dict) -> str:
    parts = []
    if msg.get("object") == "response":  # OpenAI Responses API item
        for item in msg.get("output") or []:
            if item.get("type") == "message":
                parts.append(_text(item.get("content")))
    else:
        parts.append(_text(msg.get("content")))
    extra = msg.get("extra") or {}
    if extra.get("actions"):
        parts += [a.get("command", "") for a in extra["actions"] if isinstance(a, dict)]
    else:
        for call in msg.get("tool_calls") or []:
            parts.append((call.get("function") or {}).get("arguments", ""))
    return "\n".join(p for p in parts if p)


def _usage(msg: dict) -> tuple[int, int] | None:
    """(tokens_in, tokens_out) as logged by the API, or None if the run did not log usage."""
    usage = msg.get("usage") or ((msg.get("extra") or {}).get("response") or {}).get("usage")
    if not usage:
        return None
    tokens_in = usage.get("prompt_tokens", usage.get("input_tokens"))
    tokens_out = usage.get("completion_tokens", usage.get("output_tokens"))
    if tokens_in is None or tokens_out is None:
        return None
    # Some Anthropic logs leave cache writes out of prompt_tokens; add them back when the
    # cache counts cannot already be contained in it.
    cache_write = usage.get("cache_creation_input_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    if tokens_in < cache_write + cache_read:
        tokens_in += cache_write
    return int(tokens_in), int(tokens_out)


def to_steps(traj: dict) -> tuple[list[dict], bool]:
    """Convert a raw mini-swe-agent trajectory into steps.

    One step is one model call plus everything the environment said back before the next
    call. Returns (steps, estimated); estimated is True when the run logged no token usage
    and counts are approximated as characters / 4.
    """
    steps: list[dict] = []
    estimated = False
    chars_so_far = 0  # size of the conversation the model has seen
    for msg in traj.get("messages") or []:
        if _is_model_call(msg):
            action = _action(msg)
            usage = _usage(msg)
            if usage is None:
                estimated = True
                usage = (chars_so_far // 4, len(action) // 4)
            steps.append(
                {
                    "tokens_in": usage[0],
                    "tokens_out": usage[1],
                    "action": action,
                    "observation": "",
                }
            )
            chars_so_far += len(action)
        elif msg.get("role") == "exit":
            continue  # the submitted patch, not something the agent observed
        else:
            text = _text(msg.get("content", msg.get("output")))
            chars_so_far += len(text)
            if steps:
                sep = "\n" if steps[-1]["observation"] else ""
                steps[-1]["observation"] += sep + text
    return steps, estimated


# --- downloading --------------------------------------------------------------------------

EMPTY_STEP = {"tokens_in": 0, "tokens_out": 0, "action": "", "observation": ""}


def convert_agent(agent: str, out_dir: Path, workers: int) -> dict | None:
    out_path = out_dir / f"{agent}.jsonl.gz"
    try:
        results = json.loads(fetch(RESULTS.format(agent=urllib.parse.quote(agent))))
    except urllib.error.HTTPError:
        print(f"skip {agent}: no per-instance results in swe-bench/experiments")
        return None

    if not any(r.get("resolved") for r in results.values()):
        print(f"skip {agent}: upstream results mark every task unresolved", flush=True)
        return None

    keys, _ = list_bucket(f"{PREFIX}{agent}/trajs/")
    traj_keys = {k.split("/")[-2]: k for k in keys if k.endswith(".traj.json")}
    if not traj_keys:
        print(f"skip {agent}: no trajectories in the bucket")
        return None

    def one(task_id: str) -> tuple[str, list[dict], bool, bool]:
        key = traj_keys.get(task_id)
        steps, estimated = [], False
        if key:
            steps, estimated = to_steps(json.loads(fetch(BUCKET + urllib.parse.quote(key))))
        # Runs with no trajectory or no model call still count toward the score, so they
        # are kept as a single empty step rather than dropped.
        return task_id, steps or [dict(EMPTY_STEP)], estimated, not steps

    stats = {"agent": agent, "tasks": 0, "resolved": 0, "steps": 0, "empty_runs": 0,
             "tokens_estimated": False}
    tmp_path = out_path.with_suffix(".tmp")
    with ThreadPoolExecutor(workers) as pool, gzip.open(tmp_path, "wt") as f:
        for task_id, steps, estimated, empty in pool.map(one, sorted(results)):
            outcome = int(bool(results[task_id].get("resolved")))
            for i, step in enumerate(steps, start=1):
                row = {"task_id": task_id, "agent": agent, "step": i, **step, "outcome": outcome}
                f.write(json.dumps(row) + "\n")
            stats["tasks"] += 1
            stats["resolved"] += outcome
            stats["steps"] += len(steps)
            stats["empty_runs"] += empty
            stats["tokens_estimated"] |= estimated
    tmp_path.rename(out_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--agents", default="", help="only agents whose name contains this")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    _, prefixes = list_bucket(PREFIX, delimiter="/")
    agents = [p.split("/")[1] for p in prefixes if args.agents in p]
    for agent in agents:
        if agent in manifest and (args.out / f"{agent}.jsonl.gz").exists():
            print(f"have {agent}")
            continue
        stats = convert_agent(agent, args.out, args.workers)
        if stats:
            manifest[agent] = stats
            manifest_path.write_text(json.dumps(manifest, indent=1))
            rate = 100 * stats["resolved"] / max(stats["tasks"], 1)
            print(f"done {agent}: {stats['tasks']} tasks, {rate:.1f}% resolved, "
                  f"{stats['steps'] / max(stats['tasks'], 1):.1f} steps/run", flush=True)


if __name__ == "__main__":
    main()
