"""Richer step features for the predictor, one row per step prefix.

    python predictor/step_features.py                 # SWE-bench Verified runs from coverage/

Same layout as coverage/features.py: a run of T steps gives rows k = 0..T-1, row k
describes what is known after k steps and carries the cost of step k + 1. On top of
Vedanti's counts it adds two groups:

    process   how the run is going: last test result, steps since the last edit or test,
              reproduction scripts, repeated commands, format errors, timeouts
    gold      what the benchmark owner can check against the reference solution: has the
              agent looked at or edited the files the gold patch touches, has it run the
              tests that must pass. Kept separate so results can be shown with and without.

Task-level columns (gold patch size, number of FAIL_TO_PASS tests, repository) are the same
on every row of a run. The leave-one-out task solve rate depends on the training agents, so
it is added at training time, not here.

Output: data/swebench_verified/features_rich.pkl
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# Vedanti's definitions, kept identical so her counts can be reproduced.
TEST_RUN = re.compile(r"pytest|runtests|manage\.py test|unittest|\btox\b|bin/test\b")
EDIT = re.compile(r"sed -i|cat\s*<<|cat\s*>|\btee\b|git apply|\bpatch\b|str_replace|write_text|\.write\(")
ERROR = re.compile(r"Traceback \(most recent call last\)|\bError\b|\bFAILED\b|\bfailed\b")
RETURNCODE = re.compile(r"<returncode>(-?\d+)</returncode>")
PASSED = re.compile(r"\b\d+ passed\b|\bOK\b")

BASH_BLOCK = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
BASH_CODE = re.compile(r"<bash_code>(.*?)</bash_code>", re.DOTALL)
FORMAT_ERROR = re.compile(r"EXACTLY ONE action|found \d+ actions|format your action", re.IGNORECASE)
TIMEOUT = re.compile(r"timed out|TimeoutExpired|timeout", re.IGNORECASE)
REPRO_NAME = r"(?:reproduc\w*|repro\w*|test_issue\w*|debug\w*|verify\w*|check_\w*|test_fix\w*|final_test\w*)\.py"
REPRO_CREATE = re.compile(r"(?:cat\s*>|cat\s*<<|tee|open\().{0,80}?" + REPRO_NAME, re.DOTALL)
REPRO_RUN = re.compile(r"python3?\s+(?:\S*/)?" + REPRO_NAME)
N_PASSED = re.compile(r"(\d+) passed")
N_FAILED = re.compile(r"(\d+) failed")
N_ERRORS = re.compile(r"(\d+) errors?\b")
DJANGO_OK = re.compile(r"Ran \d+ tests? in [\d.]+s\s+OK")
DJANGO_FAIL = re.compile(r"FAILED \((?:failures|errors)=(\d+)")
DIFF_FILE = re.compile(r"diff --git a/(\S+) b/")
PY_PATH = re.compile(r"[\w./-]+\.py\b")

STATIC = ["gold_lines", "gold_files_n", "f2p_n", "repo_code"]


def command(action: str) -> str:
    """The shell command in a step: the last fenced or tagged block, else the whole action."""
    blocks = BASH_CODE.findall(action) or BASH_BLOCK.findall(action)
    return blocks[-1] if blocks else action


def norm_path(path: str) -> str:
    return path.removeprefix("./").removeprefix("/testbed/").lstrip("/")


def test_status(cmd: str, obs: str, code: int | None) -> tuple[int, int, int]:
    """(status, n_passed, n_failed) for a step that ran tests. status: 1 pass, -1 fail, 0 unclear."""
    passed = sum(int(x) for x in N_PASSED.findall(obs))
    failed = sum(int(x) for x in N_FAILED.findall(obs)) + sum(int(x) for x in N_ERRORS.findall(obs))
    failed += sum(int(x) for x in DJANGO_FAIL.findall(obs))
    if DJANGO_OK.search(obs):
        passed = max(passed, 1)
    if failed > 0 or (code not in (None, 0) and passed == 0):
        return -1, passed, failed
    if passed > 0 and code in (None, 0):
        return 1, passed, failed
    return 0, passed, failed


def gold_info(meta: pd.DataFrame) -> dict:
    """Per task: files touched by the gold patch, test files, FAIL_TO_PASS test names."""
    info = {}
    repos = {r: i for i, r in enumerate(sorted(meta["repo"].unique()))}
    for row in meta.itertuples():
        patch = row.patch if isinstance(row.patch, str) else ""
        test_patch = row.test_patch if isinstance(row.test_patch, str) else ""
        gold = sorted(set(DIFF_FILE.findall(patch)))
        tests = sorted(set(DIFF_FILE.findall(test_patch)))
        raw = row.FAIL_TO_PASS
        if isinstance(raw, str):
            f2p = json.loads(raw)
        elif hasattr(raw, "__len__"):
            f2p = list(raw)
        else:  # missing in some SWE-rebench rows
            f2p = []
        names = set()
        for t in f2p:  # "path::Class::test_name[param]" or "test_name (module.Class)"
            name = re.split(r"::| \(", t)[-1 if "::" in t else 0]
            name = re.sub(r"\[.*$", "", name).strip()
            if len(name) > 4:
                names.add(name)
        lines = sum(1 for ln in patch.splitlines()
                    if ln[:1] in "+-" and not ln.startswith(("+++", "---")))
        info[row.instance_id] = {"gold": gold, "tests": tests, "f2p": sorted(names),
                                 "gold_lines": lines, "gold_files_n": len(gold),
                                 "f2p_n": len(f2p), "repo_code": repos[row.repo]}
    return info


class RunFeatures:
    """Features of one run, updated one step at a time (constant work per step).

    row() describes what is known after the steps seen so far; update(step) adds the next step.
    run_rows() is built on it, so offline features and a live harness's features are identical.
    """

    def __init__(self, g: dict | None):
        self.gold = g["gold"] if g else []
        self.tests = g["tests"] if g else []
        self.f2p = g["f2p"] if g else []
        self.static = {k: (g[k] if g else np.nan) for k in STATIC}
        self.c = dict.fromkeys(["test_runs", "edits", "errors", "nonzero_exits", "tests_passed",
                                "format_errors", "timeouts", "repro_created", "repro_runs", "repeats",
                                "git_diffs", "f2p_file_runs", "f2p_name_mentions"], 0)
        self.last = {"last_nonzero_exit": 0, "last_test_status": 0, "n_passed_last": 0,
                     "n_failed_last": 0, "consecutive_repeat": 0, "f2p_last_status": 0,
                     "obs_chars_last": 0, "context_tokens": 0}
        self.since_test, self.since_edit, self.edits_since_test = -1, -1, 0
        self.tokens, self.out_tokens, self.k = 0, 0, 0
        self.seen_cmds: set[str] = set()
        self.prev_cmd = None
        self.viewed, self.found, self.edited, self.edited_nongold = set(), set(), set(), set()
        self.diff_files: set[str] = set()

    def row(self) -> dict:
        gold, k = self.gold, self.k
        n_gold = max(len(gold), 1)
        return {
            "k": k, "tokens_so_far": self.tokens, "mean_out_tokens": self.out_tokens / k if k else 0.0,
            **self.c, **self.last,
            "steps_since_test": self.since_test, "steps_since_edit": self.since_edit,
            "edits_since_test": self.edits_since_test,
            "edited_files_n": len(self.edited) + len(self.edited_nongold),
            "gold_viewed_frac": len(self.viewed) / n_gold, "gold_found_frac": len(self.found) / n_gold,
            "gold_edited_frac": len(self.edited) / n_gold, "nongold_edited_n": len(self.edited_nongold),
            "diff_gold_frac": len(self.diff_files & set(gold)) / n_gold,
            "diff_nongold_n": len(self.diff_files - set(gold) - set(self.tests)),
            **self.static,
        }

    def update(self, step: dict) -> None:
        c, last, gold = self.c, self.last, self.gold
        action, obs = step["action"], step["observation"]
        cmd = command(action)
        m = RETURNCODE.search(obs)
        code = int(m.group(1)) if m else None
        is_test = bool(TEST_RUN.search(cmd))
        is_edit = bool(EDIT.search(cmd))

        # Vedanti's counts (on the full action text, as in coverage/features.py)
        last["last_nonzero_exit"] = int(code is not None and code != 0)
        c["test_runs"] += bool(TEST_RUN.search(action))
        c["edits"] += bool(EDIT.search(action))
        c["errors"] += bool(ERROR.search(obs))
        c["nonzero_exits"] += last["last_nonzero_exit"]
        c["tests_passed"] += (bool(TEST_RUN.search(action)) and bool(PASSED.search(obs))
                              and not last["last_nonzero_exit"])

        # process
        c["format_errors"] += bool(FORMAT_ERROR.search(obs))
        c["timeouts"] += bool(TIMEOUT.search(obs[:2000])) and code not in (0,)
        c["repro_created"] += bool(REPRO_CREATE.search(cmd))
        c["repro_runs"] += bool(REPRO_RUN.search(cmd))
        key = " ".join(cmd.split())
        c["repeats"] += bool(key) and key in self.seen_cmds
        last["consecutive_repeat"] = int(bool(key) and key == self.prev_cmd)
        self.seen_cmds.add(key)
        self.prev_cmd = key
        last["obs_chars_last"] = len(obs)
        last["context_tokens"] = step["tokens_in"]
        if "git diff" in cmd:
            c["git_diffs"] += 1
            files = {norm_path(f) for f in DIFF_FILE.findall(obs)}
            if files:
                self.diff_files = files
        if is_test:
            status, n_pass, n_fail = test_status(cmd, obs, code)
            last.update(last_test_status=status, n_passed_last=n_pass, n_failed_last=n_fail)
            self.since_test, self.edits_since_test = 0, 0
        elif self.since_test >= 0:
            self.since_test += 1
        if is_edit:
            self.since_edit = 0
            self.edits_since_test += 1
        elif self.since_edit >= 0:
            self.since_edit += 1

        # gold
        paths = {norm_path(p) for p in PY_PATH.findall(cmd)}
        for f in gold:
            if f in cmd or f in paths:
                self.viewed.add(f)
                if is_edit:
                    self.edited.add(f)
            if f in obs:
                self.found.add(f)
        if is_edit:
            for p in paths:
                base = p.rsplit("/", 1)[-1]
                if (p not in gold and not re.match(REPRO_NAME, base) and "test" not in base
                        and not any(p.endswith(f) or f.endswith(p) for f in gold)):
                    self.edited_nongold.add(p)
        if any(t in cmd for t in self.tests):
            c["f2p_file_runs"] += is_test
            if is_test:
                last["f2p_last_status"] = test_status(cmd, obs, code)[0]
        c["f2p_name_mentions"] += any(n in cmd for n in self.f2p)

        self.tokens += step["tokens_in"] + step["tokens_out"]
        self.out_tokens += step["tokens_out"]
        self.k += 1


def run_rows(steps: list[dict], g: dict | None) -> list[dict]:
    """One row per step prefix: row k describes what is known after k steps, plus step k+1's cost."""
    state = RunFeatures(g)
    rows = []
    for step in steps:
        rows.append({"task_id": step["task_id"], "agent": step["agent"], **state.row(),
                     "next_cost": step["tokens_in"] + step["tokens_out"], "outcome": step["outcome"]})
        state.update(step)
    return rows


_GOLD: dict = {}


def _init(gold: dict) -> None:
    global _GOLD
    _GOLD = gold


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    num = df.columns.difference(["task_id", "agent"])
    df[num] = df[num].astype(np.float32)
    df[["task_id", "agent"]] = df[["task_id", "agent"]].astype("category")
    return df


def agent_rows(path: Path, chunk: int = 200_000) -> pd.DataFrame:
    """Feature rows for one file, built in chunks so large files fit in memory."""
    frames, rows, steps, key = [], [], [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            step = json.loads(line)
            if step["task_id"] != key and steps:
                rows += run_rows(steps, _GOLD.get(key.split("@")[0]))
                steps = []
                if len(rows) >= chunk:
                    frames.append(_frame(rows))
                    rows = []
            key = step["task_id"]
            steps.append(step)
    if steps:
        rows += run_rows(steps, _GOLD.get(key.split("@")[0]))
    if rows:
        frames.append(_frame(rows))
    if not frames:  # e.g. a CoderForge shard whose tasks were all from Verified repositories
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df[["task_id", "agent"]] = df[["task_id", "agent"]].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/swebench_verified"))
    parser.add_argument("--meta", type=Path, nargs="+",
                        default=[Path("data/meta/swebench_verified.parquet")])
    parser.add_argument("--out", type=Path, help="default: <data>/features_rich.pkl")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    gold = gold_info(pd.concat([pd.read_parquet(m) for m in args.meta], ignore_index=True)
                     .drop_duplicates("instance_id"))
    paths = sorted(args.data.glob("*.jsonl.gz"))
    with Pool(args.workers, initializer=_init, initargs=(gold,)) as pool:
        frames = []
        for path, df in zip(paths, pool.imap(agent_rows, paths)):
            if len(df):
                frames.append(df)
            print(f"{path.name}: {len(df)} rows", flush=True)
    features = pd.concat(frames, ignore_index=True)
    out = args.out or args.data / "features_rich.pkl"
    features.to_pickle(out)
    print(f"wrote {out}: {len(features)} rows, {features['agent'].nunique()} agents, "
          f"{features.shape[1]} columns")


if __name__ == "__main__":
    main()
