"""Does a sharper task difficulty, from the SWE-bench Verified leaderboard, help the predictor?

    python predictor/leaderboard_rate_check.py fetch     # results of every Verified submission
    python predictor/leaderboard_rate_check.py train     # the held-out predictor with lb_rate added
    python predictor/evaluate.py data/predictor/predictions_gold_loao_lbrate.pkl --name lgbm_gold_loao_lbrate
    python predictor/compare.py lgbm_gold_loao_lbrate --ref lgbm_gold_loao

The predictor's strongest input is task_rate, how often the other agents solved the task, but it
comes from only 18 to 36 agents. The SWE-bench experiments repository
(github.com/SWE-bench/experiments) keeps the per-task results of every Verified submission.

fetch   Downloads evaluation/verified/<submission>/results/results.json for every submission to
        data/swebench_leaderboard/, and writes data/meta/verified_leaderboard.parquet: one row per
        task with lb_rate, the share of the other submissions that resolved it. Every
        mini-swe-agent submission is left out, because our 37 agents are among them and the
        held-out agent's own result must never enter a feature. So is any other submission whose
        results match one of the 37 agents on 99.5% of tasks or more.
train   Exactly `train_lgbm.py --features gold --split loao` (same seed, validation agents and
        cross-fitted task_rate), with lb_rate as one more feature.

Output: data/predictor/predictions_gold_loao_lbrate.pkl
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lgbm import GROUPS, predict_agent

API = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
RAW_DIR = Path("data/swebench_leaderboard")
TABLE = Path("data/meta/verified_leaderboard.parquet")
SAME_RESULTS = 0.995


def get(url: str) -> bytes:
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx, timeout=60) as response:
        return response.read()


def fetch() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = [e["name"] for e in json.loads(get(API)) if e["type"] == "dir"]

    def download(name: str) -> str | None:
        out = RAW_DIR / f"{name}.json"
        if not out.exists():
            try:
                out.write_bytes(get(f"{RAW}/{name}/results/results.json"))
            except urllib.error.HTTPError as e:
                return f"{name}: HTTP {e.code}"
        return None

    with ThreadPoolExecutor(8) as pool:
        for problem in pool.map(download, names):
            if problem:
                print("skipped", problem)

    tasks = sorted(pd.read_parquet("data/meta/swebench_verified.parquet")["instance_id"])
    ours = (pd.read_pickle("data/predictor/predictions_gold_loao.pkl").query("k == 0")
            .pivot(index="task_id", columns="agent", values="outcome").reindex(tasks))
    results = {}
    for f in sorted(RAW_DIR.glob("*.json")):
        resolved = json.loads(f.read_text()).get("resolved")
        if isinstance(resolved, list):
            results[f.stem] = np.isin(tasks, resolved).astype(float)
        else:
            print("skipped", f.stem, ": no resolved list")
    M = pd.DataFrame(results, index=tasks)
    mini = [s for s in M.columns if "mini" in s]
    print(f"{M.shape[1]} submissions with results; {len(mini)} mini-swe-agent ones left out, "
          f"{len(set(ours.columns) & set(mini))} of them our agents")
    others = [s for s in M.columns if s not in mini]
    agreement = pd.Series({s: max((M[s] == ours[a]).mean() for a in ours.columns) for s in others})
    same = agreement[agreement >= SAME_RESULTS].index.tolist()
    print(f"left out for matching one of our agents: {same or 'none'}; "
          f"highest agreement among the rest {agreement.drop(same).max():.3f}")
    keep = [s for s in others if s not in same]
    lb = M[keep].mean(axis=1)
    pd.DataFrame({"instance_id": tasks, "lb_rate": lb.to_numpy(), "lb_n": len(keep)}).to_parquet(TABLE)
    print(f"wrote {TABLE}: lb_rate from {len(keep)} submissions, mean {lb.mean():.3f}, "
          f"correlation with our agents' mean rate {np.corrcoef(lb, ours.mean(axis=1))[0, 1]:.3f}")


def train(seed: int = 0) -> None:
    features = pd.read_pickle("data/swebench_verified/features_rich.pkl")
    features["task"] = features["task_id"].str.split("@").str[0]
    outcomes = features[features["k"] == 0].pivot_table(index="task", columns="agent",
                                                        values="outcome", aggfunc="mean").dropna()
    features = features[features["task"].isin(outcomes.index)].copy()
    features["lb_rate"] = features["task"].map(pd.read_parquet(TABLE).set_index("instance_id")["lb_rate"])
    assert features["lb_rate"].notna().all()
    agents = sorted(outcomes.columns)
    cols = GROUPS["gold"] + ["lb_rate"]
    parts = []
    for agent in agents:
        train_agents = [a for a in agents if a != agent]
        part = features.loc[features["agent"] == agent, ["agent", "task_id", "k", "next_cost", "outcome"]].copy()
        part["f"], rounds = predict_agent(features, outcomes, agent, train_agents, cols, None, seed)
        parts.append(part)
        first = part[part["k"] == 0]
        print(f"{agent}: true {first['outcome'].mean():.3f}, step-0 mean {first['f'].mean():.3f}, "
              f"step-0 Brier {((first['f'] - first['outcome']) ** 2).mean():.3f}, {rounds} trees",
              flush=True)
    out = Path("data/predictor/predictions_gold_loao_lbrate.pkl")
    pd.concat(parts, ignore_index=True).to_pickle(out)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["fetch", "train"])
    args = parser.parse_args()
    fetch() if args.step == "fetch" else train()


if __name__ == "__main__":
    main()
