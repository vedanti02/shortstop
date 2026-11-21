"""Why the learned predictor did not beat LightGBM: where the cost comes from and what is fixable.

    python predictor/deep_dive.py

Nine analyses, each written to predictor/results/deep_dive_<name>.csv (theta also writes
theta_error):

    cost        Split each agent's estimator variance (coverage/THEORY.md, Lemma 2) into the
                outcome's own variance, the part caused by the step-0 prediction, and the part
                caused by predictions during the run; and the compute if either were perfect.
    step0       Split the step-0 error into not knowing the agent, task-level misses, and luck.
    luck        How much of a run's outcome is luck: the same agent on the same task, from
                Nebius (about 10 runs of one agent per task) and from near-identical agent pairs.
    arrival     How good the prediction is after a given share of the run's tokens.
    weighted    Plain against variance-weighted LightGBM, step by step.
    theta       Compute of the ability-aware predictor when its theta is off by a given amount,
                and how far off the in-evaluation estimate of theta typically is.
    importance  What gold LightGBM relies on (share of total split gain).
    edit_timing Share of a run's tokens spent by its last edit, and by its last edit to a file
                the reference patch touches.
    family      Whether a new agent's same-vendor siblings predict which tasks it solves.

All policy-dependent numbers use the tracking policy at kappa 0.25 with a 0.05 floor, the
setting of the simulations in predictor/README.md.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coverage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimal_policy as op
import stopping
from ability_eval import batch_predictions, pad_grid
from experiments import pad_runs

DATA = Path("data/predictor")
RESULTS = Path(__file__).parent / "results"
KAPPA, FLOOR = 0.25, 0.05
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]


def geo(x) -> float:
    return float(np.exp(np.mean(np.log(x))))


def cost_terms(F: np.ndarray, cost: np.ndarray) -> dict:
    """Per-task variance pieces and cost under the tracking policy (exact, no simulation)."""
    reach = stopping.reach_tracking(F, KAPPA, FLOOR)
    u = 1.0 / reach
    w = u - np.concatenate([np.ones((len(F), 1)), u[:, :-1]], axis=1)  # weight of e_{k-1}^2
    sq = (F[:, -1:] - F)[:, :-1] ** 2
    terms = w * sq
    y = F[:, -1]
    var_y, v0, v_run = y.var(), terms[:, 0].mean(), terms[:, 1:].sum(axis=1).mean()
    cost_frac = (cost * reach).sum(axis=1).mean() / cost.sum(axis=1).mean()
    rel = lambda v: (var_y + v) * cost_frac / var_y
    return {"rate": y.mean(), "var_y": var_y, "v_step0": v0, "v_in_run": v_run,
            "cost_fraction": cost_frac, "relative_compute": rel(v0 + v_run),
            "if_perfect_step0": rel(v_run), "if_perfect_in_run": rel(v0)}


def analysis_cost() -> pd.DataFrame:
    rows = []
    for name, path in [("stand-in", "data/swebench_verified/predictions_loao.pkl"),
                       ("LightGBM gold", DATA / "predictions_gold_loao.pkl"),
                       ("ability-aware, true theta", DATA / "predictions_ability_gold_oracle.pkl")]:
        for agent, g in pd.read_pickle(path).groupby("agent", sort=True):
            rows.append({"predictor": name, "agent": agent, **cost_terms(*pad_runs(g))})
    return pd.DataFrame(rows)


def analysis_luck() -> pd.DataFrame:
    rows = []
    neb = pd.read_pickle("data/openhands/features_rich.pkl", compression=None)
    first = neb.loc[neb["k"] == 0, ["task_id", "outcome"]].copy()
    first["task"] = first["task_id"].str.split("@").str[0]
    g = first.groupby("task")["outcome"].agg(["size", "mean"])
    g = g[g["size"] >= 2]
    within = (g["size"] / (g["size"] - 1) * g["mean"] * (1 - g["mean"]))  # unbiased per task
    p = first["outcome"].mean()
    rows.append({"source": "Nebius: Qwen3-Coder, ~10 runs per task", "tasks": len(g),
                 "luck_brier": float(np.average(within, weights=g["size"])),
                 "outcome_variance": p * (1 - p), "kind": "same agent, repeated runs"})
    pairs = pd.read_csv(RESULTS / "paired_check.csv")
    for _, r in pairs[~pairs["pair"].str.endswith("(reversed)")].iterrows():
        d = 1 - r["agreement"]
        rows.append({"source": r["pair"], "tasks": 500, "luck_brier": d / 2,
                     "outcome_variance": r["rate_variant"] * (1 - r["rate_variant"]),
                     "kind": "two configs (upper bound on luck)"})
    out = pd.DataFrame(rows)
    out["luck_share_of_variance"] = out["luck_brier"] / out["outcome_variance"]
    return out


def analysis_step0(luck: pd.DataFrame) -> pd.DataFrame:
    held = pd.read_pickle(DATA / "predictions_gold_loao.pkl").query("k == 0")
    known = pd.read_pickle(DATA / "predictions_ability_gold_oracle.pkl").query("k == 0")
    rows = []
    for agent, g in held.groupby("agent", sort=True):
        k = known[known["agent"] == agent]
        rows.append({"agent": agent, "rate": g["outcome"].mean(),
                     "brier_held_out": ((g["f"] - g["outcome"]) ** 2).mean(),
                     "calibration_term": (g["f"].mean() - g["outcome"].mean()) ** 2,
                     "brier_agent_known": ((k["f"] - k["outcome"]) ** 2).mean()})
    out = pd.DataFrame(rows)
    close = luck[(luck["kind"] != "same agent, repeated runs")].nsmallest(2, "luck_brier")
    out["luck_upper_bound"] = close["luck_brier"].mean()  # closest config pairs
    return out


def analysis_arrival(path: Path, name: str) -> pd.DataFrame:
    p = pd.read_pickle(path).sort_values(["agent", "task_id", "k"])
    run = p.groupby(["agent", "task_id"], sort=False)
    p["spent"] = run["next_cost"].cumsum() - p["next_cost"]
    p["frac"] = p["spent"] / run["next_cost"].transform("sum").clip(lower=1)
    p["last"] = p["k"] == run["k"].transform("max")
    rows = []
    for agent, g in p.groupby("agent", sort=True):
        row = {"predictor": name, "agent": agent}
        for frac in FRACTIONS:
            pick = g[g["frac"] <= frac].groupby("task_id").tail(1)
            row[f"brier_{frac}"] = float(((pick["f"] - pick["outcome"]) ** 2).mean())
        last = g[g["last"]]
        row["brier_last_step"] = float(((last["f"] - last["outcome"]) ** 2).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in [c for c in out.columns if c.startswith("brier_") and c != "brier_0.0"]:
        out["resolved_" + col.removeprefix("brier_")] = 1 - out[col] / out["brier_0.0"]
    return out


def analysis_weighted() -> pd.DataFrame:
    edges = [0, 1, 5, 20, 50, 10_000]
    labels = ["step 0", "steps 1-4", "steps 5-19", "steps 20-49", "step 50+"]
    rows = []
    for name, path in [("plain", "predictions_gold_loao"), ("variance-weighted", "predictions_gold_loao_vw")]:
        p = pd.read_pickle(DATA / f"{path}.pkl")
        p["bucket"] = pd.cut(p["k"], edges, right=False, labels=labels)
        b = p.assign(sq=(p["f"] - p["outcome"]) ** 2).groupby("bucket", observed=True)["sq"].mean()
        rows.append({"predictor": name, **b.to_dict()})
    w = pd.read_pickle(DATA / "weights_predictions_gold_loao.pkl")
    feats = pd.read_pickle("data/swebench_verified/features_rich.pkl")[["k"]]
    share = w.groupby(pd.cut(feats.loc[w.index, "k"], edges, right=False, labels=labels), observed=True).sum()
    rows.append({"predictor": "share of training weight", **(share / share.sum()).to_dict()})
    return pd.DataFrame(rows)


def analysis_theta() -> pd.DataFrame:
    with open(DATA / "ability_grid_gold.pkl", "rb") as f:
        grids = pickle.load(f)
    deltas = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
    rows = []
    for agent, (rows_df, grid, preds) in sorted(grids.items()):
        L, cost, y = pad_grid(rows_df, preds)
        tasks = np.arange(len(y))[None, :]
        for d in deltas:
            theta = np.clip(np.array([y.mean() + d]), grid[0], grid[-1])
            F = batch_predictions(L, tasks, theta, grid)[0]
            reach = stopping.reach_tracking(F, KAPPA, FLOOR)
            rows.append({"agent": agent, "theta_error": d,
                         "relative_compute": op.relative_compute(F, cost, reach)})
    return pd.DataFrame(rows)


def theta_error_curve(cost: pd.DataFrame, prior: float = 100.0) -> pd.DataFrame:
    """Typical error of the damped in-evaluation theta after m tasks (normal approximation)."""
    gb = cost[cost["predictor"] == "LightGBM gold"].copy()
    sigma2 = gb["var_y"] + gb["v_step0"] + gb["v_in_run"]  # variance of one task's estimate
    mean_rate = gb["rate"].mean()
    rows = []
    for m in [25, 50, 100, 200, 300, 400, 475]:
        sd = np.sqrt(m * sigma2) / (m + prior)
        bias = prior * (mean_rate - gb["rate"]) / (m + prior)
        rows.append({"tasks_seen": m, "rms_theta_error": float(np.sqrt((sd**2 + bias**2).mean()))})
    return pd.DataFrame(rows)


def analysis_importance() -> pd.DataFrame:
    from train_lgbm import GROUPS, crossfit_task_rate, fit
    f = pd.read_pickle("data/swebench_verified/features_rich.pkl")
    o = f[f["k"] == 0].pivot(index="task_id", columns="agent", values="outcome").dropna()
    agents = sorted(o.columns)
    train = f[f["agent"].isin(agents[1:])].copy()
    train["task_rate"] = crossfit_task_rate(train, o, agents[1:], 0)
    model = fit(train, GROUPS["gold"], None, 0)
    gain = pd.Series(model.feature_importance("gain"), index=GROUPS["gold"])
    share = (gain / gain.sum()).sort_values(ascending=False).rename("gain_share")
    return share.rename_axis("feature").reset_index()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    cost = analysis_cost()
    luck = analysis_luck()
    out = {"cost": cost, "luck": luck, "step0": analysis_step0(luck),
           "arrival": pd.concat([analysis_arrival(DATA / "predictions_gold_loao.pkl", "LightGBM gold"),
                                 analysis_arrival(DATA / "predictions_base_loao.pkl", "LightGBM base")]),
           "weighted": analysis_weighted(), "theta": analysis_theta(),
           "theta_error": theta_error_curve(cost), "importance": analysis_importance(),
           "edit_timing": analysis_edit_timing(), "family": analysis_family()}
    for name, df in out.items():
        df.to_csv(RESULTS / f"deep_dive_{name}.csv", index=False)

    pd.set_option("display.width", 200)
    c = cost.groupby("predictor", sort=False)
    total = cost["var_y"] + cost["v_step0"] + cost["v_in_run"]
    cost = cost.assign(share_var_y=cost["var_y"] / total, share_step0=cost["v_step0"] / total,
                       share_in_run=cost["v_in_run"] / total)
    print("== cost: where the variance comes from (mean shares) and compute (geometric mean)")
    print(cost.groupby("predictor", sort=False)[["share_var_y", "share_step0", "share_in_run", "cost_fraction"]].mean().round(3))
    print(c[["relative_compute", "if_perfect_step0", "if_perfect_in_run"]].agg(geo).round(3))
    print("\n== luck"); print(luck.round(3).to_string(index=False))
    print("\n== step0 (means over agents)"); print(out["step0"].drop(columns=["agent"]).mean().round(4))
    print("\n== arrival (means over agents)")
    print(out["arrival"].groupby("predictor").mean(numeric_only=True).round(3).T)
    print("\n== weighted"); print(out["weighted"].round(4).to_string(index=False))
    print("\n== theta (geometric mean over agents)")
    print(out["theta"].groupby("theta_error")["relative_compute"].agg(geo).round(3).to_string())
    print(out["theta_error"].round(3).to_string(index=False))
    print("\n== importance (top 10)"); print(out["importance"].head(10).round(3).to_string(index=False))



def analysis_edit_timing() -> pd.DataFrame:
    """Share of a run's tokens spent by the time of its last edit (and last edit to a gold file).

    If the decisive edit lands well before the end, running the benchmark's tests on the code at
    that point would reveal the outcome early, the mid-run signal the predictor lacks.
    """
    f = pd.read_pickle("data/swebench_verified/features_rich.pkl")[
        ["agent", "task_id", "k", "edits", "gold_edited_frac", "next_cost", "outcome"]]
    f = f.sort_values(["agent", "task_id", "k"])
    run = f.groupby(["agent", "task_id"], sort=False)
    total = run["next_cost"].transform("sum").clip(lower=1)
    f["spent_after_k"] = (run["next_cost"].cumsum()) / total  # tokens through step k + 1
    rows = []
    for col, label in [("edits", "last edit"), ("gold_edited_frac", "last edit to a gold file")]:
        final = run[col].transform("max")
        # first row whose count already equals the final count: the edit happened in step k
        hit = f[(f[col] >= final) & (final > 0)].groupby(["agent", "task_id"]).head(1)
        prev = f.loc[hit.index - 1] if len(hit) else hit
        spent = pd.Series(np.where(hit["k"].to_numpy() > 0, prev["spent_after_k"].to_numpy(), 0.0), index=hit.index)
        for outcome, g in hit.assign(spent=spent).groupby("outcome"):
            rows.append({"event": label, "outcome": int(outcome), "runs": len(g),
                         "median_token_share": g["spent"].median(),
                         "p25": g["spent"].quantile(0.25), "p75": g["spent"].quantile(0.75)})
    return pd.DataFrame(rows)


def analysis_family() -> pd.DataFrame:
    """Does a new agent's model family (same vendor) predict which tasks it solves?"""
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from standin_predictor import vendor
    p = pd.read_pickle(DATA / "predictions_gold_loao.pkl").query("k == 0")
    Y = p.pivot(index="task_id", columns="agent", values="outcome")
    F0 = p.pivot(index="task_id", columns="agent", values="f").clip(1e-3, 1 - 1e-3)
    agents = sorted(Y.columns)
    clip = lambda s: s.clip(0.02, 0.98)

    def design(a: str) -> np.ndarray:
        mates = [b for b in agents if b != a and vendor(b) == vendor(a)]
        others = [b for b in agents if b != a]
        family = logit(clip(Y[mates].mean(axis=1))) - logit(clip(Y[others].mean(axis=1)))
        return np.column_stack([logit(F0[a]), family])

    X = {a: design(a) for a in agents}
    rows = []
    for a in agents:
        train = [b for b in agents if b != a]
        model = LogisticRegression().fit(np.vstack([X[b] for b in train]),
                                         np.concatenate([Y[b] for b in train]))
        y = Y[a].to_numpy()
        b_lgbm = float(((F0[a] - y) ** 2).mean())
        b_fam = float(((model.predict_proba(X[a])[:, 1] - y) ** 2).mean())
        rows.append({"agent": a, "vendor": vendor(a),
                     "mates": sum(vendor(b) == vendor(a) for b in train),
                     "brier0_lgbm": b_lgbm, "brier0_family": b_fam,
                     "one_shot_lgbm": 0.05 + 0.95 * b_lgbm / y.var(),
                     "one_shot_family": 0.05 + 0.95 * b_fam / y.var()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
