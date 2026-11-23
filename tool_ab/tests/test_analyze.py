import json

import pytest
from analyze import compare, load_runs, mcnemar_exact, paired_bootstrap_ci


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    # 10 discordant pairs all one way: 2 * 0.5**10
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(0, 10) == mcnemar_exact(10, 0)


def test_bootstrap_ci_brackets_the_mean_and_is_degenerate_for_constants():
    lo, hi = paired_bootstrap_ci([0, 1, 0, 1, 1, 0, 1, 0], n_boot=2000, seed=0)
    assert lo < 0.5 < hi
    assert paired_bootstrap_ci([1, 1, 1], n_boot=200, seed=0) == (1.0, 1.0)


def make_runs(outcomes, tokens=100):
    return {t: {"outcome": y, "tokens": tokens} for t, y in outcomes.items()}


def test_compare_identical_arms_finds_no_difference():
    runs = make_runs({f"t{i}": i % 2 for i in range(20)})
    result = compare(runs, runs, n_boot=500, seed=0)
    assert result["diff"] == 0
    assert result["p_value"] == 1.0
    assert result["verdict"] == "no detectable difference"


def test_compare_detects_a_large_improvement_on_paired_tasks():
    base = make_runs({f"t{i}": 0 for i in range(40)}, tokens=100)
    arm = make_runs({f"t{i}": int(i < 20) for i in range(40)}, tokens=80)
    arm["extra"] = {"outcome": 1, "tokens": 1}  # unpaired tasks are ignored
    result = compare(base, arm, n_boot=500, seed=0)
    assert result["n_tasks"] == 40
    assert result["diff"] == pytest.approx(0.5)
    assert result["arm_better"] == 20 and result["base_better"] == 0
    assert result["verdict"] == "better"
    assert result["token_ratio"] == pytest.approx(0.8)


def test_load_runs_sums_tokens_per_task(tmp_path):
    rows = [
        {"task_id": "t1", "agent": "a", "step": 1, "tokens_in": 10, "tokens_out": 2,
         "action": "", "observation": "", "outcome": 1},
        {"task_id": "t1", "agent": "a", "step": 2, "tokens_in": 20, "tokens_out": 3,
         "action": "", "observation": "", "outcome": 1},
    ]
    path = tmp_path / "a.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_runs(path) == {"t1": {"outcome": 1, "tokens": 35, "steps": 2}}


@pytest.mark.parametrize(("lo", "hi", "p", "diff", "expected"), [
    (-0.23, -0.05, 0.007, -0.14, "worse"),
    (-0.15, -0.01, 0.057, -0.08, "borderline worse"),  # the CI alone
    (-0.10, 0.01, 0.04, -0.05, "borderline worse"),  # McNemar alone
    (-0.14, 0.02, 0.238, -0.06, "no detectable difference"),
    (0.1, 0.3, 0.001, 0.2, "better"),
])
def test_verdict_needs_both_tests_for_a_firm_call(lo, hi, p, diff, expected):
    from analyze import verdict_of

    assert verdict_of(lo, hi, p, diff) == expected
