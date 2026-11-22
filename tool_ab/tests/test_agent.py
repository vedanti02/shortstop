"""End-to-end runs of the plug-in with a scripted model and a local shell (no API calls)."""

import random

import pytest
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output
from shortstop_plugin.agent import StoppingAgent

from shortstop import rr_estimate

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo patch"


def scripted_model(n_work_steps=3, work="echo working"):
    outputs = [make_output(f"step {i}", [{"command": work}]) for i in range(n_work_steps)]
    outputs.append(make_output("submit", [{"command": SUBMIT}]))
    return DeterministicModel(outputs=outputs)


def make_agent(tmp_path, model=None, **config):
    env = LocalEnvironment(cwd=str(tmp_path))
    return StoppingAgent(
        model or scripted_model(), env,
        system_template="You are a test agent.", instance_template="{{task}}",
        instance_id="task-1", **{"step_limit": 20, "cost_limit": 0, **config},
    )


def test_never_stop_runs_to_submission_and_records_the_policy(tmp_path):
    agent = make_agent(tmp_path)
    info = agent.run("do it")
    assert info["exit_status"] == "Submitted"
    record = agent.serialize()["info"]["shortstop"]
    assert record["steps_run"] == 4
    assert record["stopped_early"] is False
    assert record["continue_probs"] == [1.0] * 4
    assert record["predictions"] == [0.5] * 4


def test_stopping_before_step_one(tmp_path):
    agent = make_agent(tmp_path, stop_policy="shortstop_plugin.policies.constant",
                       stop_policy_kwargs={"p": 1e-9}, seed=0)
    info = agent.run("do it")
    assert info["exit_status"] == "EarlyStopped"
    record = agent.serialize()["info"]["shortstop"]
    assert record["steps_run"] == 0
    assert record["stopped_early"] is True
    assert len(record["continue_probs"]) == 1  # the draw that said stop is kept


def test_record_has_what_the_estimator_needs(tmp_path):
    stops = []
    for seed in range(30):
        agent = make_agent(tmp_path, stop_policy="shortstop_plugin.policies.constant",
                           stop_policy_kwargs={"p": 0.6}, seed=seed)
        agent.run("do it")
        rec = agent.serialize()["info"]["shortstop"]
        s = rec["steps_run"]
        assert len(rec["predictions"]) >= s and len(rec["continue_probs"]) >= s
        f = rec["predictions"][: s + 1] if rec["stopped_early"] else rec["predictions"] + [1]
        rr_estimate(f, rec["continue_probs"], s)  # must not raise
        stops.append(s)
    assert len(set(stops)) > 1  # randomization actually varies the stop step


def test_same_seed_and_task_give_the_same_stop_step(tmp_path):
    kwargs = {"stop_policy": "shortstop_plugin.policies.constant",
              "stop_policy_kwargs": {"p": 0.5}, "seed": 7}
    a, b = make_agent(tmp_path, **kwargs), make_agent(tmp_path, **kwargs)
    a.run("x")
    b.run("x")
    assert a.serialize()["info"]["shortstop"] == b.serialize()["info"]["shortstop"]


class FlakyModel(DeterministicModel):
    """Fails to parse its first response, like a model that broke the tool-call format."""

    def query(self, messages, **kwargs):
        if not getattr(self, "_failed_once", False):
            self._failed_once = True
            raise FormatError({"role": "user", "content": "bad format", "extra": {}})
        return super().query(messages, **kwargs)


def test_one_draw_per_step_even_after_a_format_error(tmp_path):
    model = FlakyModel(outputs=[
        make_output("work", [{"command": "echo working"}]),
        make_output("submit", [{"command": SUBMIT}]),
    ])
    calls = []

    def policy(prefix, predictions):
        calls.append(len(predictions))
        return 1.0

    agent = make_agent(tmp_path, model=model)
    agent._policy = policy
    agent.run("x")
    assert agent.serialize()["info"]["shortstop"]["steps_run"] == 2
    assert calls == [1, 2]  # step 1 was queried twice but drawn once


def test_setup_scripts_are_installed_before_the_first_step(tmp_path):
    bin_dir = tmp_path / "bin"
    agent = make_agent(tmp_path, model=scripted_model(1, work=f"{bin_dir}/search -f nothing"),
                       setup_scripts=["search"], script_install_dir=str(bin_dir))
    agent.run("x")
    assert (bin_dir / "search").exists()
    assert "No files" in agent.messages[3]["content"]
    assert agent.serialize()["info"]["shortstop"]["steps_run"] == 2


def test_compression_is_applied_to_observations(tmp_path):
    noisy = "for i in $(seq 1 50); do echo same line; done"
    agent = make_agent(tmp_path, model=scripted_model(1, work=noisy),
                       compression={"max_chars": 2000, "max_line_chars": 200})
    agent.run("x")
    observation = agent.messages[3]["content"]
    assert observation.count("same line") == 1
    assert "repeated 49 more times" in observation


def test_invalid_policy_output_fails_loudly(tmp_path):
    agent = make_agent(tmp_path)
    agent._policy = lambda prefix, predictions: 0.0
    with pytest.raises(ValueError):
        agent.run("x")


def test_unseeded_agents_use_independent_randomness(tmp_path):
    random.seed(0)
    agent = make_agent(tmp_path, seed=None)
    assert agent._rng is not random  # never shares the global generator


def never_submitting_model(n=10):
    return DeterministicModel(outputs=[make_output(f"s{i}", [{"command": "echo w"}]) for i in range(n)])


def test_no_draw_when_the_step_limit_ends_the_run(tmp_path):
    agent = make_agent(tmp_path, model=never_submitting_model(), step_limit=3,
                       stop_policy="shortstop_plugin.policies.constant",
                       stop_policy_kwargs={"p": 1e-9}, seed=0)
    agent._rng.random = lambda: 0.0  # always continue whenever a draw happens
    info = agent.run("x")
    assert info["exit_status"] == "LimitsExceeded"
    record = agent.serialize()["info"]["shortstop"]
    assert record["stopped_early"] is False
    assert record["steps_run"] == 3
    assert len(record["continue_probs"]) == 3  # p_1..p_T only, no draw for a step that cannot run
    assert len(record["predictions"]) == 3


def test_predictor_and_policy_are_loaded_once_per_config(tmp_path):
    a, b = make_agent(tmp_path), make_agent(tmp_path)
    assert a._predictor is b._predictor and a._policy is b._policy
    c = make_agent(tmp_path, predictor_kwargs={"p": 0.3})
    assert c._predictor is not a._predictor


def test_a_failing_policy_leaves_the_record_consistent(tmp_path):
    agent = make_agent(tmp_path)
    agent._policy = lambda prefix, predictions: 2.0
    with pytest.raises(ValueError):
        agent.run("x")
    record = agent.shortstop_record()
    assert len(record["predictions"]) == len(record["continue_probs"]) == 0
