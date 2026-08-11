import math

import numpy as np
from fixtures import two_state_chain

from mdpagg import run as run_module
from mdpagg.adaptive import AdaptiveState
from mdpagg.config import (
    AlgorithmCfg,
    ExecutionCfg,
    FixedEpsilonCfg,
    MazeProblem,
    RunCfg,
)
from mdpagg.partition import allocate as allocate_partition
from mdpagg.solve import GroundTruth
from mdpagg.trace import allocate as allocate_trace
from mdpagg.types import VALUE, Phase


def run_fixture():
    fixture = two_state_chain()
    cfg = RunCfg(
        problem=MazeProblem(
            dims=(1, fixture.mdp.num_states),
            p=0.92,
            seed=0,
            gamma=fixture.gamma,
        ),
        algorithm=AlgorithmCfg(
            iterations=1,
            epsilon=FixedEpsilonCfg(value=0.5),
        ),
    )
    truth = GroundTruth(
        hash="fixture",
        scale=1.0,
        v_star=fixture.v_star,
        iterations=1,
        wall_ns=0,
    )
    state = AdaptiveState(
        v=fixture.v_star.copy(),
        w=np.zeros(1, dtype=VALUE),
        part=allocate_partition(fixture.mdp.num_states, 1),
    )
    return fixture, cfg, truth, state


def test_execution_settings_reach_the_solver():
    """A timing run whose thread count came from the environment is unreadable.

    The result file has to say what was asked for, or a sweep over thread
    counts cannot be told apart from five runs at the machine default.
    """
    fixture, cfg, _truth, _state = run_fixture()
    cfg = cfg.model_copy(
        update={"execution": ExecutionCfg(parallel=True, threads=2)}
    )

    result = run_module.solve(cfg, fixture.mdp, observer=None)

    assert result.parallel is True
    assert result.threads_requested == 2


def test_run_document_records_what_the_solver_was_asked_for():
    fixture, cfg, _truth, _state = run_fixture()
    cfg = cfg.model_copy(
        update={"execution": ExecutionCfg(parallel=True, threads=2)}
    )

    result = run_module.solve(cfg, fixture.mdp, observer=None)
    block = run_module.execution_of(result)

    assert block["parallel"] is True
    assert block["threads_requested"] == 2
    assert block["threads_observed"] == 2
    assert block["threading_layer"] is not None


def test_execution_defaults_to_serial():
    """The reproduction must stay runnable without any parallel transform."""
    _fixture, cfg, _truth, _state = run_fixture()

    assert cfg.execution.parallel is False
    assert cfg.execution.threads is None


def test_observer_can_skip_intermediate_policy_evaluation(monkeypatch):
    fixture, cfg, truth, state = run_fixture()
    trace = allocate_trace(1, fine_stride=1, coarse_stride=1)
    calls = 0

    def fake_loss(*_args):
        nonlocal calls
        calls += 1
        return 123.0

    monkeypatch.setattr(run_module, "loss_against", fake_loss)
    observe = run_module.observer_for(
        cfg,
        fixture.mdp,
        truth,
        trace,
        trace_policy_loss=False,
    )
    observe(0, Phase.GLOBAL, state)

    assert calls == 0
    assert math.isnan(trace.policy_loss[0])


def test_observer_evaluates_requested_policy_loss_by_default(monkeypatch):
    fixture, cfg, truth, state = run_fixture()
    trace = allocate_trace(1, fine_stride=1, coarse_stride=1)

    monkeypatch.setattr(run_module, "loss_against", lambda *_args: 123.0)
    observe = run_module.observer_for(cfg, fixture.mdp, truth, trace)
    observe(0, Phase.GLOBAL, state)

    assert trace.policy_loss[0] == 123.0
