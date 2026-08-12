import math

import numpy as np
import pytest
from fixtures import two_state_chain
from pydantic import ValidationError

from mdpagg import run as run_module
from mdpagg.adaptive import AdaptiveState
from mdpagg.config import (
    AlgorithmCfg,
    FixedEpsilonCfg,
    MazeProblem,
    RunCfg,
    TraceCfg,
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


def cfg_with_checkpoints(iterations: int, fine_stride: int, at: tuple[int, ...]) -> RunCfg:
    return RunCfg(
        problem=MazeProblem(dims=(4, 4), p=0.9, seed=0),
        algorithm=AlgorithmCfg(iterations=iterations, epsilon=FixedEpsilonCfg(value=0.5)),
        trace=TraceCfg(fine_stride=fine_stride, policy_loss_at=at),
    )


def test_a_checkpoint_off_the_fine_stride_is_rejected():
    """The observer never sees an iteration it did not trace: it returns early
    unless `wants_row(t)`. A checkpoint between traced rows is therefore not a
    sparse checkpoint -- it is one that never fires at all, leaving the curve
    short of a point that the config file still advertises. Silent, and only
    visible by counting rows in a finished result file."""
    with pytest.raises(ValidationError, match="never fire"):
        cfg_with_checkpoints(iterations=100, fine_stride=10, at=(25,))


def test_a_checkpoint_at_or_past_the_last_iteration_is_rejected():
    """The loop runs `t` over `range(iterations)`, so `t == iterations` is one
    past the end. Asking to score there is the natural way to write "and at the
    finish", and it would silently score nothing -- the final vector is already
    scored by `summary_of`, outside the trace."""
    with pytest.raises(ValidationError, match="never fire"):
        cfg_with_checkpoints(iterations=100, fine_stride=10, at=(100,))


def test_checkpoints_on_the_stride_and_inside_the_horizon_are_accepted():
    cfg = cfg_with_checkpoints(iterations=100, fine_stride=10, at=(0, 50, 90))

    assert cfg.trace.policy_loss_at == (0, 50, 90)
