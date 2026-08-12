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
from mdpagg.policy import greedy_policy, policy_loss
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

    monkeypatch.setattr(run_module, "loss_of_policy", fake_loss)
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

    monkeypatch.setattr(run_module, "loss_of_policy", lambda *_args: 123.0)
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


def test_the_scorer_reuses_its_answer_when_the_greedy_policy_is_unchanged(monkeypatch):
    """5.6's warm-start, in the one form that cannot move a number.

    Warm-starting `policy_value` from the previous iterate is the obvious
    reading and the wrong one: span-based stopping leaves the returned vector
    dependent on where the iteration began, so `policy_loss` would become a
    function of the checkpoint schedule. The shift is about tol/(1-gamma), some
    2e-9, invisible in any reported digit -- but 6.2 asks whether the fixed
    arm's traces are unchanged, and a control whose numbers move when the
    measurement schedule moves is not a control.

    Keying on the policy instead makes the saving exact: identical policy,
    identical evaluation, identical number. Adding a constant to V shifts every
    Q(s, a) by gamma times that constant, so the greedy policy is unchanged by
    construction -- which is what distinguishes a cache keyed on the policy from
    one keyed on the value vector.
    """
    fixture, _cfg, truth, _state = run_fixture()
    calls = 0

    def counted(*_args):
        nonlocal calls
        calls += 1
        return 7.0

    assert np.array_equal(
        greedy_policy(fixture.mdp, fixture.v_star, fixture.gamma),
        greedy_policy(fixture.mdp, fixture.v_star + 1000.0, fixture.gamma),
    ), "the shift moved the greedy policy; the fixture has a near-tie"

    monkeypatch.setattr(run_module, "loss_of_policy", counted)
    scorer = run_module.PolicyScorer(fixture.mdp, truth.v_star, fixture.gamma, 1e-10)

    first = scorer(fixture.v_star)
    second = scorer(fixture.v_star + 1000.0)

    assert first == second == 7.0
    assert calls == 1
    assert scorer.evaluations == 1


def test_the_scorer_returns_exactly_what_the_uncached_path_returns():
    """A cache keyed too loosely reports a stale policy's score under a new
    policy's label -- a corrupted headline metric with nothing else to catch it.
    Exact equality, not approx: both paths run the same evaluator on the same
    policy at the same tolerance, so anything but a bitwise match means the
    cache changed the computation."""
    fixture, _cfg, truth, _state = run_fixture()
    scorer = run_module.PolicyScorer(fixture.mdp, truth.v_star, fixture.gamma, 1e-10)

    cached = scorer(fixture.v_star)

    assert cached == policy_loss(
        fixture.mdp, fixture.v_star, truth.v_star, fixture.gamma
    )
