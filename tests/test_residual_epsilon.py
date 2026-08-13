import math

import numpy as np
import pytest
from conftest import requires_jit
from fixtures import tiny_gridworld
from pydantic import ValidationError

from mdpagg.adaptive import (
    AdaptiveState,
    AlternatingSchedule,
    FixedEpsilon,
    GeometricEpsilon,
    ResidualSpanEpsilon,
    run_adaptive,
)
from mdpagg.config import RunCfg
from mdpagg.partition import allocate
from mdpagg.rng import streams
from mdpagg.run import make_epsilon
from mdpagg.types import VALUE


def state_with(residual_span: float) -> AdaptiveState:
    return AdaptiveState(
        v=np.zeros(2, dtype=VALUE),
        w=np.zeros(1, dtype=VALUE),
        part=allocate(2, 1),
        residual_span=residual_span,
    )


def state_at_cycle(cycle: int) -> AdaptiveState:
    state = state_with(math.inf)
    state.cycle = cycle
    return state


@pytest.mark.parametrize("residual_span", [math.inf, -math.inf, math.nan])
def test_nonfinite_residual_uses_epsilon_floor(residual_span):
    epsilon = ResidualSpanEpsilon(c=0.25, eps_min=0.01)

    assert epsilon(state_with(residual_span)) == 0.01


def test_epsilon_tracks_residual_span_above_floor():
    epsilon = ResidualSpanEpsilon(c=0.25, eps_min=0.01)

    assert epsilon(state_with(0.8)) == pytest.approx(0.2)


def test_epsilon_never_falls_below_floor():
    epsilon = ResidualSpanEpsilon(c=0.25, eps_min=0.01)

    assert epsilon(state_with(0.001)) == 0.01


def residual_config() -> RunCfg:
    return RunCfg.model_validate(
        {
            "problem": {
                "kind": "maze",
                "dims": [20, 20],
                "p": 0.92,
                "seed": 0,
                "gamma": 0.95,
            },
            "algorithm": {
                "iterations": 100,
                "epsilon": {
                    "kind": "residual_span",
                    "c": 0.25,
                    "eps_min": 0.01,
                },
            },
        }
    )


def test_residual_config_dispatches_to_policy():
    policy = make_epsilon(residual_config().algorithm.epsilon)

    assert isinstance(policy, ResidualSpanEpsilon)
    assert policy.c == 0.25
    assert policy.eps_min == 0.01


@pytest.mark.parametrize("field", ["c", "eps_min"])
def test_residual_config_requires_positive_parameters(field):
    doc = residual_config().model_dump()
    doc["algorithm"]["epsilon"][field] = 0.0

    with pytest.raises(ValidationError):
        RunCfg.model_validate(doc)


## The non-adaptive arm: eps falls on a timetable with no reference to the
## residual, which is what separates feedback from ordinary annealing.


def test_geometric_starts_at_eps_0():
    epsilon = GeometricEpsilon(eps_0=0.5, eps_min=0.05, cycles=14)

    assert epsilon(state_at_cycle(0)) == 0.5


def test_geometric_reaches_the_floor_on_the_last_scheduled_cycle():
    ## i = C-1 puts the exponent at exactly 1, so this is an endpoint the
    ## schedule promises rather than approaches.
    epsilon = GeometricEpsilon(eps_0=0.5, eps_min=0.05, cycles=14)

    assert epsilon(state_at_cycle(13)) == pytest.approx(0.05)


def test_geometric_holds_the_floor_past_the_schedule():
    ## A run has more aggregate cycles than C whenever C anneals early, which
    ## is exactly the rate-matched arm.
    epsilon = GeometricEpsilon(eps_0=0.5, eps_min=0.05, cycles=14)

    assert epsilon(state_at_cycle(14)) == 0.05
    assert epsilon(state_at_cycle(1000)) == 0.05


def test_geometric_midpoint_is_the_geometric_mean():
    ## Distinguishes geometric decay from linear: the middle value is
    ## sqrt(eps_0 * eps_min), not their average.
    epsilon = GeometricEpsilon(eps_0=1.0, eps_min=0.01, cycles=3)

    assert epsilon(state_at_cycle(1)) == pytest.approx(0.1)


def test_geometric_decreases_every_cycle_until_the_floor():
    epsilon = GeometricEpsilon(eps_0=0.5, eps_min=0.05, cycles=14)
    seen = [epsilon(state_at_cycle(i)) for i in range(14)]

    assert all(b < a for a, b in zip(seen, seen[1:], strict=False))


def test_geometric_rejects_a_single_cycle():
    ## C = 1 divides by zero in the exponent.
    with pytest.raises(ValueError, match="cycles"):
        GeometricEpsilon(eps_0=0.5, eps_min=0.05, cycles=1)


def test_geometric_rejects_a_floor_above_its_start():
    ## eps_min >= eps_0 floors every cycle, so the arm would silently become a
    ## constant-eps run wearing a schedule's name.
    with pytest.raises(ValueError, match="eps_min"):
        GeometricEpsilon(eps_0=0.05, eps_min=0.5, cycles=14)


def geometric_config() -> RunCfg:
    doc = residual_config().model_dump()
    doc["algorithm"]["epsilon"] = {
        "kind": "geometric",
        "eps_0": 0.4985,
        "eps_min": 0.05,
        "cycles": 14,
    }
    return RunCfg.model_validate(doc)


def test_geometric_config_dispatches_to_policy():
    policy = make_epsilon(geometric_config().algorithm.epsilon)

    assert isinstance(policy, GeometricEpsilon)
    assert policy.eps_0 == 0.4985
    assert policy.eps_min == 0.05
    assert policy.cycles == 14


def test_geometric_config_requires_at_least_two_cycles():
    doc = geometric_config().model_dump()
    doc["algorithm"]["epsilon"]["cycles"] = 1

    with pytest.raises(ValidationError):
        RunCfg.model_validate(doc)


def test_geometric_config_requires_ordered_endpoints():
    doc = geometric_config().model_dump()
    doc["algorithm"]["epsilon"]["eps_min"] = 0.6

    with pytest.raises(ValidationError):
        RunCfg.model_validate(doc)


## Adding the experimental arms must leave the fixed control untouched. The
## cycle counter increments on every aggregate entry, including the fixed arm's,
## so this is a real claim rather than a tautology. Values were captured from
## the fixed arm before the counter existed.

FIXED_ARM_V = (
    3.792765015596272,
    3.05532201132702,
    2.3164479027169165,
    3.05532201132702,
    2.3164479027169165,
    1.5422879808159884,
    2.3164479027169165,
    1.5422879808159884,
    0.542287980815988,
)


@requires_jit
def test_fixed_arm_is_unchanged_by_the_experimental_arms():
    f = tiny_gridworld()

    result = run_adaptive(
        f.mdp, f.gamma, 200, AlternatingSchedule(2, 5),
        FixedEpsilon(0.25), streams(7).sampling, max_groups=16,
    )

    assert [float(x) for x in result.v] == list(FIXED_ARM_V)
    assert result.t_sa == 143
    assert result.counters.billed == 1222
    assert result.counters.actual == 1744


def test_the_loop_hands_the_geometric_policy_cycle_zero_first():
    ## The counter increments after the policy is consulted, so the first
    ## aggregate entry anneals from eps_0. Incrementing before would drop the
    ## coarsest partition -- invisible to any test that builds the state by hand.
    f = tiny_gridworld()
    seen: list[float] = []

    def observe(t, phase, state):
        if t % 7 == 2:
            seen.append(state.part.eps_effective)

    run_adaptive(
        f.mdp, f.gamma, 60, AlternatingSchedule(2, 5),
        GeometricEpsilon(eps_0=0.5, eps_min=0.05, cycles=3),
        streams(7).sampling, max_groups=16, observer=observe,
    )

    assert seen[0] == pytest.approx(0.5)
    assert seen[1] == pytest.approx(math.sqrt(0.5 * 0.05))
    assert seen[2] == pytest.approx(0.05)
    assert seen[3] == pytest.approx(0.05)


def test_the_cycle_counter_never_resets():
    ## Same reasoning as t_sa: it lives on AdaptiveState precisely so a
    ## refactor cannot turn it into a per-phase local.
    f = tiny_gridworld()

    result = run_adaptive(
        f.mdp, f.gamma, 200, AlternatingSchedule(2, 5),
        FixedEpsilon(0.25), streams(7).sampling, max_groups=16,
    )

    assert result.cycle == 200 // 7 + 1
