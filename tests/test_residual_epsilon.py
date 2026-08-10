import math

import numpy as np
import pytest
from pydantic import ValidationError

from mdpagg.adaptive import AdaptiveState, ResidualSpanEpsilon
from mdpagg.config import RunCfg
from mdpagg.partition import allocate
from mdpagg.run import make_epsilon
from mdpagg.types import VALUE


def state_with(residual_span: float) -> AdaptiveState:
    return AdaptiveState(
        v=np.zeros(2, dtype=VALUE),
        w=np.zeros(1, dtype=VALUE),
        part=allocate(2, 1),
        residual_span=residual_span,
    )


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
