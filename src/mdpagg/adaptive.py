import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numba
import numpy as np

from .backup import backup_lifted
from .counters import Counters
from .mdp import TabularMdp, unpack
from .norms import span_seminorm
from .partition import Partition, allocate, lift_into, rebin_by_value
from .timer import timed
from .types import VALUE, IndexArray, Phase, ValueArray
from .vi import _NO_GROUPS, _sweep

ModelArrays = tuple[IndexArray, IndexArray, IndexArray, ValueArray, ValueArray]


def inverse_sqrt(t: int) -> float:
    return 1.0 / math.sqrt(t)


@dataclass(frozen=True)
class AlternatingSchedule:

    global_len: int = 2
    agg_len: int = 5

    def __post_init__(self) -> None:
        if self.global_len < 1:
            raise ValueError(f"global_len must be >= 1, got {self.global_len}")
        if self.agg_len < 0:
            raise ValueError(f"agg_len must be >= 0, got {self.agg_len}")

    @property
    def cycle(self) -> int:
        return self.global_len + self.agg_len

    def phase_at(self, t: int) -> Phase:
        if t % self.cycle < self.global_len:
            return Phase.GLOBAL
        return Phase.AGGREGATE

    def is_entry(self, t: int) -> bool:
        return t == 0 or self.phase_at(t) is not self.phase_at(t - 1)


@dataclass
class AdaptiveState:

    v: ValueArray
    w: ValueArray
    part: Partition
    t_sa: int = 1
    residual_span: float = math.inf
    counters: Counters = field(default_factory=Counters)


EpsilonPolicy = Callable[[AdaptiveState], float]


@dataclass(frozen=True)
class FixedEpsilon:

    value: float

    def __call__(self, state: AdaptiveState) -> float:
        return self.value


@dataclass(frozen=True)
class ResidualSpanEpsilon:

    c: float
    eps_min: float

    def __call__(self, state: AdaptiveState) -> float:
        if not math.isfinite(state.residual_span):
            return self.eps_min
        return max(self.eps_min, self.c * state.residual_span)


@dataclass(frozen=True)
class AdaptiveResult:

    v: ValueArray
    iterations: int
    t_sa: int
    counters: Counters
    wall_ns: int


@numba.njit
def _aggregate_sweep(
    sa_begin,
    succ_begin,
    succ_state,
    succ_prob,
    cost,
    w,
    out,
    group_of,
    members,
    offset,
    num_groups,
    draws,
    alpha,
    gamma,
):
    for j in range(num_groups):
        lo = offset[j]
        s = members[lo + int(draws[j] * (offset[j + 1] - lo))]
        q = backup_lifted(
            sa_begin, succ_begin, succ_state, succ_prob, cost, s, w, group_of, gamma
        )[0]
        out[j] = (1.0 - alpha) * w[j] + alpha * q


def _warm(arrays: ModelArrays, num_states: int, gamma: float) -> None:

    v = np.zeros(num_states, dtype=VALUE)
    out = np.empty(num_states, dtype=VALUE)
    _sweep(*arrays, v, out, _NO_GROUPS, gamma)

    part = allocate(num_states, 1)
    rebin_by_value(v, 1.0, 1, part)
    w = np.zeros(1, dtype=VALUE)
    w_out = np.empty(1, dtype=VALUE)
    _aggregate_sweep(
        *arrays,
        w,
        w_out,
        part.group_of,
        part.members,
        part.offset,
        1,
        np.zeros(1, dtype=VALUE),
        1.0,
        gamma,
    )
    lift_into(part, w, out)


def run_adaptive(
    mdp: TabularMdp,
    gamma: float,
    iterations: int,
    schedule: AlternatingSchedule,
    epsilon: EpsilonPolicy,
    sampling_rng: np.random.Generator,
    alpha: Callable[[int], float] = inverse_sqrt,
    max_groups: int = 4096,
    observer: Callable[[int, Phase, AdaptiveState], None] | None = None,
) -> AdaptiveResult:
    arrays = unpack(mdp)
    state = AdaptiveState(
        v=np.zeros(mdp.num_states, dtype=VALUE),
        w=np.zeros(min(max_groups, mdp.num_states), dtype=VALUE),
        part=allocate(mdp.num_states, max_groups),
    )
    v_nxt = np.empty_like(state.v)
    w_nxt = np.empty_like(state.w)

    _warm(arrays, mdp.num_states, gamma)

    with timed() as elapsed:
        for t in range(iterations):
            phase = schedule.phase_at(t)

            if phase is Phase.GLOBAL:
                if schedule.is_entry(t) and state.part.num_groups > 0:
                    lift_into(state.part, state.w, state.v)
                    state.counters.lift_ops += mdp.num_states

                _sweep(*arrays, state.v, v_nxt, _NO_GROUPS, gamma)
                state.counters.global_backups += mdp.num_states
                state.residual_span = span_seminorm(v_nxt - state.v)
                state.v, v_nxt = v_nxt, state.v
            else:
                if schedule.is_entry(t):
                    rebin_by_value(state.v, epsilon(state), max_groups, state.part)
                    state.counters.rebin_ops += mdp.num_states
                    state.w[: state.part.num_groups] = state.part.centers[
                        : state.part.num_groups
                    ]

                groups = state.part.num_groups
                _aggregate_sweep(
                    *arrays,
                    state.w,
                    w_nxt,
                    state.part.group_of,
                    state.part.members,
                    state.part.offset,
                    groups,
                    sampling_rng.random(groups),
                    alpha(state.t_sa),
                    gamma,
                )
                state.w[:groups] = w_nxt[:groups]
                state.counters.aggregate_backups += groups
                state.t_sa += 1

            if observer is not None:
                observer(t, phase, state)

        if iterations and schedule.phase_at(iterations - 1) is Phase.AGGREGATE:
            lift_into(state.part, state.w, state.v)
            state.counters.lift_ops += mdp.num_states

    return AdaptiveResult(
        v=state.v,
        iterations=iterations,
        t_sa=state.t_sa,
        counters=state.counters,
        wall_ns=elapsed.wall_ns,
    )
