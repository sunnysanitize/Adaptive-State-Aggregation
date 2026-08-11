import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numba
import numpy as np

from .backup import backup_lifted
from .counters import Counters, PhaseTimes
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
    wall_ns: int = 0
    counters: Counters = field(default_factory=Counters)
    phase_times: PhaseTimes | None = None

EpsilonPolicy = Callable[[AdaptiveState], float]


@dataclass(frozen=True)
class FixedEpsilon:

    value: float

    def __call__(self, state: AdaptiveState) -> float:
        return self.value


@dataclass(frozen=True)
class AdaptiveResult:

    v: ValueArray
    iterations: int
    t_sa: int
    counters: Counters
    wall_ns: int
    phase_times: PhaseTimes | None = None


@contextmanager
def _clocked(state: AdaptiveState) -> Iterator[None]:

    with timed() as elapsed:
        yield
    state.wall_ns += elapsed.wall_ns


# One source, compiled twice. `numba.prange` is `range` when parallel=False, so
# the serial and threaded kernels cannot drift apart the way two hand-written
# copies would. Every group writes its own `out[j]` and reads only `w`, so the
# loop carries no dependency and each group's arithmetic is unchanged -- which
# is what makes exact equality between the two a reasonable thing to demand.
#
# `draws` arrives already generated. That is deliberate: drawing inside the
# kernel would let thread scheduling decide which states get sampled, and the
# two kernels would then be running different algorithms rather than the same
# one at different widths.
def make_aggregate_sweep(parallel):

    @numba.njit(parallel=parallel)
    def aggregate_sweep(
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

        for j in numba.prange(num_groups):
            lo = offset[j]
            s = members[lo + int(draws[j] * (offset[j + 1] - lo))]
            q = backup_lifted(
                sa_begin, succ_begin, succ_state, succ_prob, cost, s, w, group_of, gamma
            )[0]
            out[j] = (1.0 - alpha) * w[j] + alpha * q

    return aggregate_sweep


_aggregate_sweep = make_aggregate_sweep(False)
_aggregate_sweep_parallel = make_aggregate_sweep(True)


def _aggregate_into(
    arrays: ModelArrays,
    part: Partition,
    w: ValueArray,
    out: ValueArray,
    draws: ValueArray,
    alpha: float,
    gamma: float,
) -> None:

    _aggregate_sweep(
        *arrays,
        w,
        out,
        part.group_of,
        part.members,
        part.offset,
        part.num_groups,
        draws,
        alpha,
        gamma,
    )


# Warm up Numba-compiled kernels so JIT compilation cost is excluded from timed runs.
def _warm(arrays: ModelArrays, num_states: int, gamma: float) -> None:

    v = np.zeros(num_states, dtype=VALUE)
    out = np.empty(num_states, dtype=VALUE)
    _sweep(*arrays, v, out, _NO_GROUPS, gamma)
    span_seminorm(out - v)

    part = allocate(num_states, 1)
    spread = np.arange(num_states, dtype=VALUE)
    rebin_by_value(spread, float(max(num_states, 1)), 1, part)

    w = np.zeros(1, dtype=VALUE)
    w_out = np.empty(1, dtype=VALUE)
    _aggregate_into(arrays, part, w, w_out, np.zeros(1, dtype=VALUE), 1.0, gamma)
    lift_into(part, w, out)


@dataclass
class _AdaptiveLoop:

    arrays: ModelArrays
    num_states: int
    gamma: float
    schedule: AlternatingSchedule
    epsilon: EpsilonPolicy
    alpha: Callable[[int], float]
    sampling_rng: np.random.Generator
    max_groups: int
    state: AdaptiveState
    v_nxt: ValueArray
    w_nxt: ValueArray

    def step(self, t: int) -> Phase:

        phase = self.schedule.phase_at(t)
        entry = self.schedule.is_entry(t)

        with _clocked(self.state):
            if phase is Phase.GLOBAL:
                if entry:
                    self.run(self.lift, "lift_ns")
                self.run(self.global_sweep, "global_ns")
            else:
                if entry:
                    self.run(self.rebin, "rebin_ns")
                self.run(self.aggregate_sweep, "aggregate_ns")

        return phase

    # Timed at the call site, not inside each method, so the four buckets
    # partition the same region `_clocked` measures and cannot double-count.
    # Off by default: the headline time-to-target runs should carry no
    # instrumentation the comparison does not need.
    def run(self, work: Callable[[], None], bucket: str) -> None:
        times = self.state.phase_times
        if times is None:
            work()
            return

        start = time.perf_counter_ns()
        work()
        setattr(times, bucket, getattr(times, bucket) + time.perf_counter_ns() - start)

    def lift(self) -> None:
        state = self.state
        if state.part.num_groups == 0:
            return

        lift_into(state.part, state.w, state.v)
        state.counters.lift_ops += self.num_states

    def global_sweep(self) -> None:

        state = self.state

        _sweep(*self.arrays, state.v, self.v_nxt, _NO_GROUPS, self.gamma)
        state.counters.global_backups += self.num_states
        state.residual_span = span_seminorm(self.v_nxt - state.v)
        state.v, self.v_nxt = self.v_nxt, state.v

    def rebin(self) -> None:
        state = self.state

        rebin_by_value(state.v, self.epsilon(state), self.max_groups, state.part)
        state.counters.rebin_ops += self.num_states
        groups = state.part.num_groups
        state.w[:groups] = state.part.centers[:groups]

    def aggregate_sweep(self) -> None:

        state = self.state
        groups = state.part.num_groups

        _aggregate_into(
            self.arrays,
            state.part,
            state.w,
            self.w_nxt,
            self.sampling_rng.random(groups),
            self.alpha(state.t_sa),
            self.gamma,
        )

        state.w[:groups] = self.w_nxt[:groups]
        state.counters.aggregate_backups += groups
        state.t_sa += 1


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
    phase_timing: bool = False,
) -> AdaptiveResult:

    arrays = unpack(mdp)
    state = AdaptiveState(
        v=np.zeros(mdp.num_states, dtype=VALUE),
        w=np.zeros(min(max_groups, mdp.num_states), dtype=VALUE),
        part=allocate(mdp.num_states, max_groups),
        phase_times=PhaseTimes() if phase_timing else None,
    )
    loop = _AdaptiveLoop(
        arrays=arrays,
        num_states=mdp.num_states,
        gamma=gamma,
        schedule=schedule,
        epsilon=epsilon,
        alpha=alpha,
        sampling_rng=sampling_rng,
        max_groups=max_groups,
        state=state,
        v_nxt=np.empty_like(state.v),
        w_nxt=np.empty_like(state.w),
    )

    _warm(arrays, mdp.num_states, gamma)

    for t in range(iterations):
        phase = loop.step(t)

        if observer is not None:
            observer(t, phase, state)

    if iterations and schedule.phase_at(iterations - 1) is Phase.AGGREGATE:
        with _clocked(state):
            loop.run(loop.lift, "lift_ns")

    return AdaptiveResult(v=state.v, iterations=iterations, t_sa=state.t_sa,
                          counters=state.counters, wall_ns=state.wall_ns,
                          phase_times=state.phase_times)
