"""Gate 4: threading must not change the answer, at any thread count.

The study compares two solvers by wall-clock. That comparison means nothing
unless each threaded kernel computes what its serial form computed -- a
"speedup" that quietly changes the arithmetic is measuring a different
algorithm, not a faster one.

Exact equality is the right bar here, not a tolerance. Every kernel below
splits a loop whose iterations write distinct outputs and read only inputs, so
no iteration's arithmetic changes and no summation is reassociated. If a
comparison ever needs `approx` to pass, the split is wrong.

Same mode-scoping as the other exactness gates: `prange` degrades to `range`
under `NUMBA_DISABLE_JIT=1`, which would make every assertion here trivially
true rather than meaningfully true.
"""

import numba
import numpy as np
import pytest
from conftest import requires_jit

from mdpagg.adaptive import (
    AlternatingSchedule,
    FixedEpsilon,
    _aggregate_into,
    _aggregate_sweep,
    _aggregate_sweep_parallel,
    run_adaptive,
)
from mdpagg.maze import make_standard_maze
from mdpagg.mdp import unpack
from mdpagg.partition import allocate, lift_into, lift_into_parallel, rebin_by_value
from mdpagg.rng import streams
from mdpagg.types import VALUE
from mdpagg.vi import _NO_GROUPS, _sweep, _sweep_parallel, value_iteration

GAMMA = 0.95
SEED = 0

# The preregistered ladder, clamped to what this machine will actually give.
# Asking for more threads than Numba was configured with raises, and a gate
# that silently tested three of five thread counts would be worth little.
LADDER = (1, 2, 4, 8, 10)
THREADS = tuple(p for p in LADDER if p <= numba.config.NUMBA_NUM_THREADS)


@pytest.fixture(scope="module")
def mdp():
    return make_standard_maze((100, 100), 0.92, seed=SEED)


@pytest.fixture(autouse=True)
def restore_threads():
    yield
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)


# A vector with structure rather than zeros: a constant V makes every backup
# pick the same action, so a threaded kernel could mis-slice its range and
# still agree.
def values(n):
    return (np.arange(n, dtype=VALUE) % 97) * 0.37


@requires_jit
@pytest.mark.parametrize("threads", THREADS)
def test_parallel_global_sweep_matches_serial_bitwise(mdp, threads):
    numba.set_num_threads(threads)

    arrays = unpack(mdp)
    v = values(mdp.num_states)
    serial = np.empty(mdp.num_states, dtype=VALUE)
    threaded = np.empty(mdp.num_states, dtype=VALUE)

    _sweep(*arrays, v, serial, _NO_GROUPS, GAMMA)
    _sweep_parallel(*arrays, v, threaded, _NO_GROUPS, GAMMA)

    assert np.array_equal(serial, threaded)


def binned(mdp, eps=0.5):
    part = allocate(mdp.num_states, 4096)
    rebin_by_value(values(mdp.num_states), eps, 4096, part)
    return part


@requires_jit
@pytest.mark.parametrize("threads", THREADS)
def test_parallel_lift_matches_serial_bitwise(mdp, threads):
    numba.set_num_threads(threads)

    part = binned(mdp)
    w = (np.arange(part.num_groups, dtype=VALUE) + 1) * 1.7
    serial = np.empty(mdp.num_states, dtype=VALUE)
    threaded = np.empty(mdp.num_states, dtype=VALUE)

    lift_into(part, w, serial)
    lift_into_parallel(part, w, threaded)

    assert np.array_equal(serial, threaded)


def solve(mdp, parallel, threads=None, iterations=30):
    return run_adaptive(
        mdp,
        GAMMA,
        iterations,
        AlternatingSchedule(global_len=2, agg_len=5),
        FixedEpsilon(0.5),
        streams(SEED).sampling,
        max_groups=4096,
        parallel=parallel,
        threads=threads,
    )


@requires_jit
@pytest.mark.parametrize("threads", THREADS)
def test_threaded_run_reaches_the_same_vector_bitwise(mdp, threads):
    """The end-to-end claim. Kernel-level equality is necessary, not sufficient.

    A solver can hold every kernel exact and still diverge: pick the threaded
    aggregate kernel on one path and the serial one on the other after the
    sampling stream has moved, and the two arms drift apart while every
    individual kernel still agrees with its twin.
    """
    serial = solve(mdp, parallel=False)
    threaded = solve(mdp, parallel=True, threads=threads)

    assert np.array_equal(serial.v, threaded.v)
    assert serial.t_sa == threaded.t_sa
    assert serial.counters.billed == threaded.counters.billed
    assert serial.counters.actual == threaded.counters.actual


@requires_jit
@pytest.mark.parametrize("threads", THREADS)
def test_requested_thread_count_is_recorded_and_honored(mdp, threads):
    """A run that silently used a machine default would look identical."""
    result = solve(mdp, parallel=True, threads=threads)

    assert result.threads_requested == threads
    assert result.threads_observed == threads


@requires_jit
def test_serial_run_records_one_thread(mdp):
    result = solve(mdp, parallel=False)

    assert result.parallel is False
    assert result.threads_observed == 1


@requires_jit
@pytest.mark.parametrize("threads", THREADS)
def test_parallel_aggregate_sweep_matches_serial_bitwise(mdp, threads):
    """Regression, not a gate this test drove.

    The threaded aggregate kernel predates this file -- it was written for
    `scripts/aggregate_grain.py`, which compared it against the serial form at
    15 group counts by 5 thread counts and found zero mismatches. This pins
    that result so a later edit cannot quietly undo it.
    """
    numba.set_num_threads(threads)

    part = binned(mdp)
    groups = part.num_groups
    arrays = unpack(mdp)
    w = (np.arange(groups, dtype=VALUE) + 1) * 0.9
    draws = np.linspace(0.0, 0.999, groups).astype(VALUE)
    serial = np.empty(groups, dtype=VALUE)
    threaded = np.empty(groups, dtype=VALUE)

    _aggregate_into(arrays, part, w, serial, draws, 0.5, GAMMA, kernel=_aggregate_sweep)
    _aggregate_into(
        arrays, part, w, threaded, draws, 0.5, GAMMA, kernel=_aggregate_sweep_parallel
    )

    assert np.array_equal(serial, threaded)


@requires_jit
@pytest.mark.parametrize("threads", THREADS)
def test_threaded_value_iteration_matches_serial_bitwise(mdp, threads):
    """`value_iteration` is what solves ground truth, so it needs the option too.

    It also stops on a measured quantity, so a threaded form that changed the
    arithmetic would change the iteration count and not just the vector.
    """
    numba.set_num_threads(threads)

    serial = value_iteration(mdp, GAMMA, tol=1e-10)
    threaded = value_iteration(mdp, GAMMA, tol=1e-10, parallel=True)

    assert np.array_equal(serial.v, threaded.v)
    assert serial.iterations == threaded.iterations
    assert serial.backups == threaded.backups


def test_rebin_leaves_members_ascending_within_every_group():
    """The invariant any parallel rebin would have to preserve, pinned now.

    The aggregate step samples a state by its *position* within a group:
    `members[offset[j] + int(draw * size)]`. So member order is not an
    implementation detail, it decides which state gets backed up. A parallel
    scatter that let threads interleave their writes would still produce a
    valid partition, still pass every set-based check, and silently run a
    different algorithm -- no numeric tolerance could detect it.

    Rebinning is therefore left serial, and this test is what would catch a
    future change that forgets why.
    """
    mdp = make_standard_maze((60, 60), 0.92, seed=SEED)
    part = binned(mdp)

    for j in range(part.num_groups):
        members = part.group(j)
        assert np.all(np.diff(members) > 0), f"group {j} is not ascending"
