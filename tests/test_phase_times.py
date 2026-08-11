"""Phase timing decides the Amdahl argument, so mis-attribution is not cosmetic.

The parallel study's central claim rests on where the adaptive solver spends
its time. Global sweeps and lifting parallelize over states; rebinning has to
keep `members` in a stable order, because the aggregate step samples a state by
its position within a group. So rebinning's share of runtime *is* the serial
fraction, and the serial fraction sets the ceiling on any speedup the solver
can ever show. A bucket that quietly collects another phase's time would move
that ceiling and the conclusion with it, while every number still looked
plausible.

These assertions are all exact or structural -- no wall-clock thresholds -- so
they fail on a bug rather than on a busy machine. What they cannot check is
that the numbers are *useful*; that is what the measurement script reports.
"""

import pytest

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.maze import make_standard_maze
from mdpagg.rng import streams

GAMMA = 0.95
ITERATIONS = 40
MAX_GROUPS = 64
SEED = 0

BUCKETS = ("global_ns", "aggregate_ns", "rebin_ns", "lift_ns")


@pytest.fixture(scope="module")
def mdp():
    return make_standard_maze((20, 20), 0.92, seed=SEED)


def run(mdp, schedule, phase_timing):
    return run_adaptive(
        mdp,
        GAMMA,
        ITERATIONS,
        schedule,
        FixedEpsilon(0.5),
        streams(SEED).sampling,
        max_groups=MAX_GROUPS,
        phase_timing=phase_timing,
    )


def test_instrumentation_is_off_by_default(mdp):
    """The timed runs that decide the study must carry no extra timer calls."""
    result = run(mdp, AlternatingSchedule(global_len=2, agg_len=5), phase_timing=False)

    assert result.phase_times is None


def test_every_phase_is_measured_when_all_four_run(mdp):
    result = run(mdp, AlternatingSchedule(global_len=2, agg_len=5), phase_timing=True)
    times = result.phase_times

    assert times is not None
    assert all(getattr(times, bucket) > 0 for bucket in BUCKETS)


def test_no_aggregate_phase_leaves_its_buckets_at_exactly_zero(mdp):
    """`agg_len = 0` never takes the aggregate branch, so these are exact.

    An exact zero is what makes this worth asserting: if the rebin bucket were
    accumulating a slice of the global sweep, it would show up here as a small
    nonzero rather than as an obviously wrong figure in a report.
    """
    result = run(mdp, AlternatingSchedule(global_len=2, agg_len=0), phase_timing=True)
    times = result.phase_times

    assert times is not None
    assert times.aggregate_ns == 0
    assert times.rebin_ns == 0
    assert times.global_ns > 0


def test_buckets_nest_inside_the_clocked_total(mdp):
    """Each phase is timed inside the region `wall_ns` already covers.

    Exceeding it would mean a phase is counted twice, which is exactly the bug
    that would inflate one share at another's expense.
    """
    result = run(mdp, AlternatingSchedule(global_len=2, agg_len=5), phase_timing=True)
    times = result.phase_times

    assert times is not None
    assert times.total_ns <= result.wall_ns


def test_shares_are_a_partition(mdp):
    result = run(mdp, AlternatingSchedule(global_len=2, agg_len=5), phase_timing=True)
    times = result.phase_times

    assert times is not None
    assert sum(times.share().values()) == pytest.approx(1.0)
