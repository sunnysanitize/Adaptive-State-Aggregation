"""Task 3.2, Gate 2: with no aggregate phase, Algorithm 3 *is* value iteration.

Set `|Ai| = 0` and every branch that makes the adaptive layer adaptive is
switched off: no rebinning, no sampling, no lifting, no step size. What is left
is a global sweep repeated, which is exactly what `value_iteration` does. So
the two must not merely agree -- they must produce the *same floats*, on every
iterate, because they are running the same kernel over the same numbers in the
same order.

Hence plain `==`. An approximate match would mean the loop has picked up an
extra arithmetic step somewhere -- a stale buffer, a swap in the wrong place, a
sweep that reads `V` while writing it -- and every result downstream of here
would inherit it. This is a hard gate: if it fails, bisect against the Phase 1
anchors rather than loosening the comparison.

Same mode-scoping as Gate 1. Numba may contract a multiply-add into a single
FMA where CPython will not, so the exactness claim is scoped to `make test` and
the gate skips itself under `make debug` with the reason printed.

Two references, deliberately. Repeated `bellman` supplies the per-iterate
sequence, which `value_iteration` does not expose -- it returns only its final
vector. `value_iteration` itself is then pinned on that final vector, so the
gate holds against the real function and not only against a stand-in built from
the same parts.

The schedule assertion lives here rather than beside the loop because it is the
premise the gate rests on: if `phase_at` said AGGREGATE anywhere under
`agg_len = 0`, the comparison would be vacuous rather than failing.
"""

import numpy as np
import pytest
from conftest import requires_jit
from fixtures import ALL_FIXTURES

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.maze import make_standard_maze
from mdpagg.rng import streams
from mdpagg.types import VALUE, Phase
from mdpagg.vi import bellman, value_iteration

GLOBAL_LEN = 2
ITERATIONS = 40
MAX_GROUPS = 64
SEED = 0

# The hand-computed fixtures carry their own gamma; the maze is the first model
# here with enough transition structure for a mis-sliced CSR read to survive
# into the iterates rather than crashing on the first sweep.
MODELS = [(f.name, f.mdp, f.gamma) for f in (make() for make in ALL_FIXTURES)]
MODELS.append(("maze_20x20", make_standard_maze((20, 20), 0.92, seed=SEED), 0.95))

IDS = [name for name, _, _ in MODELS]


def pure_global() -> AlternatingSchedule:
    return AlternatingSchedule(global_len=GLOBAL_LEN, agg_len=0)


def run(mdp, gamma, iterations, observer=None):
    return run_adaptive(
        mdp,
        gamma,
        iterations,
        pure_global(),
        FixedEpsilon(0.5),
        streams(SEED).sampling,
        max_groups=MAX_GROUPS,
        observer=observer,
    )


def test_phase_at_matches_a_hand_written_sequence():
    """|Bi| = 2, |Ai| = 5 -> a cycle of 7, the first two of each global.

    Written out rather than computed, because a schedule bug reproduced in the
    expectation would agree with itself. The off-by-one this catches is whether
    the cycle opens on the global block, which Algorithm 3 requires -- the
    horizon is divided B1, A1, B2, A2, and there is no V to bin until a global
    sweep has produced one.
    """
    schedule = AlternatingSchedule(global_len=2, agg_len=5)
    g, a = Phase.GLOBAL, Phase.AGGREGATE

    assert [schedule.phase_at(t) for t in range(30)] == [g, g, a, a, a, a, a] * 4 + [g, g]
    assert [t for t in range(30) if schedule.is_entry(t)] == [0, 2, 7, 9, 14, 16, 21, 23, 28]


def test_zero_length_aggregate_is_global_at_every_iteration():
    schedule = pure_global()

    assert all(schedule.phase_at(t) is Phase.GLOBAL for t in range(ITERATIONS))


@pytest.mark.parametrize("name, mdp, gamma", MODELS, ids=IDS)
def test_zero_length_aggregate_does_no_aggregate_work(name, mdp, gamma):
    """The gate is only meaningful if the aggregate branch really never ran.

    `t_sa` is the tell. It starts at 1 and increments once per aggregate
    iteration, so a `t_sa` above 1 means the loop took the other branch and the
    bitwise match below would be saying something other than what it claims.
    """
    result = run(mdp, gamma, ITERATIONS)

    assert result.t_sa == 1
    assert result.counters.aggregate_backups == 0
    assert result.counters.rebin_ops == 0
    assert result.counters.lift_ops == 0
    assert result.counters.global_backups == ITERATIONS * mdp.num_states


def test_t_sa_counts_aggregate_iterations_and_never_resets():
    """The one hazard `agg_len = 0` cannot reach, so it gets its own run.

    `t_sa` is the step-size clock: `alpha = 1/sqrt(t_sa)`. Resetting it at each
    aggregate entry would restore a step size of 1 every seventh iteration, so
    the run would keep discarding what it had learned and the error curve would
    flatten early -- a plausible-looking figure with nothing wrong on its face.
    It lives on `AdaptiveState` precisely so a refactor cannot lose it in a
    loop variable, and every mutation that resets it survives the gate above.
    """
    name, mdp, gamma = MODELS[-1]
    seen = []
    run_adaptive(
        mdp,
        gamma,
        30,
        AlternatingSchedule(global_len=2, agg_len=5),
        FixedEpsilon(0.5),
        streams(SEED).sampling,
        max_groups=MAX_GROUPS,
        observer=lambda t, phase, state: seen.append((phase, state.t_sa)),
    )

    aggregate = [t_sa for phase, t_sa in seen if phase is Phase.AGGREGATE]

    assert len(aggregate) == 20
    assert aggregate == list(range(2, 22))


@requires_jit
@pytest.mark.parametrize("name, mdp, gamma", MODELS, ids=IDS)
def test_zero_length_aggregate_matches_every_bellman_iterate_bitwise(name, mdp, gamma):
    iterates = []
    run(mdp, gamma, ITERATIONS, observer=lambda t, phase, state: iterates.append(state.v.copy()))

    assert len(iterates) == ITERATIONS

    v = np.zeros(mdp.num_states, dtype=VALUE)
    for i, got in enumerate(iterates):
        v = bellman(mdp, v, gamma)
        assert np.array_equal(v, got), f"{name} diverges at iterate {i}"


@requires_jit
@pytest.mark.parametrize("name, mdp, gamma", MODELS, ids=IDS)
def test_zero_length_aggregate_reaches_the_same_final_vector(name, mdp, gamma):
    """Pinned to `value_iteration` itself, run for the iteration count it chose.

    Its untimed warm-up sweep leaves `V` at zero, so after `n` loop passes both
    it and the adaptive loop have applied the Bellman operator exactly `n`
    times to the zero vector. Anything else here means one of the two is
    dropping or repeating a sweep.
    """
    exact = value_iteration(mdp, gamma, tol=1e-12)
    result = run(mdp, gamma, exact.iterations)

    assert np.array_equal(result.v, exact.v), name
    assert result.counters.global_backups == exact.backups
