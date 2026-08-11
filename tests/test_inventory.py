"""Tasks 5.1 and 5.2: the inventory state space, its dynamics, and the
constraint that keeps the formulation well-posed.

The state is a signed inventory vector `(q_1, ..., q_N)`, each `q_i` in
`{-Q, ..., Q}`, mixed-radix encoded to a flat index. Two things about the
encoding are worth pinning before anything is built on top.

**Why a round-trip alone is not enough.** `encode(decode(s)) == s` holds for
*any* bijection, including one that is consistently off by one and enumerates
`{-Q+1, ..., Q+1}`. The round-trip would never notice, and the error would
surface at 5.2 as corner states that clip on one side only. So the second test
checks the grid itself against `itertools.product`, which shares no code with
the implementation, and pins the row order too: 5.2's clipping arithmetic reads
the layout, so a switch to Fortran order would silently relabel every state.

**The action constraint.** `|A| = 5` quote-aggressiveness levels, the same five
in every state, *independent of N*. This is the load-bearing design decision of
the whole inventory formulation: if actions ever became per-asset the space
would be `5^N`, growing at the same exponential rate as the state space, and
aggregation -- which compresses states, not actions -- would buy nothing. The
study needs an exponential state space against a fixed, small action space.

It is a module constant precisely because a constant cannot depend on `N`.
Asserting `NUM_ACTIONS == 5` on its own would be a tautology, so the check that
earns its place runs against the *built* MDP at `N = 1, 2, 3`: the state count
moves 5, 25, 125 while every state keeps exactly five actions. A drift to
per-asset levels would read 5, 25, 125 on both.

**Dynamics (5.2).** One fill per period, so `2N+1` successors per pair: no fill,
or asset `i` fills on the bid (`+1`) or the ask (`-1`). Fills are exogenous --
the kernel depends on the action and on clipping, not on the inventory, which
enters through cost at 5.3.

Clipping at `+-Q` is the whole difficulty, because a clipped branch collides
with the no-fill branch and the two probabilities must add. Collided successors
are left as duplicate CSR entries at uniform width `2N+1` rather than compacted;
`backup.py` accumulates `p * V[s']` over the slice, so duplicates sum to exactly
the merged distribution, and the uniform width is what makes the `N = 4`
instance the 105 MB the plan predicts. `_row` therefore sums duplicates, which
keeps every assertion here true of either representation.

Row sums get no test: `build()` validates them and names the offending state,
and it does reject the lossy version -- checked by construction, not assumed.
What the corner tests add beyond that is the *merge*. Reflecting at the boundary
instead of clipping conserves mass and stays in range, so `build()` accepts it
and every structural test here still passes; only the two corner tests catch it.
"""

import itertools

import numpy as np
import pytest

from mdpagg.inventory import NUM_ACTIONS, decode, encode, num_states, transitions
from mdpagg.mdp import build
from mdpagg.types import INDEX, VALUE

N = 3
Q = 10

# Total fill probability per aggressiveness level. Distinct values so a test
# cannot pass by reading the wrong action.
FILL = np.array([0.1, 0.2, 0.4, 0.6, 0.8])


def _state(inventory: tuple[int, ...], q_max: int) -> int:
    return int(encode(np.array([inventory]), q_max)[0])


def _row(m, s: int, a: int, n: int) -> np.ndarray:
    """The successor distribution as a dense row over states.

    Sums duplicate successor indices, which is what the backup kernel does, so
    these assertions hold whether or not collided successors are compacted.
    """
    pair = m.pair_index(s, a)
    row = np.zeros(n)
    np.add.at(row, m.successors(pair), m.probabilities(pair))
    return row


def _mdp(num_assets: int, q_max: int):
    """Dynamics only -- costs arrive at 5.3, so this builds against zero cost.
    `build()` still validates row sums, which is where a lost-mass clipping bug
    dies without a test of its own."""
    sa_begin, succ_begin, succ_state, succ_prob = transitions(num_assets, q_max, FILL)
    return build(sa_begin, succ_begin, succ_state, succ_prob, np.zeros(int(sa_begin[-1]), dtype=VALUE))


def test_every_state_round_trips_at_the_study_size():
    """All 9,261 states at N=3, Q=10 -- exhaustive, and cheap enough to stay so."""
    states = np.arange(num_states(N, Q), dtype=INDEX)

    assert np.array_equal(encode(decode(states, N, Q), Q), states)


def test_decoding_enumerates_the_signed_inventory_grid():
    """The range and the row order, against a reference that shares no code."""
    expected = np.array(list(itertools.product(range(-Q, Q + 1), repeat=N)))

    assert np.array_equal(decode(np.arange(num_states(N, Q), dtype=INDEX), N, Q), expected)


def test_indices_stay_int32():
    """`np.ravel_multi_index` returns int64 and every index array downstream is
    INDEX. An int64 that reads correctly here doubles the memory of the N=4
    instance at 5.6 and nothing else in the toolchain would ever mention it."""
    states = np.arange(num_states(N, Q), dtype=INDEX)

    assert decode(states, N, Q).dtype == INDEX
    assert encode(decode(states, N, Q), Q).dtype == INDEX


@pytest.mark.parametrize("num_assets", [1, 2, 3])
def test_every_state_has_five_actions_whatever_the_number_of_assets(num_assets):
    """The 5.1 constraint, in the only form that can actually fail. The state
    count here runs 5, 25, 125 while the action count does not move; a drift to
    per-asset levels would read 5, 25, 125 as well."""
    m = _mdp(num_assets, 2)

    assert m.num_states == num_states(num_assets, 2)
    assert {m.num_actions(s) for s in range(m.num_states)} == {NUM_ACTIONS}


def test_a_bid_fill_at_the_top_corner_clips_back_onto_itself():
    """One asset, Q=2, hand-checked. At `q = +Q` a bid fill would take the book
    to `+Q+1`; it clips, so that branch collides with the no-fill branch and the
    two probabilities add. Getting this wrong loses `f/2` of mass at exactly the
    states clipping affects, while every interior row stays perfect.
    """
    m, n, a, f = _mdp(1, 2), num_states(1, 2), 2, FILL[2]
    top, below = _state((2,), 2), _state((1,), 2)

    row = _row(m, top, a, n)

    assert row[top] == pytest.approx(1.0 - f + f / 2)
    assert row[below] == pytest.approx(f / 2)
    assert row.sum() == pytest.approx(1.0)


def test_an_ask_fill_at_the_bottom_corner_clips_back_onto_itself():
    """The mirror of the above. Both ends, because a clip written as `min` alone
    is correct at one boundary and absent at the other."""
    m, n, a, f = _mdp(1, 2), num_states(1, 2), 3, FILL[3]
    bottom, above = _state((-2,), 2), _state((-1,), 2)

    row = _row(m, bottom, a, n)

    assert row[bottom] == pytest.approx(1.0 - f + f / 2)
    assert row[above] == pytest.approx(f / 2)
    assert row.sum() == pytest.approx(1.0)


def test_an_interior_state_reaches_every_neighbour_without_clipping():
    """The control the corner tests need: away from the boundary all `2N+1`
    branches land on distinct states and no probability is pooled. Without it,
    an implementation that clipped everywhere onto itself would still satisfy
    both corner tests."""
    m, n, a, f = _mdp(2, 3), num_states(2, 3), 1, FILL[1]
    here = _state((0, 0), 3)

    row = _row(m, here, a, n)

    assert row[here] == pytest.approx(1.0 - f)
    for neighbour in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        assert row[_state(neighbour, 3)] == pytest.approx(f / 4), neighbour
    assert np.count_nonzero(row) == 2 * 2 + 1


def test_at_most_one_asset_fills_per_period():
    """The modelling assumption the `2N+1` width encodes. If two assets could
    move in one period the successor count would be `3^N` and the whole CSR
    layout would be wrong, so this is checked over every state and action rather
    than spot-checked."""
    m, q_max = _mdp(2, 3), 3
    inventories = decode(np.arange(num_states(2, 3), dtype=INDEX), 2, q_max)

    for s in range(m.num_states):
        for a in range(NUM_ACTIONS):
            moved = inventories[m.successors(m.pair_index(s, a))] - inventories[s]
            assert (np.count_nonzero(moved, axis=1) <= 1).all(), (s, a)
