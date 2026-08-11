"""Task 5.1: the inventory state space, and the constraint that keeps it well-posed.

The state is a signed inventory vector `(q_1, ..., q_N)`, each `q_i` in
`{-Q, ..., Q}`, mixed-radix encoded to a flat index. That is the whole of 5.1 --
the dynamics arrive at 5.2 -- but two things about it are worth pinning before
anything is built on top.

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

It is a module constant here precisely because a constant cannot depend on `N`.
There is nothing this file can assert about it that is not a tautology, so the
real check lives at 5.2, where `num_actions(s) == 5` is asserted against the
built MDP at two different `N` and would actually fail if the space went
per-asset.
"""

import itertools

import numpy as np

from mdpagg.inventory import NUM_ACTIONS, decode, encode, num_states
from mdpagg.types import INDEX

N = 3
Q = 10


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


def test_the_action_space_is_five_levels():
    """Guards the constant against a silent edit. That `5` does not depend on N
    is structural, not asserted -- see the module docstring, and 5.2."""
    assert NUM_ACTIONS == 5
