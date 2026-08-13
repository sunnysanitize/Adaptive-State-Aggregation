"""The CSR model and its builder.

Two of the builder's five checks get tests, because they are the two that fail
silently: a row that sums to something near-but-not 1, and an index array that
reads correctly at the wrong dtype. An out-of-range index or an action-less
state crashes loudly on first use.
"""

import numpy as np
import pytest

from mdpagg.mdp import build
from mdpagg.types import INDEX, VALUE

# The hand-built two-state model, written as (s, a) -> (cost, {s': p}).
# State 0 has two actions, state 1 has one, so the action counts differ and the
# pair-index arithmetic has something to get wrong.
TWO_STATE = {
    (0, 0): (1.0, {0: 0.7, 1: 0.3}),
    (0, 1): (2.0, {1: 1.0}),
    (1, 0): (0.5, {0: 0.4, 1: 0.6}),
}


def two_state_arrays():
    """The CSR arrays for TWO_STATE, laid out by hand."""
    sa_begin = np.array([0, 2, 3], dtype=INDEX)
    succ_begin = np.array([0, 2, 3, 5], dtype=INDEX)
    succ_state = np.array([0, 1, 1, 0, 1], dtype=INDEX)
    succ_prob = np.array([0.7, 0.3, 1.0, 0.4, 0.6], dtype=VALUE)
    cost = np.array([1.0, 2.0, 0.5], dtype=VALUE)
    return sa_begin, succ_begin, succ_state, succ_prob, cost


def test_round_trips_every_transition():
    m = build(*two_state_arrays())

    assert m.num_states == 2

    seen = {}
    for s in range(m.num_states):
        for a in range(m.num_actions(s)):
            p = m.pair_index(s, a)
            transitions = dict(
                zip(
                    m.successors(p).tolist(),
                    m.probabilities(p).tolist(),
                    strict=True,
                )
            )
            seen[(s, a)] = (m.cost[p], transitions)

    assert seen == TWO_STATE


def test_rejects_row_that_does_not_sum_to_one():
    sa_begin, succ_begin, succ_state, succ_prob, cost = two_state_arrays()
    succ_prob[3] = 0.39  # pair 2 is state 1's only action: 0.39 + 0.6 = 0.99

    with pytest.raises(ValueError, match="state 1"):
        build(sa_begin, succ_begin, succ_state, succ_prob, cost)


def test_rejects_index_array_of_wrong_dtype():
    sa_begin, succ_begin, succ_state, succ_prob, cost = two_state_arrays()
    succ_state = succ_state.astype(np.int64)  # reads correctly, doubles memory

    with pytest.raises(ValueError, match="succ_state"):
        build(sa_begin, succ_begin, succ_state, succ_prob, cost)
