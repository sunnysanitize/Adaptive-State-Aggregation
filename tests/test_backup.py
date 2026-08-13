"""The two specializations agree bitwise.

Under the identity partition -- one group per state, `w = v` -- the lifted
lookup `w[group_of[s']]` reads the same slot as the direct lookup `v[s']`. Same
operations in the same order on the same numbers, so the two kernels must
return the *same float*, not a nearby one.

Hence plain `==`, not `pytest.approx`. Approximate agreement would mean the
specializations have drifted apart somewhere, and the whole design -- one
template, two inlined lookups, one backup in the codebase -- rests on their not
having. `test_adaptive` pins the adaptive layer to VI bit for bit, which is only
reachable because both call this same kernel.
"""

import numpy as np
import pytest
from conftest import requires_jit
from fixtures import ALL_FIXTURES

from mdpagg.backup import backup_direct, backup_lifted
from mdpagg.mdp import unpack
from mdpagg.types import INDEX, VALUE


def value_vectors(num_states: int, v_star):
    """Three points to test at, none of them special to one kernel.

    V0 = 0 is the literal first iterate of every experiment; V* is the fixed
    point, where the min is least discriminating; the seeded random vector is
    neither, and is where a wrong-action tie-break would show up.
    """
    rng = np.random.default_rng(20260808)
    return [
        ("zeros", np.zeros(num_states, dtype=VALUE)),
        ("v_star", v_star.astype(VALUE)),
        ("random", rng.uniform(-50.0, 50.0, num_states).astype(VALUE)),
    ]


@requires_jit
@pytest.mark.parametrize("make", ALL_FIXTURES, ids=lambda f: f.__name__)
def test_identity_partition_makes_the_two_kernels_agree_bitwise(make):
    f = make()
    m = f.mdp
    arrays = unpack(m)

    # The identity partition: K = |S|, group j holds state j, w = v.
    ident = np.arange(m.num_states, dtype=INDEX)

    for label, v in value_vectors(m.num_states, f.v_star):
        for s in range(m.num_states):
            direct = backup_direct(*arrays, s, v, ident, f.gamma)
            lifted = backup_lifted(*arrays, s, v, ident, f.gamma)
            assert direct == lifted, f"{f.name}, v={label}, s={s}"


@pytest.mark.parametrize("make", ALL_FIXTURES, ids=lambda f: f.__name__)
def test_lifted_reads_values_through_the_partition(make):
    """The one thing the identity partition cannot check.

    `group_of[s] == s` there, so a lifted lookup that ignored `group_of`
    entirely and read `values[s]` would pass the test above on every state.
    That is the single property distinguishing the lifted kernel, and the
    would not catch it either -- with `agg_len = 0` the aggregate phase never
    runs. It would surface later as "aggregation is broken", a long way from
    its cause.

    So: collapse every state into one group. Successor values are then all
    `c` regardless of which successor, and the backup must reduce to
    `min_a [cost(s, a)] + gamma * c`. A kernel reading `values[s]` would index
    a length-1 array and fail loudly instead.

    Approximate, not bitwise: the expectation is a sum of `p_k * c` and need
    not land on `c` exactly. The bitwise claim is the test above; this one is
    about which slot gets read, so it holds in every run mode.
    """
    f = make()
    m = f.mdp
    arrays = unpack(m)

    c = 7.5
    one_group = np.zeros(m.num_states, dtype=INDEX)
    w = np.array([c], dtype=VALUE)

    for s in range(m.num_states):
        pairs = [m.pair_index(s, a) for a in range(m.num_actions(s))]
        costs = [float(m.cost[p]) for p in pairs]
        expected = min(costs) + f.gamma * c

        value, action = backup_lifted(*arrays, s, w, one_group, f.gamma)

        assert value == pytest.approx(expected, abs=1e-12), f"{f.name}, s={s}"
        assert costs[action] == min(costs), f"{f.name}, s={s}"
