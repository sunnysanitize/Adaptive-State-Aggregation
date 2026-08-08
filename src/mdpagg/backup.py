"""

Nothing else in the codebase ever computes a backup. It is only called. 

"""

import numba
import numpy as np


@numba.njit(inline="always")
def _lookup_direct(values, group_of, s):
    return values[s]


@numba.njit(inline="always")
def _lookup_lifted(values, group_of, s):
    return values[group_of[s]]


def make_backup(lookup):

    @numba.njit(cache=True)
    def backup(
        sa_begin,
        succ_begin,
        succ_state,
        succ_prob,
        cost,
        s,
        values,
        group_of,
        gamma,
    ):
        best = np.inf
        best_action = -1

        first_pair = sa_begin[s]
        for a in range(sa_begin[s + 1] - first_pair):
            pair = first_pair + a

            expectation = 0.0
            for k in range(succ_begin[pair], succ_begin[pair + 1]):
                expectation += succ_prob[k] * lookup(values, group_of, succ_state[k])

            q = cost[pair] + gamma * expectation

            if q < best:
                best = q
                best_action = a

        return best, best_action

    return backup


backup_direct = make_backup(_lookup_direct)
backup_lifted = make_backup(_lookup_lifted)
