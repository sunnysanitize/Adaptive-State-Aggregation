from dataclasses import dataclass

import numba
import numpy as np

from .backup import backup_direct
from .counters import Counters
from .mdp import TabularMdp, unpack
from .norms import max_abs_diff
from .timer import timed
from .types import INDEX, VALUE, ValueArray


_NO_GROUPS = np.empty(0, dtype=INDEX)


@dataclass(frozen=True)
class VIResult:

    v: ValueArray
    iterations: int
    backups: int
    wall_ns: int


@numba.njit
def _sweep(sa_begin, succ_begin, succ_state, succ_prob, cost, v, out, group_of, gamma):

    for s in range(out.shape[0]):
        out[s] = backup_direct(
            sa_begin, succ_begin, succ_state, succ_prob, cost, s, v, group_of, gamma
        )[0]


def bellman(mdp: TabularMdp, v: ValueArray, gamma: float) -> ValueArray:
    out = np.empty_like(v)
    _sweep(*unpack(mdp), v, out, _NO_GROUPS, gamma)
    return out


def value_iteration(
    mdp: TabularMdp,
    gamma: float,
    tol: float = 1e-10,
    max_iterations: int = 100_000,
) -> VIResult:
    arrays = unpack(mdp)
    v = np.zeros(mdp.num_states, dtype=VALUE)
    nxt = np.empty_like(v)
    counters = Counters()

    _sweep(*arrays, v, nxt, _NO_GROUPS, gamma)
    max_abs_diff(v, nxt)

    with timed() as elapsed:
        for iterations in range(1, max_iterations + 1):
            _sweep(*arrays, v, nxt, _NO_GROUPS, gamma)
            counters.global_backups += mdp.num_states

            delta = max_abs_diff(nxt, v)
            v, nxt = nxt, v

            if delta < tol:
                break
        else:
            raise RuntimeError(
                f"value iteration did not converge in {max_iterations} sweeps: "
                f"last sup-norm change {delta:.3e}, tol {tol:.3e}"
            )

    return VIResult(v, iterations, counters.global_backups, elapsed.wall_ns)
