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


# One source, compiled twice: `numba.prange` is `range` when parallel=False, so
# the two forms cannot drift. Every state writes its own `out[s]` and reads only
# `v`, so splitting the loop changes no state's arithmetic -- which is what lets
# Gate 4 demand exact equality rather than a tolerance.
def make_sweep(parallel):

    @numba.njit(parallel=parallel)
    def sweep(sa_begin, succ_begin, succ_state, succ_prob, cost, v, out, group_of, gamma):

        for s in numba.prange(out.shape[0]):
            out[s] = backup_direct(
                sa_begin, succ_begin, succ_state, succ_prob, cost, s, v, group_of, gamma
            )[0]

    return sweep


_sweep = make_sweep(False)
_sweep_parallel = make_sweep(True)


def bellman(mdp: TabularMdp, v: ValueArray, gamma: float) -> ValueArray:
    out = np.empty_like(v)
    _sweep(*unpack(mdp), v, out, _NO_GROUPS, gamma)
    return out


def value_iteration(
    mdp: TabularMdp,
    gamma: float,
    tol: float = 1e-10,
    max_iterations: int = 100_000,
    parallel: bool = False,
) -> VIResult:
    arrays = unpack(mdp)
    sweep = _sweep_parallel if parallel else _sweep
    v = np.zeros(mdp.num_states, dtype=VALUE)
    nxt = np.empty_like(v)
    counters = Counters()

    sweep(*arrays, v, nxt, _NO_GROUPS, gamma)
    max_abs_diff(v, nxt)

    with timed() as elapsed:
        # noqa B007: `iterations` is read after the loop, which the rule does
        # not model -- taking its suggested rename would break the return.
        for iterations in range(1, max_iterations + 1):  # noqa: B007
            sweep(*arrays, v, nxt, _NO_GROUPS, gamma)
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
