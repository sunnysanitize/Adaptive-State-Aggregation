import numba
import numpy as np

from .backup import backup_direct
from .mdp import TabularMdp, unpack
from .norms import max_abs_diff, max_norm
from .types import INDEX, VALUE, IndexArray, ValueArray

_NO_GROUPS = np.empty(0, dtype=INDEX)


@numba.njit
def _greedy(sa_begin, succ_begin, succ_state, succ_prob, cost, v, out, group_of, gamma):

    for s in range(out.shape[0]):
        out[s] = backup_direct(
            sa_begin, succ_begin, succ_state, succ_prob, cost, s, v, group_of, gamma
        )[1]


@numba.njit
def _evaluate_sweep(
    sa_begin, succ_begin, succ_state, succ_prob, cost, policy, v, out, gamma
):
    for s in range(out.shape[0]):
        pair = sa_begin[s] + policy[s]

        expectation = 0.0
        for k in range(succ_begin[pair], succ_begin[pair + 1]):
            expectation += succ_prob[k] * v[succ_state[k]]

        out[s] = cost[pair] + gamma * expectation


def greedy_policy(mdp: TabularMdp, v: ValueArray, gamma: float) -> IndexArray:
    out = np.empty(mdp.num_states, dtype=INDEX)
    _greedy(*unpack(mdp), v, out, _NO_GROUPS, gamma)
    return out


def policy_value(
    mdp: TabularMdp,
    policy: IndexArray,
    gamma: float,
    tol: float = 1e-10,
    max_iterations: int = 100_000,
) -> ValueArray:
    arrays = unpack(mdp)
    v = np.zeros(mdp.num_states, dtype=VALUE)
    nxt = np.empty_like(v)

    for _ in range(max_iterations):
        _evaluate_sweep(*arrays, policy, v, nxt, gamma)
        delta = max_abs_diff(nxt, v)
        v, nxt = nxt, v

        if delta < tol:
            return v

    raise RuntimeError(
        f"policy evaluation did not converge in {max_iterations} sweeps: "
        f"last sup-norm change {delta:.3e}, tol {tol:.3e}"
    )


def policy_loss(
    mdp: TabularMdp,
    v: ValueArray,
    v_star: ValueArray,
    gamma: float,
    tol: float = 1e-10,
) -> float:
    greedy = policy_value(mdp, greedy_policy(mdp, v, gamma), gamma, tol)
    return float(max_norm(greedy - v_star))
