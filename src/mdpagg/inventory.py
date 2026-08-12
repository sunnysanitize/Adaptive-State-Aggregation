import numpy as np

from .mdp import TabularMdp, build
from .types import INDEX, VALUE, IndexArray, ValueArray

NUM_ACTIONS = 5


def radix(q_max: int) -> int:
    return 2 * q_max + 1


def num_states(num_assets: int, q_max: int) -> int:
    return radix(q_max) ** num_assets


def decode(states: IndexArray, num_assets: int, q_max: int) -> IndexArray:
    shape = (radix(q_max),) * num_assets
    return (np.stack(np.unravel_index(states, shape), axis=-1) - q_max).astype(INDEX)


def encode(inventories: IndexArray, q_max: int) -> IndexArray:
    shape = (radix(q_max),) * inventories.shape[-1]
    return np.ravel_multi_index(tuple((inventories + q_max).T), shape).astype(INDEX)


def _deltas(num_assets: int) -> IndexArray:
    eye = np.eye(num_assets, dtype=INDEX)
    return np.concatenate((np.zeros((1, num_assets), dtype=INDEX), eye, -eye))


def _live_fractions(inventories: IndexArray, q_max: int) -> ValueArray:
    branches = 2 * inventories.shape[1]
    pinned = np.count_nonzero(np.abs(inventories) == q_max, axis=1)
    return ((branches - pinned) / branches).astype(VALUE)


def costs(
    num_assets: int,
    q_max: int,
    fill: ValueArray,
    lam: float,
    sigma: ValueArray,
    spread: ValueArray,
) -> ValueArray:
    states = np.arange(num_states(num_assets, q_max), dtype=INDEX)
    inventories = decode(states, num_assets, q_max)

    held = inventories.astype(VALUE)
    risk = lam * np.einsum("si,ij,sj->s", held, sigma, held)
    revenue = spread * _live_fractions(inventories, q_max)[:, None] * fill[None, :]

    return (risk[:, None] - revenue).reshape(-1).astype(VALUE)


def transitions(
    num_assets: int, q_max: int, fill: ValueArray
) -> tuple[IndexArray, IndexArray, IndexArray, ValueArray]:
    n = num_states(num_assets, q_max)
    width = 2 * num_assets + 1

    delta = _deltas(num_assets)
    inventories = decode(np.arange(n, dtype=INDEX), num_assets, q_max)
    reached = np.clip(inventories[:, None, :] + delta[None, :, :], -q_max, q_max)
    succ = encode(reached.reshape(-1, num_assets), q_max).reshape(n, width)

    prob = np.empty((NUM_ACTIONS, width), dtype=VALUE)
    prob[:, 0] = 1.0 - fill
    prob[:, 1:] = (fill / (2 * num_assets))[:, None]

    sa_begin = (np.arange(n + 1) * NUM_ACTIONS).astype(INDEX)
    succ_begin = (np.arange(n * NUM_ACTIONS + 1) * width).astype(INDEX)

    return (
        sa_begin,
        succ_begin,
        np.repeat(succ, NUM_ACTIONS, axis=0).reshape(-1).astype(INDEX),
        np.tile(prob, (n, 1)).reshape(-1).astype(VALUE),
    )


def make_inventory_mdp(
    num_assets: int,
    q_max: int,
    fill: ValueArray,
    lam: float,
    sigma: ValueArray,
    spread: ValueArray,
) -> TabularMdp:
    sa_begin, succ_begin, succ_state, succ_prob = transitions(num_assets, q_max, fill)
    return build(
        sa_begin,
        succ_begin,
        succ_state,
        succ_prob,
        costs(num_assets, q_max, fill, lam, sigma, spread),
    )


def equicorrelated(num_assets: int, rho: float) -> ValueArray:
    off = np.full((num_assets, num_assets), rho, dtype=VALUE)
    return off + np.eye(num_assets, dtype=VALUE) * (1.0 - rho)


def do_nothing_policy(num_assets: int, q_max: int) -> IndexArray:
    return np.zeros(num_states(num_assets, q_max), dtype=INDEX)


def immediate_cost_policy(
    num_assets: int,
    q_max: int,
    fill: ValueArray,
    lam: float,
    sigma: ValueArray,
    spread: ValueArray,
) -> IndexArray:
    cost = costs(num_assets, q_max, fill, lam, sigma, spread)
    return np.argmin(cost.reshape(-1, NUM_ACTIONS), axis=1).astype(INDEX)


def linear_hedge_policy(num_assets: int, q_max: int) -> IndexArray:
    states = np.arange(num_states(num_assets, q_max), dtype=INDEX)
    exposure = np.abs(decode(states, num_assets, q_max)).max(axis=1)
    return np.rint((NUM_ACTIONS - 1) * exposure / q_max).astype(INDEX)
