import numpy as np

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
