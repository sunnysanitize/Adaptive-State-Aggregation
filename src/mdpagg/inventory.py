import numpy as np

from .types import INDEX, IndexArray

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
