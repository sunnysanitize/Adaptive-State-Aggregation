import numpy as np

from .mdp import TabularMdp, build
from .types import INDEX, VALUE

GOAL = 0


def _strides(dims: tuple[int, ...]) -> np.ndarray:
    strides = np.ones(len(dims), dtype=np.int64)
    for axis in range(len(dims) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * dims[axis + 1]
    return strides


def _carve(dims: tuple[int, ...], seed: int) -> np.ndarray:
    num_cells = int(np.prod(dims))
    ndim = len(dims)
    strides = _strides(dims)
    coords = np.array(np.unravel_index(np.arange(num_cells), dims))

    rng = np.random.default_rng(seed)
    open_ = np.zeros((num_cells, 2 * ndim), dtype=bool)
    visited = np.zeros(num_cells, dtype=bool)

    stack = [0]
    visited[0] = True
    candidates = np.empty(2 * ndim, dtype=np.int64)

    while stack:
        cell = stack[-1]

        n = 0
        for d in range(2 * ndim):
            axis, step = d >> 1, (d & 1) * 2 - 1
            position = coords[axis, cell] + step
            if 0 <= position < dims[axis] and not visited[cell + step * strides[axis]]:
                candidates[n] = d
                n += 1

        if n == 0:
            stack.pop()
            continue

        d = int(candidates[rng.integers(n)])
        axis, step = d >> 1, (d & 1) * 2 - 1
        neighbour = int(cell + step * strides[axis])

        open_[cell, d] = True
        open_[neighbour, d ^ 1] = True
        visited[neighbour] = True
        stack.append(neighbour)

    return open_


def _neighbours(dims: tuple[int, ...], cells: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    strides = _strides(dims)
    axis, step = dirs >> 1, (dirs & 1) * 2 - 1
    return cells + step * strides[axis]


def make_standard_maze(
    dims: tuple[int, ...], p: float, seed: int, goal_reward: float = 1.0
) -> TabularMdp:
    open_ = _carve(dims, seed)

    open_cell, open_dir = np.nonzero(open_)
    degree = open_.sum(axis=1).astype(np.int64)
    state_begin = np.concatenate(([0], np.cumsum(degree)))
    neighbour = _neighbours(dims, open_cell, open_dir)

    moving = open_cell != GOAL
    pair_state = open_cell[moving]
    pair_dir = open_dir[moving]
    width = degree[pair_state]

    num_actions = degree.copy()
    num_actions[GOAL] = 1
    sa_begin = np.concatenate(([0], np.cumsum(num_actions)))

    succ_begin = np.concatenate(([0], np.cumsum(np.concatenate(([1], width)))))
    total = int(width.sum())

    local_start = np.concatenate(([0], np.cumsum(width)))[:-1]
    offset = np.arange(total) - np.repeat(local_start, width)
    source = np.repeat(state_begin[pair_state], width) + offset

    succ_state = neighbour[source]
    intended = np.repeat(pair_dir, width)
    prob = (1.0 - p) / np.repeat(width, width)
    prob[open_dir[source] == intended] += p

    # Every move costs 1 except one landing on the goal, which pays a reward.
    # The model carries c(s, a), so take the expectation over successors -- the
    # Bellman operator only ever sees the expected immediate cost, so this is
    # exact rather than an approximation.
    reaching = np.add.reduceat(prob * (succ_state == GOAL), local_start)
    cost = 1.0 - reaching * (1.0 + goal_reward)

    return build(
        sa_begin=sa_begin.astype(INDEX),
        succ_begin=succ_begin.astype(INDEX),
        succ_state=np.concatenate(([GOAL], succ_state)).astype(INDEX),
        succ_prob=np.concatenate(([1.0], prob)).astype(VALUE),
        cost=np.concatenate(([0.0], cost)).astype(VALUE),
    )
