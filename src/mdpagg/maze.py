import numpy as np


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
