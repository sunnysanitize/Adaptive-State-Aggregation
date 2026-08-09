import numba
import numpy as np


@numba.njit
def max_norm(v):
    return np.max(np.abs(v))


@numba.njit
def max_abs_diff(a, b):
    return np.max(np.abs(a - b))


@numba.njit
def span_seminorm(v):
    return np.max(v) - np.min(v)
