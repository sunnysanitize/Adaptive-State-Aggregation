from dataclasses import dataclass

import numba
import numpy as np

from .types import INDEX, VALUE, IndexArray, ValueArray

MAX_RAW_BINS = 1 << 24


@dataclass
class Partition:

    group_of: IndexArray
    members: IndexArray
    offset: IndexArray
    centers: ValueArray
    num_groups: int = 0
    eps_effective: float = 0.0
    groups_clamped: bool = False

    @property
    def capacity(self) -> int:
        return self.centers.shape[0]

    def group(self, j: int) -> IndexArray:
        return self.members[self.offset[j] : self.offset[j + 1]]


def allocate(num_states: int, max_groups: int) -> Partition:
    if max_groups < 1:
        raise ValueError(f"max_groups must be >= 1, got {max_groups}")
    if max_groups > num_states:
        max_groups = num_states

    return Partition(
        group_of=np.zeros(num_states, dtype=INDEX),
        members=np.zeros(num_states, dtype=INDEX),
        offset=np.zeros(max_groups + 1, dtype=INDEX),
        centers=np.zeros(max_groups, dtype=VALUE),
    )


def lift_into(part: Partition, w: ValueArray, v: ValueArray) -> None:
    np.take(w, part.group_of, out=v)


# A gather, one independent write per state. No arithmetic, so the threaded
# form is exact by construction rather than by luck.
@numba.njit(parallel=True)
def _gather(w, group_of, v):
    for s in numba.prange(v.shape[0]):
        v[s] = w[group_of[s]]


def lift_into_parallel(part: Partition, w: ValueArray, v: ValueArray) -> None:
    _gather(w, part.group_of, v)


@numba.njit(inline="always")
def _raw_bin(value, b1, eps, raw_bins):

    i = int((value - b1) / eps)
    if i >= raw_bins:
        return raw_bins - 1
    if i < 0:
        return 0
    return i


@numba.njit(cache=True)
def _count_groups(v, b1, eps, raw_bins):
    seen = np.zeros(raw_bins, dtype=np.uint8)

    num_groups = 0
    for s in range(v.shape[0]):
        i = _raw_bin(v[s], b1, eps, raw_bins)
        if seen[i] == 0:
            seen[i] = 1
            num_groups += 1

    return num_groups


@numba.njit(cache=True)
def _fill(v, b1, eps, raw_bins, group_of, members, offset, centers):
    counts = np.zeros(raw_bins, dtype=np.int32)
    for s in range(v.shape[0]):
        counts[_raw_bin(v[s], b1, eps, raw_bins)] += 1

    group_of_raw = np.empty(raw_bins, dtype=np.int32)
    num_groups = 0
    total = 0
    for i in range(raw_bins):
        if counts[i] > 0:
            group_of_raw[i] = num_groups
            offset[num_groups] = total
            centers[num_groups] = b1 + (i + 0.5) * eps
            total += counts[i]
            num_groups += 1
    offset[num_groups] = total

    cursor = np.empty(num_groups, dtype=np.int32)
    for j in range(num_groups):
        cursor[j] = offset[j]

    for s in range(v.shape[0]):
        j = group_of_raw[_raw_bin(v[s], b1, eps, raw_bins)]
        group_of[s] = j
        members[cursor[j]] = s
        cursor[j] += 1

    return num_groups


def rebin_by_value(
    v: ValueArray,
    eps: float,
    max_groups: int,
    out: Partition,
) -> None:
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    if v.shape[0] != out.group_of.shape[0]:
        raise ValueError(
            f"v has {v.shape[0]} states, partition was allocated for "
            f"{out.group_of.shape[0]}"
        )
    max_groups = min(max_groups, out.capacity)

    b1 = float(np.min(v))
    b2 = float(np.max(v))
    if not (np.isfinite(b1) and np.isfinite(b2)):
        raise ValueError(f"v is not finite: min {b1!r}, max {b2!r}")

    if b1 == b2:
        out.group_of[:] = 0
        out.members[:] = np.arange(v.shape[0], dtype=INDEX)
        out.offset[0] = 0
        out.offset[1] = v.shape[0]
        out.centers[0] = b1
        out.num_groups = 1
        out.eps_effective = eps
        out.groups_clamped = False
        return

    raw_bins = int(np.ceil((b2 - b1) / eps))
    if raw_bins > MAX_RAW_BINS:
        raise ValueError(
            f"eps {eps!r} over a value range of {b2 - b1!r} needs {raw_bins} "
            f"raw bins, above the {MAX_RAW_BINS} cap"
        )

    groups_clamped = _count_groups(v, b1, eps, raw_bins) > max_groups
    if groups_clamped:
        raw_bins = max_groups
        eps = (b2 - b1) / max_groups

    out.num_groups = _fill(
        v, b1, eps, raw_bins, out.group_of, out.members, out.offset, out.centers
    )
    out.eps_effective = eps
    out.groups_clamped = groups_clamped
