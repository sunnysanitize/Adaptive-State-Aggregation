from dataclasses import dataclass
import numpy as np
from .types import INDEX, VALUE, IndexArray, ValueArray

ROW_SUM_TOL = 1e-12

@dataclass(frozen=True)
class TabularMdp:

    sa_begin: IndexArray  
    succ_begin: IndexArray 
    succ_state: IndexArray  
    succ_prob: ValueArray  
    cost: ValueArray  

    @property
    def num_states(self) -> int:
        return self.sa_begin.shape[0] - 1

    @property
    def num_pairs(self) -> int:
        return self.cost.shape[0]

    def num_actions(self, s: int) -> int:
        return int(self.sa_begin[s + 1] - self.sa_begin[s])

    def pair_index(self, s: int, a: int) -> int:
        return int(self.sa_begin[s]) + a

    def successors(self, pair: int) -> IndexArray:
        return self.succ_state[self.succ_begin[pair] : self.succ_begin[pair + 1]]

    def probabilities(self, pair: int) -> ValueArray:
        return self.succ_prob[self.succ_begin[pair] : self.succ_begin[pair + 1]]


def build(
    sa_begin: IndexArray,
    succ_begin: IndexArray,
    succ_state: IndexArray,
    succ_prob: ValueArray,
    cost: ValueArray,
) -> TabularMdp:

    _check_dtypes(sa_begin, succ_begin, succ_state, succ_prob, cost)
    _check_shapes(sa_begin, succ_begin, succ_state, succ_prob, cost)
    _check_actions_exist(sa_begin)
    _check_indices_in_range(sa_begin, succ_begin, succ_state, succ_prob)
    _check_probabilities(sa_begin, succ_begin, succ_prob)

    for array in (sa_begin, succ_begin, succ_state, succ_prob, cost):
        array.setflags(write=False)

    return TabularMdp(sa_begin, succ_begin, succ_state, succ_prob, cost)


def unpack(
    m: TabularMdp,
) -> tuple[IndexArray, IndexArray, IndexArray, ValueArray, ValueArray]:
    ## Numba cannot take the frozen dataclass, so this is the boundary between the
    ## model and every jitted function:

    return m.sa_begin, m.succ_begin, m.succ_state, m.succ_prob, m.cost


def _state_of(sa_begin: IndexArray, pair: int) -> int:
    return int(np.searchsorted(sa_begin, pair, side="right") - 1)


def _check_dtypes(sa_begin, succ_begin, succ_state, succ_prob, cost) -> None:
    declared = [
        ("sa_begin", sa_begin, INDEX),
        ("succ_begin", succ_begin, INDEX),
        ("succ_state", succ_state, INDEX),
        ("succ_prob", succ_prob, VALUE),
        ("cost", cost, VALUE),
    ]
    for name, array, dtype in declared:
        if array.dtype != dtype:
            raise ValueError(
                f"{name} has dtype {array.dtype}, expected {np.dtype(dtype)}"
            )


def _check_shapes(sa_begin, succ_begin, succ_state, succ_prob, cost) -> None:
    for name, array in [
        ("sa_begin", sa_begin),
        ("succ_begin", succ_begin),
        ("succ_state", succ_state),
        ("succ_prob", succ_prob),
        ("cost", cost),
    ]:
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {array.shape}")

    if sa_begin.shape[0] < 2:
        raise ValueError("sa_begin must have length |S| + 1 >= 2")
    if sa_begin[0] != 0:
        raise ValueError(f"sa_begin[0] must be 0, got {sa_begin[0]}")
    if succ_begin[0] != 0:
        raise ValueError(f"succ_begin[0] must be 0, got {succ_begin[0]}")

    num_pairs = int(sa_begin[-1])
    if succ_begin.shape[0] != num_pairs + 1:
        raise ValueError(
            f"succ_begin must have length {num_pairs + 1} "
            f"(|pairs| + 1), got {succ_begin.shape[0]}"
        )
    if cost.shape[0] != num_pairs:
        raise ValueError(
            f"cost must have length {num_pairs} (|pairs|), got {cost.shape[0]}"
        )
    if succ_state.shape[0] != succ_prob.shape[0]:
        raise ValueError(
            f"succ_state and succ_prob must be the same length, got "
            f"{succ_state.shape[0]} and {succ_prob.shape[0]}"
        )
    if int(succ_begin[-1]) != succ_state.shape[0]:
        raise ValueError(
            f"succ_begin[-1] is {int(succ_begin[-1])} but succ_state has "
            f"{succ_state.shape[0]} entries"
        )


def _check_actions_exist(sa_begin: IndexArray) -> None:
    counts = np.diff(sa_begin)
    bad = np.flatnonzero(counts < 1)
    if bad.size:
        raise ValueError(f"state {bad[0]} has no actions")


def _check_indices_in_range(sa_begin, succ_begin, succ_state, succ_prob) -> None:
    num_states = sa_begin.shape[0] - 1

    widths = np.diff(succ_begin)
    bad = np.flatnonzero(widths < 1)
    if bad.size:
        pair = int(bad[0])
        raise ValueError(
            f"state {_state_of(sa_begin, pair)}, action "
            f"{pair - int(sa_begin[_state_of(sa_begin, pair)])} has no successors"
        )

    bad = np.flatnonzero((succ_state < 0) | (succ_state >= num_states))
    if bad.size:
        entry = int(bad[0])
        pair = int(np.searchsorted(succ_begin, entry, side="right") - 1)
        raise ValueError(
            f"state {_state_of(sa_begin, pair)} has successor "
            f"{int(succ_state[entry])}, outside [0, {num_states})"
        )


def _check_probabilities(sa_begin, succ_begin, succ_prob) -> None:
    bad = np.flatnonzero(~(succ_prob >= 0.0))  # catches NaN too
    if bad.size:
        entry = int(bad[0])
        pair = int(np.searchsorted(succ_begin, entry, side="right") - 1)
        raise ValueError(
            f"state {_state_of(sa_begin, pair)} has probability "
            f"{succ_prob[entry]}, which is not >= 0"
        )

    row_sums = np.add.reduceat(succ_prob, succ_begin[:-1].astype(np.intp))
    bad = np.flatnonzero(np.abs(row_sums - 1.0) > ROW_SUM_TOL)
    if bad.size:
        pair = int(bad[0])
        state = _state_of(sa_begin, pair)
        raise ValueError(
            f"state {state}, action {pair - int(sa_begin[state])} has "
            f"probabilities summing to {float(row_sums[pair])!r}, not 1"
        )
