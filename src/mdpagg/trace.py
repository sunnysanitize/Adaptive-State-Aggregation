import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import pydantic

from .types import INDEX, VALUE, IndexArray, ValueArray

COUNT = np.int64
CountArray = npt.NDArray[np.int64]

COLUMNS = (
    "iteration",
    "phase",
    "err_inf",
    "residual_span",
    "policy_loss",
    "num_groups",
    "eps",
    "clamped",
    "billed",
    "actual",
    "wall_ns",
)


@dataclass
class Trace:

    fine_stride: int
    coarse_stride: int

    iteration: IndexArray
    phase: IndexArray
    err_inf: ValueArray
    residual_span: ValueArray
    policy_loss: ValueArray
    num_groups: IndexArray
    eps: ValueArray
    clamped: IndexArray
    billed: CountArray
    actual: CountArray
    wall_ns: CountArray

    rows: int = 0
    policy_loss_at: frozenset[int] | None = None

    def wants_row(self, t: int) -> bool:
        return t % self.fine_stride == 0

    def wants_policy_loss(self, t: int) -> bool:
        if self.policy_loss_at is not None:
            return t in self.policy_loss_at
        return t % self.coarse_stride == 0

    def record(
        self,
        t: int,
        phase: int,
        err_inf: float,
        residual_span: float,
        num_groups: int,
        eps: float,
        clamped: bool,
        billed: int,
        actual: int,
        wall_ns: int,
        policy_loss: float = math.nan,
    ) -> None:
        i = self.rows
        if i >= self.iteration.shape[0]:
            raise IndexError(
                f"trace is full at {i} rows; it was allocated for iterations "
                f"stepping by {self.fine_stride}"
            )

        self.iteration[i] = t
        self.phase[i] = phase
        self.err_inf[i] = err_inf
        self.residual_span[i] = residual_span
        self.policy_loss[i] = policy_loss
        self.num_groups[i] = num_groups
        self.eps[i] = eps
        self.clamped[i] = clamped
        self.billed[i] = billed
        self.actual[i] = actual
        self.wall_ns[i] = wall_ns
        self.rows = i + 1

    def columns(self) -> dict[str, list[Any]]:
        return {name: getattr(self, name)[: self.rows].tolist() for name in COLUMNS}


def allocate(
    iterations: int,
    fine_stride: int = 1,
    coarse_stride: int = 50,
    policy_loss_at: tuple[int, ...] | None = None,
) -> Trace:
    if fine_stride < 1 or coarse_stride < 1:
        raise ValueError(
            f"strides must be >= 1, got fine {fine_stride}, coarse {coarse_stride}"
        )

    n = -(-iterations // fine_stride)

    return Trace(
        fine_stride=fine_stride,
        coarse_stride=coarse_stride,
        policy_loss_at=None if policy_loss_at is None else frozenset(policy_loss_at),
        iteration=np.zeros(n, dtype=INDEX),
        phase=np.zeros(n, dtype=INDEX),
        err_inf=np.full(n, math.nan, dtype=VALUE),
        residual_span=np.full(n, math.nan, dtype=VALUE),
        policy_loss=np.full(n, math.nan, dtype=VALUE),
        num_groups=np.zeros(n, dtype=INDEX),
        eps=np.full(n, math.nan, dtype=VALUE),
        clamped=np.zeros(n, dtype=INDEX),
        billed=np.zeros(n, dtype=COUNT),
        actual=np.zeros(n, dtype=COUNT),
        wall_ns=np.zeros(n, dtype=COUNT),
    )


def environment() -> dict[str, Any]:
    return {
        "cpython": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "pydantic": pydantic.VERSION,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def document(
    trace: Trace,
    config: dict[str, Any],
    seeds: dict[str, Any],
    v_star_hash: str,
    wall_ns: int,
) -> dict[str, Any]:
    return {
        "config": config,
        "seeds": seeds,
        "environment": environment(),
        "v_star_hash": v_star_hash,
        "wall_ns": wall_ns,
        "strides": {
            "fine": trace.fine_stride,
            "coarse": trace.coarse_stride,
        },
        "policy_loss_at": (
            None if trace.policy_loss_at is None else sorted(trace.policy_loss_at)
        ),
        "trace": trace.columns(),
    }


def write(path: str | Path, doc: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2, sort_keys=True))


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
