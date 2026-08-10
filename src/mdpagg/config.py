import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")


class MazeProblem(Frozen):

    kind: Literal["maze"] = "maze"
    dims: tuple[int, ...]
    p: float = Field(gt=0.0, le=1.0)
    seed: int
    gamma: float = Field(default=0.95, gt=0.0, lt=1.0)
    solve_tol: float = Field(default=1e-10, gt=0.0)
    target_norm: float = Field(default=100.0, gt=0.0)


ProblemCfg = MazeProblem


class FixedEpsilonCfg(Frozen):

    kind: Literal["fixed"] = "fixed"
    value: float = Field(gt=0.0)


EpsilonCfg = FixedEpsilonCfg


class ScheduleCfg(Frozen):

    global_len: int = Field(default=2, ge=1)
    agg_len: int = Field(default=5, ge=0)


class AlgorithmCfg(Frozen):

    iterations: int = Field(gt=0)
    epsilon: EpsilonCfg
    schedule: ScheduleCfg = ScheduleCfg()
    max_groups: int = Field(default=4096, ge=1)


class TraceCfg(Frozen):

    fine_stride: int = Field(default=1, ge=1)
    coarse_stride: int = Field(default=50, ge=1)


class RunCfg(Frozen):

    problem: ProblemCfg
    algorithm: AlgorithmCfg
    trace: TraceCfg = TraceCfg()
    master_seed: int = 0


def canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )


def problem_hash(problem: ProblemCfg) -> str:
    return hashlib.sha256(canonical_json(problem).encode()).hexdigest()


def load(path: str | Path) -> RunCfg:
    return RunCfg.model_validate_json(Path(path).read_text())
