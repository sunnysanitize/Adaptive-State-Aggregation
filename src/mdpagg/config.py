import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class InventoryProblem(Frozen):

    kind: Literal["inventory"] = "inventory"
    num_assets: int = Field(ge=1)
    q_max: int = Field(ge=1)
    lam: float = Field(gt=0.0)
    # Equicorrelated sigma, unit diagonal. Bounded below 1 to stay positive
    # definite; 0 is permitted so the separable control arm stays expressible.
    rho: float = Field(ge=0.0, lt=1.0)
    fill: tuple[float, ...]
    spread: tuple[float, ...]
    gamma: float = Field(default=0.95, gt=0.0, lt=1.0)
    solve_tol: float = Field(default=1e-10, gt=0.0)
    target_norm: float = Field(default=100.0, gt=0.0)


ProblemCfg = Annotated[
    MazeProblem | InventoryProblem, Field(discriminator="kind")
]


class FixedEpsilonCfg(Frozen):

    kind: Literal["fixed"] = "fixed"
    value: float = Field(gt=0.0)


class ResidualSpanEpsilonCfg(Frozen):

    kind: Literal["residual_span"] = "residual_span"
    c: float = Field(gt=0.0)
    eps_min: float = Field(gt=0.0)


class GeometricEpsilonCfg(Frozen):

    kind: Literal["geometric"] = "geometric"
    eps_0: float = Field(gt=0.0)
    eps_min: float = Field(gt=0.0)
    # The schedule divides by cycles - 1, and is defined for C >= 2.
    cycles: int = Field(ge=2)

    @model_validator(mode="after")
    def _endpoints_are_ordered(self) -> "GeometricEpsilonCfg":
        if self.eps_min >= self.eps_0:
            raise ValueError(
                f"eps_min {self.eps_min} must be below eps_0 {self.eps_0}: the "
                "schedule is floored at eps_min, so an inverted pair would run "
                "at a constant eps while reporting itself as a decay arm"
            )
        return self


EpsilonCfg = Annotated[
    FixedEpsilonCfg | ResidualSpanEpsilonCfg | GeometricEpsilonCfg,
    Field(discriminator="kind"),
]


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
    policy_loss_at: tuple[int, ...] | None = None


class RunCfg(Frozen):

    problem: ProblemCfg
    algorithm: AlgorithmCfg
    trace: TraceCfg = TraceCfg()
    master_seed: int = 0

    @model_validator(mode="after")
    def _checkpoints_are_reachable(self) -> "RunCfg":
        for t in self.trace.policy_loss_at or ():
            if t < 0 or t >= self.algorithm.iterations:
                raise ValueError(
                    f"policy_loss checkpoint {t} would never fire: the loop runs "
                    f"t over range({self.algorithm.iterations})"
                )
            if t % self.trace.fine_stride:
                raise ValueError(
                    f"policy_loss checkpoint {t} would never fire: the observer "
                    f"only sees rows on the fine stride of {self.trace.fine_stride}"
                )
        return self


def problem_seed(problem: ProblemCfg) -> int | None:
    # None means the generator is deterministic in its parameters, not that a
    # seed was forgotten. The inventory MDP has no problem-level randomness --
    # its 20 paired seeds at 5.6 are sampling seeds, carried on `master_seed`.
    return problem.seed if problem.kind == "maze" else None


def with_problem_seed(problem: ProblemCfg, seed: int) -> ProblemCfg:
    if problem.kind != "maze":
        raise ValueError(
            f"seeding the problem is meaningless for a {problem.kind!r} problem: "
            "it is deterministic in its parameters, so it would change nothing "
            "while appearing to vary the instance. Vary the sampling stream "
            "instead -- `--seed` on a single run, `--vary sampling` on a sweep."
        )
    return problem.model_copy(update={"seed": seed})


def canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )


def problem_hash(problem: ProblemCfg) -> str:
    return hashlib.sha256(canonical_json(problem).encode()).hexdigest()


def load(path: str | Path) -> RunCfg:
    return RunCfg.model_validate_json(Path(path).read_text())
