import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    ProblemCfg,
    RunCfg,
    canonical_json,
    load,
    problem_hash,
    with_problem_seed,
)
from .inventory import equicorrelated, make_inventory_mdp
from .maze import make_standard_maze
from .mdp import TabularMdp, build, unpack
from .norms import max_norm
from .timer import timed
from .types import VALUE, ValueArray
from .vi import value_iteration

CACHE_ROOT = Path("results/ground_truth")


@dataclass(frozen=True)
class GroundTruth:

    hash: str
    scale: float
    v_star: ValueArray
    iterations: int
    wall_ns: int


def build_problem(problem: ProblemCfg) -> TabularMdp:
    if problem.kind == "maze":
        return make_standard_maze(problem.dims, problem.p, problem.seed)
    if problem.kind == "inventory":
        return make_inventory_mdp(
            problem.num_assets,
            problem.q_max,
            np.asarray(problem.fill, dtype=VALUE),
            problem.lam,
            equicorrelated(problem.num_assets, problem.rho),
            np.asarray(problem.spread, dtype=VALUE),
        )

    raise ValueError(f"unknown problem kind {problem.kind!r}")


def rescale(mdp: TabularMdp, scale: float) -> TabularMdp:
    sa_begin, succ_begin, succ_state, succ_prob, cost = unpack(mdp)

    return build(
        sa_begin.copy(),
        succ_begin.copy(),
        succ_state.copy(),
        succ_prob.copy(),
        (cost * scale).astype(VALUE),
    )


def cache_path(problem: ProblemCfg, root: Path = CACHE_ROOT) -> Path:
    return root / f"{problem_hash(problem)}.npz"


def solve(problem: ProblemCfg) -> tuple[GroundTruth, TabularMdp]:
    with timed() as elapsed:
        raw = build_problem(problem)
        first = value_iteration(raw, problem.gamma, tol=problem.solve_tol)

        norm = float(max_norm(first.v))
        if norm == 0.0:
            raise ValueError(
                "||V*||inf is 0 on the unscaled instance; there is nothing to "
                "rescale and every cost-to-go is identical"
            )

        scale = problem.target_norm / norm
        scaled = rescale(raw, scale)
        second = value_iteration(scaled, problem.gamma, tol=problem.solve_tol)

    return (
        GroundTruth(
            hash=problem_hash(problem),
            scale=scale,
            v_star=second.v,
            iterations=second.iterations,
            wall_ns=elapsed.wall_ns,
        ),
        scaled,
    )


def save(truth: GroundTruth, problem: ProblemCfg, root: Path = CACHE_ROOT) -> Path:
    path = cache_path(problem, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        hash=truth.hash,
        scale=truth.scale,
        v_star=truth.v_star,
        iterations=truth.iterations,
        wall_ns=truth.wall_ns,
        problem=canonical_json(problem),
    )

    return path


def load_ground_truth(
    problem: ProblemCfg, root: Path = CACHE_ROOT
) -> tuple[GroundTruth, TabularMdp]:
    path = cache_path(problem, root)

    if not path.exists():
        raise FileNotFoundError(
            f"no cached V* for this problem at {path}\n"
            f"  problem: {canonical_json(problem)}\n"
            f"  run:     python -m mdpagg.solve <config>\n"
            "This is never recomputed silently: a run measured against a V* "
            "from a different problem would look plausible and be wrong."
        )

    with np.load(path, allow_pickle=False) as data:
        stored = str(data["problem"])
        if stored != canonical_json(problem):
            raise ValueError(
                f"hash collision or stale cache at {path}\n"
                f"  stored: {stored}\n"
                f"  wanted: {canonical_json(problem)}"
            )

        truth = GroundTruth(
            hash=str(data["hash"]),
            scale=float(data["scale"]),
            v_star=data["v_star"],
            iterations=int(data["iterations"]),
            wall_ns=int(data["wall_ns"]),
        )

    mdp = rescale(build_problem(problem), truth.scale)

    if mdp.num_states != truth.v_star.shape[0]:
        raise ValueError(
            f"cached V* has {truth.v_star.shape[0]} states but the problem "
            f"rebuilds to {mdp.num_states}; the generator has changed under "
            f"the cache at {path}"
        )

    return truth, mdp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mdpagg.solve")
    parser.add_argument("config", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--problem-seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg: RunCfg = load(args.config)
    problem = cfg.problem
    if args.problem_seed is not None:
        problem = with_problem_seed(problem, args.problem_seed)

    path = cache_path(problem, args.root)

    if path.exists() and not args.force:
        truth, _ = load_ground_truth(problem, args.root)
        print(f"cache hit  {path}")
        print(f"  |S| = {truth.v_star.shape[0]}  scale = {truth.scale:.6g}")
        print(f"  ||V*||inf = {float(max_norm(truth.v_star)):.10g}")
        return 0

    truth, _ = solve(problem)
    save(truth, problem, args.root)

    print(f"solved     {path}")
    print(f"  |S| = {truth.v_star.shape[0]}  scale = {truth.scale:.6g}")
    print(f"  ||V*||inf = {float(max_norm(truth.v_star)):.10g}")
    print(f"  {truth.iterations} sweeps, {truth.wall_ns / 1e9:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
