import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numba
import numpy as np

from . import trace as trace_module
from .adaptive import (
    AdaptiveResult,
    AdaptiveState,
    AlternatingSchedule,
    EpsilonPolicy,
    FixedEpsilon,
    run_adaptive,
)
from .config import EpsilonCfg, RunCfg, load
from .mdp import TabularMdp
from .norms import max_norm
from .partition import lift_into
from .policy import policy_loss
from .rng import streams
from .solve import CACHE_ROOT, GroundTruth, load_ground_truth
from .trace import Trace
from .types import VALUE, Phase, ValueArray

RESULTS_ROOT = Path("results")

Observer = Callable[[int, Phase, AdaptiveState], None]


def make_epsilon(cfg: EpsilonCfg) -> EpsilonPolicy:
    return FixedEpsilon(cfg.value)


def seeds_of(cfg: RunCfg) -> dict[str, Any]:
    return {
        "master": cfg.master_seed,
        "problem": cfg.problem.seed,
        "sampling_derivation": "SeedSequence(master).spawn(2)[1]",
    }


def loss_against(
    cfg: RunCfg, mdp: TabularMdp, v: ValueArray, truth: GroundTruth
) -> float:
    return policy_loss(mdp, v, truth.v_star, cfg.problem.gamma, cfg.problem.solve_tol)


def observer_for(
    cfg: RunCfg,
    mdp: TabularMdp,
    truth: GroundTruth,
    trace: Trace,
    trace_policy_loss: bool = True,
) -> Observer:
    lifted: ValueArray = np.empty(mdp.num_states, dtype=VALUE)

    def iterate(phase: Phase, state: AdaptiveState) -> ValueArray:
        if phase is Phase.AGGREGATE and state.part.num_groups > 0:
            lift_into(state.part, state.w, lifted)
            return lifted
        return state.v

    def observe(t: int, phase: Phase, state: AdaptiveState) -> None:
        if not trace.wants_row(t):
            return

        current = iterate(phase, state)
        loss = math.nan

        if trace_policy_loss and trace.wants_policy_loss(t):
            loss = loss_against(cfg, mdp, current, truth)

        trace.record(
            t=t,
            phase=0 if phase is Phase.GLOBAL else 1,
            err_inf=float(max_norm(current - truth.v_star)),
            residual_span=state.residual_span,
            num_groups=state.part.num_groups,
            eps=state.part.eps_effective,
            clamped=state.part.groups_clamped,
            billed=state.counters.billed,
            actual=state.counters.actual,
            wall_ns=state.wall_ns,
            policy_loss=loss,
        )

    return observe


def solve(cfg: RunCfg, mdp: TabularMdp, observer: Observer) -> AdaptiveResult:

    schedule = AlternatingSchedule(global_len=cfg.algorithm.schedule.global_len,
                                   agg_len=cfg.algorithm.schedule.agg_len)

    return run_adaptive(mdp, cfg.problem.gamma, cfg.algorithm.iterations,
                        schedule, make_epsilon(cfg.algorithm.epsilon),
                        streams(cfg.master_seed).sampling,
                        max_groups=cfg.algorithm.max_groups, observer=observer,
                        parallel=cfg.execution.parallel,
                        threads=cfg.execution.threads)


# Requested and observed both, because they can differ: Numba clamps a request
# above its configured maximum, and a run that quietly got fewer threads than
# it asked for would otherwise look like poor scaling.
def execution_of(result: AdaptiveResult) -> dict[str, Any]:
    return {
        "parallel": result.parallel,
        "threads_requested": result.threads_requested,
        "threads_observed": result.threads_observed,
        "threading_layer": str(numba.threading_layer()) if result.parallel else None,
    }


def summary_of(cfg: RunCfg, mdp: TabularMdp, truth: GroundTruth,
               trace: Trace, result: AdaptiveResult) -> dict[str, Any]:

    return {
        "err_inf": float(max_norm(result.v - truth.v_star)),
        "policy_loss": loss_against(cfg, mdp, result.v, truth),
        "num_groups": int(trace.num_groups[: trace.rows].max()) if trace.rows else 0,
        "t_sa": result.t_sa,
        "billed": result.counters.billed,
        "actual": result.counters.actual,
        "overhead_fraction": result.counters.overhead_fraction,
    }


def execute(
    cfg: RunCfg,
    root: Path = CACHE_ROOT,
    trace_policy_loss: bool = True,
) -> dict[str, Any]:

    truth, mdp = load_ground_truth(cfg.problem, root)

    trace = trace_module.allocate(cfg.algorithm.iterations,
                                  fine_stride=cfg.trace.fine_stride,
                                  coarse_stride=cfg.trace.coarse_stride)

    result = solve(
        cfg,
        mdp,
        observer_for(cfg, mdp, truth, trace, trace_policy_loss=trace_policy_loss),
    )

    doc = trace_module.document(trace, cfg.model_dump(mode="json"),
                                seeds_of(cfg), truth.hash, result.wall_ns)

    doc["final"] = summary_of(cfg, mdp, truth, trace, result)
    doc["execution"] = execution_of(result)

    return doc


def default_output(cfg: RunCfg, config_path: Path) -> Path:
    return (
        RESULTS_ROOT
        / f"{config_path.stem}_p{cfg.problem.seed}_seed{cfg.master_seed}.json"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m mdpagg.run")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--problem-seed", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args(argv)


def report(out: Path, doc: dict[str, Any]) -> None:
    final = doc["final"]
    execution = doc["execution"]
    print(f"wrote      {out}")
    if execution["parallel"]:
        print(
            f"  threads      {execution['threads_observed']} observed"
            f"  (requested {execution['threads_requested']},"
            f" layer {execution['threading_layer']})"
        )
    print(f"  err_inf      {final['err_inf']:.6g}")
    print(f"  policy_loss  {final['policy_loss']:.6g}")
    print(f"  K (max)      {final['num_groups']}")
    print(f"  billed       {final['billed']}  actual {final['actual']}")
    print(f"  overhead     {final['overhead_fraction']:.1%}")
    print(f"  wall         {doc['wall_ns'] / 1e9:.3f} s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg: RunCfg = load(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"master_seed": args.seed})
    if args.problem_seed is not None:
        problem = cfg.problem.model_copy(update={"seed": args.problem_seed})
        cfg = cfg.model_copy(update={"problem": problem})
    if args.parallel or args.threads is not None:
        execution = cfg.execution.model_copy(update={
            "parallel": args.parallel or cfg.execution.parallel,
            "threads": args.threads if args.threads is not None else cfg.execution.threads,
        })
        cfg = cfg.model_copy(update={"execution": execution})

    try:
        doc = execute(cfg, args.root)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    out = args.out or default_output(cfg, args.config)
    trace_module.write(out, doc)
    report(out, doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
