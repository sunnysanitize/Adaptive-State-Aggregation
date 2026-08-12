import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import trace as trace_module
from .adaptive import (
    AdaptiveResult,
    AdaptiveState,
    AlternatingSchedule,
    EpsilonPolicy,
    FixedEpsilon,
    ResidualSpanEpsilon,
    run_adaptive,
)
from .config import EpsilonCfg, RunCfg, load, problem_seed, with_problem_seed
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
    if cfg.kind == "fixed":
        return FixedEpsilon(cfg.value)
    if cfg.kind == "residual_span":
        return ResidualSpanEpsilon(cfg.c, cfg.eps_min)

    raise ValueError(f"unrecognized epsilon kind {cfg.kind!r}")


def seeds_of(cfg: RunCfg) -> dict[str, Any]:
    return {
        "master": cfg.master_seed,
        "problem": problem_seed(cfg.problem),
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
                        max_groups=cfg.algorithm.max_groups, observer=observer)


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

    return doc


def default_output(cfg: RunCfg, config_path: Path) -> Path:
    seed = problem_seed(cfg.problem)
    tag = "" if seed is None else f"_p{seed}"
    return RESULTS_ROOT / f"{config_path.stem}{tag}_seed{cfg.master_seed}.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m mdpagg.run")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--problem-seed", type=int, default=None)
    return parser.parse_args(argv)


def report(out: Path, doc: dict[str, Any]) -> None:
    final = doc["final"]
    print(f"wrote      {out}")
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
        problem = with_problem_seed(cfg.problem, args.problem_seed)
        cfg = cfg.model_copy(update={"problem": problem})

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
