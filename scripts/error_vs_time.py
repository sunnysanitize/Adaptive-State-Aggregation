import argparse
import json
from pathlib import Path

import numpy as np

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.config import load
from mdpagg.norms import max_norm
from mdpagg.partition import lift_into
from mdpagg.rng import streams
from mdpagg.solve import CACHE_ROOT, load_ground_truth
from mdpagg.types import VALUE, Phase

BUDGETS = {0.95: [0.25, 0.5, 1.0, 2.0], 0.999: [0.5, 1.0, 2.0, 4.0]}


# Diagnostic, not an endpoint measurement. An observer runs here, which the
# timed runs in scripts/scaling.py deliberately avoid. It sits outside the
# solver's own clock so `wall_ns` stays solver-only, but it does disturb cache
# state, so these curves are for shape and for reading error at a budget --
# not for time-to-target, which scaling.py owns.
def curve(mdp, truth, cfg, schedule, parallel, threads, stride):
    lifted = np.empty(mdp.num_states, dtype=VALUE)
    wall, err = [], []

    def observe(t, phase, state):
        if t % stride:
            return
        if phase is Phase.AGGREGATE and state.part.num_groups > 0:
            lift_into(state.part, state.w, lifted)
            current = lifted
        else:
            current = state.v
        wall.append(state.wall_ns / 1e9)
        err.append(float(max_norm(current - truth.v_star)))

    run_adaptive(
        mdp,
        cfg.problem.gamma,
        cfg.algorithm.iterations,
        schedule,
        FixedEpsilon(cfg.algorithm.epsilon.value),
        streams(cfg.master_seed).sampling,
        max_groups=cfg.algorithm.max_groups,
        observer=observe,
        parallel=parallel,
        threads=threads,
    )

    return wall, err


def at_budgets(wall, err, budgets):
    out = {}
    for b in budgets:
        reached = [e for w, e in zip(wall, err, strict=True) if w <= b]
        out[str(b)] = float(reached[-1]) if reached else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/error_vs_time.json"))
    args = parser.parse_args()

    rows = []
    for path in args.configs:
        cfg = load(path)
        problem = cfg.problem.model_copy(update={"seed": args.seed})
        cfg = cfg.model_copy(update={"problem": problem})
        truth, mdp = load_ground_truth(cfg.problem, args.root)

        arms = {
            "vi": AlternatingSchedule(global_len=1, agg_len=0),
            "adaptive": AlternatingSchedule(
                global_len=cfg.algorithm.schedule.global_len,
                agg_len=cfg.algorithm.schedule.agg_len,
            ),
        }
        budgets = BUDGETS[cfg.problem.gamma]

        print(f"{path}  |S| = {mdp.num_states}")
        for arm, schedule in arms.items():
            for threads in (1, args.threads):
                wall, err = curve(
                    mdp, truth, cfg, schedule, threads > 1, threads, args.stride
                )
                rows.append({
                    "config": str(path),
                    "arm": arm,
                    "threads": threads,
                    "num_states": mdp.num_states,
                    "gamma": cfg.problem.gamma,
                    "wall_s": wall,
                    "err_inf": err,
                    "err_at_budget": at_budgets(wall, err, budgets),
                })
                shown = " ".join(
                    f"{b}s:{'--' if v is None else format(v, '.4g')}"
                    for b, v in rows[-1]["err_at_budget"].items()
                )
                print(f"  {arm:9s} p={threads:<3d} final {err[-1]:.4g}   {shown}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"budgets": BUDGETS, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
