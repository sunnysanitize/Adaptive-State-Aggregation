import argparse
import json
from pathlib import Path

import numpy as np

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.config import load
from mdpagg.norms import max_norm
from mdpagg.rng import streams
from mdpagg.solve import CACHE_ROOT, load_ground_truth

TARGETS = (10.0, 5.0, 3.0, 2.5, 2.0, 1.5, 1.0)


def trajectory(mdp, truth, cfg, schedule):
    err = []
    wall = []

    def observe(_t, _phase, state):
        err.append(float(max_norm(state.v - truth.v_star)))
        wall.append(state.wall_ns)

    result = run_adaptive(
        mdp,
        cfg.problem.gamma,
        cfg.algorithm.iterations,
        schedule,
        FixedEpsilon(cfg.algorithm.epsilon.value),
        streams(cfg.master_seed).sampling,
        max_groups=cfg.algorithm.max_groups,
        observer=observe,
    )

    return np.array(err), np.array(wall), result


# The aggregate phase holds V fixed while W moves, so err_inf on V is flat
# across a phase and steps only on lift. First crossing is therefore reported
# on the recorded iterate, not interpolated.
def crossings(err, wall, targets):
    out = {}
    for target in targets:
        idx = np.where(err <= target)[0]
        out[str(target)] = (
            None
            if idx.size == 0
            else {
                "iteration": int(idx[0]),
                "wall_s": float(wall[idx[0]] / 1e9),
                "err_inf": float(err[idx[0]]),
            }
        )
    return out


def arm(mdp, truth, cfg, schedule, targets):
    err, wall, result = trajectory(mdp, truth, cfg, schedule)

    return {
        "schedule": {"global_len": schedule.global_len, "agg_len": schedule.agg_len},
        "iterations": cfg.algorithm.iterations,
        "final_err_inf": float(max_norm(result.v - truth.v_star)),
        "min_err_inf": float(err.min()),
        "min_at_iteration": int(err.argmin()),
        "total_wall_s": result.wall_ns / 1e9,
        "billed": result.counters.billed,
        "actual": result.counters.actual,
        "crossings": crossings(err, wall, targets),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/baseline_targets.json"))
    parser.add_argument("--problem-seed", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for path in args.configs:
        cfg = load(path)
        if args.problem_seed is not None:
            problem = cfg.problem.model_copy(update={"seed": args.problem_seed})
            cfg = cfg.model_copy(update={"problem": problem})

        truth, mdp = load_ground_truth(cfg.problem, args.root)

        configured = AlternatingSchedule(
            global_len=cfg.algorithm.schedule.global_len,
            agg_len=cfg.algorithm.schedule.agg_len,
        )

        row = {
            "config": str(path),
            "dims": list(cfg.problem.dims),
            "num_states": mdp.num_states,
            "gamma": cfg.problem.gamma,
            "eps": cfg.algorithm.epsilon.value,
            "problem_seed": cfg.problem.seed,
            "master_seed": cfg.master_seed,
            # agg_len = 0 turns the loop into exact value iteration. Running
            # both solvers through it keeps their timing measured the same way,
            # rather than comparing against vi.value_iteration separately.
            "vi": arm(mdp, truth, cfg, AlternatingSchedule(1, 0), TARGETS),
            "adaptive": arm(mdp, truth, cfg, configured, TARGETS),
        }
        rows.append(row)

        print(f"{path}  |S| = {mdp.num_states}  gamma = {cfg.problem.gamma}")
        for name in ("vi", "adaptive"):
            a = row[name]
            print(
                f"  {name:9s} final {a['final_err_inf']:.4g}  "
                f"min {a['min_err_inf']:.4g} @ {a['min_at_iteration']}  "
                f"{a['total_wall_s']:.3f} s"
            )
            for target, hit in a["crossings"].items():
                if hit is not None:
                    print(
                        f"    <= {target:>5s}  iter {hit['iteration']:>6d}  "
                        f"{hit['wall_s']:.4f} s"
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"targets": list(TARGETS), "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
