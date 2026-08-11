import argparse
import json
from pathlib import Path

import numba
import numpy as np

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.config import load
from mdpagg.norms import max_norm
from mdpagg.rng import streams
from mdpagg.solve import CACHE_ROOT, load_ground_truth
from mdpagg.trace import environment
from mdpagg.types import Phase

TARGET = 2.0
MIN_TRIALS = 7
MIN_BLOCK_S = 1.0
MAX_TRIALS = 51


def schedules(cfg):
    return {
        # agg_len = 0 is exact value iteration, pinned bit for bit.
        "vi": AlternatingSchedule(global_len=1, agg_len=0),
        "adaptive": AlternatingSchedule(
            global_len=cfg.algorithm.schedule.global_len,
            agg_len=cfg.algorithm.schedule.agg_len,
        ),
    }


def solve(mdp, cfg, schedule, iterations, observer=None, parallel=False, threads=None):
    return run_adaptive(
        mdp,
        cfg.problem.gamma,
        iterations,
        schedule,
        FixedEpsilon(cfg.algorithm.epsilon.value),
        streams(cfg.master_seed).sampling,
        max_groups=cfg.algorithm.max_groups,
        observer=observer,
        parallel=parallel,
        threads=threads,
    )


# Untimed. Finds how many iterations the arm needs to reach the target, so the
# timed runs below can be observer-free: measuring err_inf inside a timed loop
# would put an |S|-sized pass into the thing being measured.
def calibrate(mdp, truth, cfg, schedule, target):
    err = []

    def observe(_t, _phase, state):
        err.append(float(max_norm(state.v - truth.v_star)))

    solve(mdp, cfg, schedule, cfg.algorithm.iterations, observer=observe)

    hit = np.where(np.array(err) <= target)[0]
    if hit.size == 0:
        return None

    crossing = int(hit[0])
    # err_inf only moves on a global sweep -- V is untouched while the
    # aggregate phase works on W -- so the crossing must land on a global
    # iteration. If it did not, the tail lift would make the timed run's final
    # vector something other than the one calibration measured.
    if schedule.phase_at(crossing) is not Phase.GLOBAL:
        raise RuntimeError(f"crossing at iteration {crossing} is not a global phase")

    return crossing + 1


def trials_for(seconds):
    if seconds <= 0:
        return MAX_TRIALS
    return int(min(MAX_TRIALS, max(MIN_TRIALS, np.ceil(MIN_BLOCK_S / seconds))))


def time_arm(mdp, truth, cfg, schedule, iterations, parallel, threads, target):
    first = solve(mdp, cfg, schedule, iterations, parallel=parallel, threads=threads)
    seconds = [first.wall_ns / 1e9]

    for _ in range(trials_for(seconds[0]) - 1):
        result = solve(mdp, cfg, schedule, iterations, parallel=parallel, threads=threads)
        seconds.append(result.wall_ns / 1e9)

    # After the clock, never inside it.
    err = float(max_norm(first.v - truth.v_star))
    if err > target:
        raise RuntimeError(f"calibrated run ended at err_inf {err:.6g}, above {target}")

    q1, median, q3 = np.percentile(seconds, [25, 50, 75])

    return {
        "trials": len(seconds),
        "median_s": float(median),
        "iqr_s": [float(q1), float(q3)],
        "min_s": float(min(seconds)),
        "seconds": [float(s) for s in seconds],
        "final_err_inf": err,
        "threads_observed": first.threads_observed,
    }


def measure(path, seed, thread_grid, root, target):
    cfg = load(path)
    problem = cfg.problem.model_copy(update={"seed": seed})
    cfg = cfg.model_copy(update={"problem": problem})
    truth, mdp = load_ground_truth(cfg.problem, root)

    rows = []
    for arm, schedule in schedules(cfg).items():
        iterations = calibrate(mdp, truth, cfg, schedule, target)
        if iterations is None:
            print(f"  {arm:9s} never reaches {target}; skipped")
            continue

        timings = {"1": time_arm(mdp, truth, cfg, schedule, iterations, False, None, target)}
        for p in thread_grid:
            timings[str(p)] = time_arm(
                mdp, truth, cfg, schedule, iterations, True, p, target
            )

        base = timings["1"]["median_s"]
        rows.append({
            "config": str(path),
            "problem_seed": seed,
            "arm": arm,
            "num_states": mdp.num_states,
            "gamma": cfg.problem.gamma,
            "eps": cfg.algorithm.epsilon.value,
            "iterations_to_target": iterations,
            "timings": timings,
            "speedup": {p: base / t["median_s"] for p, t in timings.items()},
            "efficiency": {
                p: base / t["median_s"] / int(p) for p, t in timings.items()
            },
        })

        row = rows[-1]
        print(f"  {arm:9s} {iterations:>6d} iters  serial {base:7.4f} s  "
              f"({timings['1']['trials']} trials)")
        for p in thread_grid:
            t = timings[str(p)]
            print(f"      p={p:<3d} {t['median_s']:7.4f} s  "
                  f"speedup {row['speedup'][str(p)]:5.2f}x  "
                  f"efficiency {row['efficiency'][str(p)]:.2f}")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8, 10])
    parser.add_argument("--target", type=float, default=TARGET)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/scaling.json"))
    args = parser.parse_args()

    rows = []
    for path in args.configs:
        for seed in args.seeds:
            print(f"{path}  seed {seed}")
            rows.extend(measure(path, seed, args.threads, args.root, args.target))

    doc = {
        "target": args.target,
        "threads": args.threads,
        "min_trials": MIN_TRIALS,
        "min_block_s": MIN_BLOCK_S,
        "environment": environment(),
        "threading_layer": str(numba.threading_layer()),
        "physical_cores": 10,
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
