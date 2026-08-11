import argparse
import json
from pathlib import Path

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.config import load
from mdpagg.rng import streams
from mdpagg.solve import CACHE_ROOT, load_ground_truth


def measure(mdp, cfg, schedule):
    result = run_adaptive(
        mdp,
        cfg.problem.gamma,
        cfg.algorithm.iterations,
        schedule,
        FixedEpsilon(cfg.algorithm.epsilon.value),
        streams(cfg.master_seed).sampling,
        max_groups=cfg.algorithm.max_groups,
        phase_timing=True,
    )

    times = result.phase_times
    assert times is not None

    return {
        "wall_s": result.wall_ns / 1e9,
        "measured_s": times.total_ns / 1e9,
        "ns": {
            "global": times.global_ns,
            "aggregate": times.aggregate_ns,
            "rebin": times.rebin_ns,
            "lift": times.lift_ns,
        },
        "share": times.share(),
        # Rebinning is the phase that cannot be parallelized without changing
        # which states get sampled, so its share is the serial fraction. Amdahl
        # caps speedup at 1/serial no matter how many threads are added.
        "amdahl_ceiling": (
            float("inf") if times.rebin_ns == 0 else times.total_ns / times.rebin_ns
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/phase_split.json"))
    args = parser.parse_args()

    rows = []
    for path in args.configs:
        cfg = load(path)
        _, mdp = load_ground_truth(cfg.problem, args.root)

        schedule = AlternatingSchedule(
            global_len=cfg.algorithm.schedule.global_len,
            agg_len=cfg.algorithm.schedule.agg_len,
        )

        row = {
            "config": str(path),
            "num_states": mdp.num_states,
            "gamma": cfg.problem.gamma,
            "eps": cfg.algorithm.epsilon.value,
            "iterations": cfg.algorithm.iterations,
            "adaptive": measure(mdp, cfg, schedule),
        }
        rows.append(row)

        a = row["adaptive"]
        print(f"{path}  |S| = {mdp.num_states}  gamma = {cfg.problem.gamma}")
        print(f"  clocked {a['wall_s']:.3f} s, of which {a['measured_s']:.3f} s attributed")
        for name, share in a["share"].items():
            print(f"    {name:<10s} {share:6.2%}  {a['ns'][name] / 1e9:7.3f} s")
        print(f"  speedup ceiling if rebin stays serial: {a['amdahl_ceiling']:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
