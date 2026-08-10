import argparse
import json
from pathlib import Path

from mdpagg.adaptive import AlternatingSchedule, FixedEpsilon, run_adaptive
from mdpagg.config import load
from mdpagg.norms import max_norm
from mdpagg.rng import streams
from mdpagg.solve import CACHE_ROOT, load_ground_truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity_check.json"))
    parser.add_argument("--target", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load(args.config)
    truth, mdp = load_ground_truth(cfg.problem, args.root)
    epsilon = FixedEpsilon(cfg.algorithm.epsilon.value)

    exact = run_adaptive(
        mdp,
        cfg.problem.gamma,
        cfg.algorithm.iterations,
        AlternatingSchedule(global_len=1, agg_len=0),
        epsilon,
        streams(cfg.master_seed).sampling,
        max_groups=cfg.algorithm.max_groups,
    )

    clamped = False
    final_groups = 0

    def observe(_iteration, _phase, state):
        nonlocal clamped, final_groups
        clamped = clamped or state.part.groups_clamped
        final_groups = state.part.num_groups

    adaptive = run_adaptive(
        mdp,
        cfg.problem.gamma,
        cfg.algorithm.iterations,
        AlternatingSchedule(
            global_len=cfg.algorithm.schedule.global_len,
            agg_len=cfg.algorithm.schedule.agg_len,
        ),
        epsilon,
        streams(cfg.master_seed).sampling,
        max_groups=cfg.algorithm.max_groups,
        observer=observe,
    )

    doc = {
        "config": str(args.config),
        "target": args.target,
        "exact_err_inf": float(max_norm(exact.v - truth.v_star)),
        "adaptive_err_inf": float(max_norm(adaptive.v - truth.v_star)),
        "adaptive_final_groups": final_groups,
        "groups_clamped": clamped,
    }
    doc["passes"] = (
        doc["exact_err_inf"] < args.target
        and doc["adaptive_err_inf"] < args.target
        and not doc["groups_clamped"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    if not doc["passes"]:
        raise SystemExit("sensitivity configuration failed its pre-freeze criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
