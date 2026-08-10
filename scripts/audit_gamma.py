import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from audit_value_space import partition_metrics  # noqa: E402

from mdpagg.config import MazeProblem
from mdpagg.solve import CACHE_ROOT, cache_path, load_ground_truth, save, solve

GAMMAS = (0.95, 0.99, 0.999)
SIZES = (100, 200)
SEEDS = (0,)


def solve_or_load(problem: MazeProblem, root: Path):
    path = cache_path(problem, root)
    if path.exists():
        truth, _ = load_ground_truth(problem, root)
        return truth, True

    truth, _ = solve(problem)
    save(truth, problem, root)
    return truth, False


def summarize(rows: list[dict[str, Any]], size: int, workers: int, eps: float) -> dict[str, Any]:
    candidates = []
    for gamma in sorted({row["gamma"] for row in rows}):
        selected = [row for row in rows if row["size"] == size and row["gamma"] == gamma]
        largest = float(np.mean([row["partition"]["largest_group_fraction"] for row in selected]))
        groups = float(np.mean([row["partition"]["num_groups"] for row in selected]))
        passes = largest < 0.9 and groups >= 8 * workers
        candidates.append(
            {
                "gamma": gamma,
                "size": size,
                "largest_group_fraction_mean": largest,
                "num_groups_mean": groups,
                "workers": workers,
                "required_groups": 8 * workers,
                "passes": passes,
            }
        )

    passing = [candidate for candidate in candidates if candidate["passes"] and candidate["gamma"] != 0.95]
    return {
        "selection_rule": (
            f"smallest non-paper gamma at eps={eps:g} with largest group < 90% "
            "and K >= 8 groups per worker"
        ),
        "candidates": candidates,
        "selected_gamma": min((candidate["gamma"] for candidate in passing), default=None),
    }


def plot(rows: list[dict[str, Any]], eps: float, path: Path) -> None:
    sizes = sorted({row["size"] for row in rows})
    gammas = sorted({row["gamma"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=180)

    for size in sizes:
        largest = []
        groups = []
        for gamma in gammas:
            selected = [row for row in rows if row["size"] == size and row["gamma"] == gamma]
            largest.append(np.mean([row["partition"]["largest_group_fraction"] for row in selected]))
            groups.append(np.mean([row["partition"]["num_groups"] for row in selected]))
        axes[0].plot(gammas, np.asarray(largest) * 100.0, marker="o", label=f"{size}²")
        axes[1].plot(gammas, groups, marker="o", label=f"{size}²")

    axes[0].axhline(90.0, linestyle="--", color="#777777", label="selection ceiling")
    axes[0].set_ylabel(f"largest ε={eps:g} group (% of states)")
    axes[0].set_xlabel("discount γ")
    axes[0].set_xticks(gammas, [f"{gamma:g}" for gamma in gammas])
    axes[0].set_title("Value saturation")
    axes[0].legend(frameon=False)

    axes[1].axhline(8 * (os.cpu_count() or 1), linestyle="--", color="#777777", label="8 groups / worker")
    axes[1].set_ylabel(f"occupied groups K at ε={eps:g}")
    axes[1].set_xlabel("discount γ")
    axes[1].set_xticks(gammas, [f"{gamma:g}" for gamma in gammas])
    axes[1].set_title("Aggregate parallel grain")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/gamma_audit.json"))
    parser.add_argument("--figure", type=Path, default=Path("results/figures/gamma_audit.png"))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--p", type=float, default=0.92)
    args = parser.parse_args()

    rows = []
    for size in args.sizes:
        for gamma in args.gammas:
            for seed in args.seeds:
                problem = MazeProblem(dims=(size, size), p=args.p, seed=seed, gamma=gamma)
                print(f"gamma audit size={size} gamma={gamma} seed={seed}", flush=True)
                truth, cached = solve_or_load(problem, args.root)
                rows.append(
                    {
                        "size": size,
                        "num_states": size * size,
                        "gamma": gamma,
                        "seed": seed,
                        "cached": cached,
                        "solve_iterations": truth.iterations,
                        "solve_wall_ns": truth.wall_ns,
                        "partition": partition_metrics(truth.v_star, args.eps),
                    }
                )

    workers = os.cpu_count() or 1
    doc = {
        "sizes": args.sizes,
        "gammas": args.gammas,
        "seeds": args.seeds,
        "eps": args.eps,
        "workers": workers,
        "rows": rows,
        "decision": summarize(rows, max(args.sizes), workers, args.eps),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    plot(rows, args.eps, args.figure)
    print(json.dumps(doc["decision"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
