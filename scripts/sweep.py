import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from mdpagg.config import FixedEpsilonCfg, RunCfg, load, with_problem_seed
from mdpagg.run import RESULTS_ROOT, execute
from mdpagg.solve import CACHE_ROOT

EPS_GRID = (0.05, 0.1, 0.5)
SEEDS = tuple(range(20))


def arm(cfg: RunCfg, eps: float, seed: int, vary: str) -> RunCfg:
    ## Paired: seed index i gives the same maze and the same sampling stream in
    ## every arm, so a difference between arms can only come from eps.
    ##
    ## vary="both" is the meaningful reading of "20 seeds" -- 20 maze instances.
    ## vary="sampling" holds the maze fixed and varies only the sampling stream;
    ## it reproduces the zero-variance result recorded in the repro note, since
    ## the state attaining the max error sits in a size-one group.
    algorithm = cfg.algorithm.model_copy(update={"epsilon": FixedEpsilonCfg(value=eps)})
    update: dict[str, Any] = {"algorithm": algorithm, "master_seed": seed}
    if vary == "both":
        update["problem"] = with_problem_seed(cfg.problem, seed)

    return cfg.model_copy(update=update)


def spread(values: list[float], prefix: str) -> dict[str, float]:
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")

    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_stdev": statistics.stdev(values),
        f"{prefix}_median": median,
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def tail_mean(doc: dict[str, Any], iterations: int, fraction: float = 0.1) -> float:
    cut = iterations * (1.0 - fraction)
    trace = doc["trace"]

    return statistics.fmean(
        [
            err
            for t, err in zip(trace["iteration"], trace["err_inf"], strict=True)
            if t >= cut
        ]
    )


def summarize(
    rows: list[dict[str, Any]], grid: tuple[float, ...], gamma: float
) -> list[dict[str, Any]]:
    out = []
    for eps in grid:
        at_eps = [r for r in rows if r["eps"] == eps]
        out.append(
            {
                "eps": eps,
                "n": len(at_eps),
                **spread([r["err_inf"] for r in at_eps], "err"),
                **spread([r["policy_loss"] for r in at_eps], "policy_loss"),
                "err_tail_mean": statistics.fmean(
                    [r["err_tail_mean"] for r in at_eps]
                ),
                "num_groups_mean": statistics.fmean(
                    [r["num_groups"] for r in at_eps]
                ),
                "bound": 2.0 * eps / (1.0 - gamma),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/sweep.py")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--vary", choices=("both", "sampling"), default=None)
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--policy-loss-curve", action="store_true")
    ## The full grid is Gate 3. The paper's 500^2 / 1000^2 comparison is a single
    ## eps, and running the whole grid there triples the cost for nothing.
    parser.add_argument("--eps", type=float, nargs="+", default=list(EPS_GRID))
    args = parser.parse_args(argv)

    grid: tuple[float, ...] = tuple(args.eps)

    base: RunCfg = load(args.config)
    ## A maze varies its instance per seed; an inventory problem is deterministic
    ## in its parameters, so only the sampling stream can move.
    vary = args.vary or ("both" if base.problem.kind == "maze" else "sampling")
    gamma = base.problem.gamma
    rows: list[dict[str, Any]] = []

    for eps in grid:
        for seed in SEEDS:
            cfg = arm(base, eps, seed, vary)
            try:
                # The sweep consumes only final policy loss. Dense policy
                # evaluation is observational work outside the solver timer,
                # but interleaving it dominates process time and perturbs later
                # timings through cache and thermal state.
                doc = execute(
                    cfg, args.root, trace_policy_loss=args.policy_loss_curve
                )
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                return 1

            final = doc["final"]
            row: dict[str, Any] = {
                "eps": eps,
                "seed": seed,
                "wall_ns": doc["wall_ns"],
                "err_tail_mean": tail_mean(doc, base.algorithm.iterations),
                **final,
            }
            if args.curves:
                row["trace"] = doc["trace"]
            rows.append(row)
            print(
                f"eps={eps:<5} seed={seed:<3} "
                f"err_inf={final['err_inf']:.6g} "
                f"policy_loss={final['policy_loss']:.6g} "
                f"K={final['num_groups']}"
            )

    summary = summarize(rows, grid, gamma)
    out = args.out or RESULTS_ROOT / f"sweep_{args.config.stem}_{vary}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": base.model_dump(mode="json"),
                "vary": vary,
                "eps_grid": list(grid),
                "seeds": list(SEEDS),
                "rows": rows,
                "summary": summary,
            },
            indent=2,
        )
    )

    print(f"\nwrote {out}")
    for s in summary:
        print(
            f"  eps={s['eps']:<5} err {s['err_median']:.4g} "
            f"[{s['err_q1']:.4g}, {s['err_q3']:.4g}]  mean {s['err_mean']:.4g}  "
            f"loss {s['policy_loss_median']:.4g}  K~{s['num_groups_mean']:.0f}  "
            f"bound {s['bound']:g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
