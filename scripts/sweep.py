import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from mdpagg.config import FixedEpsilonCfg, RunCfg, load
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
        update["problem"] = cfg.problem.model_copy(update={"seed": seed})

    return cfg.model_copy(update=update)


def summarize(rows: list[dict[str, Any]], grid: tuple[float, ...]) -> list[dict[str, Any]]:
    out = []
    for eps in grid:
        errs = [r["err_inf"] for r in rows if r["eps"] == eps]
        losses = [r["policy_loss"] for r in rows if r["eps"] == eps]
        groups = [r["num_groups"] for r in rows if r["eps"] == eps]
        out.append(
            {
                "eps": eps,
                "n": len(errs),
                "err_mean": statistics.fmean(errs),
                "err_stdev": statistics.stdev(errs),
                "err_median": statistics.median(errs),
                "err_min": min(errs),
                "err_max": max(errs),
                "policy_loss_mean": statistics.fmean(losses),
                "num_groups_mean": statistics.fmean(groups),
                "bound": 2.0 * eps / (1.0 - 0.95),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/sweep.py")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--vary", choices=("both", "sampling"), default="both")
    ## The full grid is Gate 3. The paper's 500^2 / 1000^2 comparison is a single
    ## eps, and running the whole grid there triples the cost for nothing.
    parser.add_argument("--eps", type=float, nargs="+", default=list(EPS_GRID))
    args = parser.parse_args(argv)

    grid: tuple[float, ...] = tuple(args.eps)

    base: RunCfg = load(args.config)
    rows: list[dict[str, Any]] = []

    for eps in grid:
        for seed in SEEDS:
            cfg = arm(base, eps, seed, args.vary)
            try:
                # The sweep consumes only final policy loss. Dense policy
                # evaluation is observational work outside the solver timer,
                # but interleaving it dominates process time and perturbs later
                # timings through cache and thermal state.
                doc = execute(cfg, args.root, trace_policy_loss=False)
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                return 1

            final = doc["final"]
            rows.append({"eps": eps, "seed": seed, "wall_ns": doc["wall_ns"], **final})
            print(
                f"eps={eps:<5} seed={seed:<3} "
                f"err_inf={final['err_inf']:.6g} "
                f"policy_loss={final['policy_loss']:.6g} "
                f"K={final['num_groups']}"
            )

    summary = summarize(rows, grid)
    out = args.out or RESULTS_ROOT / f"sweep_{args.config.stem}_{args.vary}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": base.model_dump(mode="json"),
                "vary": args.vary,
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
            f"  eps={s['eps']:<5} err {s['err_mean']:.4g} +/- {s['err_stdev']:.3g}  "
            f"median {s['err_median']:.4g}  K~{s['num_groups_mean']:.0f}  "
            f"bound {s['bound']:g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
