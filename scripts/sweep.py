import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mdpagg.config import (
    EpsilonCfg,
    FixedEpsilonCfg,
    GeometricEpsilonCfg,
    ResidualSpanEpsilonCfg,
    RunCfg,
    load,
    with_problem_seed,
)
from mdpagg.run import RESULTS_ROOT, execute
from mdpagg.solve import CACHE_ROOT

EPS_GRID = (0.05, 0.1, 0.5)
SEEDS = tuple(range(20))


RESIDUAL_C = 0.084
EPS_0 = 0.4985  ## = RESIDUAL_C * span_0, span_0 = 5.93429 and deterministic
EPS_MIN = 0.05
CYCLES_SLOW = 1429  ## decay spread over every aggregate cycle
CYCLES_FAST = 14  ## rate-matched to residual's projected floor at cycle 13

ARMS: dict[str, EpsilonCfg] = {
    "fixed_0.05": FixedEpsilonCfg(value=0.05),
    "fixed_0.1": FixedEpsilonCfg(value=0.1),
    "fixed_0.5": FixedEpsilonCfg(value=0.5),
    "residual": ResidualSpanEpsilonCfg(c=RESIDUAL_C, eps_min=EPS_MIN),
    "geometric_slow": GeometricEpsilonCfg(
        eps_0=EPS_0, eps_min=EPS_MIN, cycles=CYCLES_SLOW
    ),
    "geometric_fast": GeometricEpsilonCfg(
        eps_0=EPS_0, eps_min=EPS_MIN, cycles=CYCLES_FAST
    ),
}

## (treatment, control, predicted outcome). Predictions are preregistered; they
## are carried into the output so a reader sees what was expected beside what
## happened, rather than having to take the note's word for it afterwards.
CONTRASTS = (
    ("residual", "fixed_0.05", "null or negative"),
    ("residual", "geometric_slow", "residual wins, on annealing rate not feedback"),
    ("residual", "geometric_fast", "null -- the real test"),
)

BUDGETS_MS = (5, 10, 20, 50, 100, 200, 400, 800)
PRIMARY_BUDGET_MS = 20
NULL_REGION = 0.02


def with_epsilon(
    cfg: RunCfg, epsilon: EpsilonCfg, seed: int, vary: str
) -> RunCfg:
    ## Paired: seed index i gives the same maze and the same sampling stream in
    ## every arm, so a difference between arms can only come from eps.
    ##
    ## vary="both" is the meaningful reading of "20 seeds" -- 20 maze instances.
    ## vary="sampling" holds the maze fixed and varies only the sampling stream;
    ## it reproduces the zero-variance result recorded in the repro note, since
    ## the state attaining the max error sits in a size-one group.
    algorithm = cfg.algorithm.model_copy(update={"epsilon": epsilon})
    update: dict[str, Any] = {"algorithm": algorithm, "master_seed": seed}
    if vary == "both":
        update["problem"] = with_problem_seed(cfg.problem, seed)

    return cfg.model_copy(update=update)


def arm(cfg: RunCfg, eps: float, seed: int, vary: str) -> RunCfg:
    return with_epsilon(cfg, FixedEpsilonCfg(value=eps), seed, vary)


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


def err_at_budget(trace: dict[str, Any], budget_ns: float) -> float:
    ## A step function, deliberately, not an interpolation: the endpoint is the
    ## error the solver had actually delivered by the budget, not the one it was
    ## on its way to. NaN when the budget expires before the first traced row,
    ## which is a real answer -- that arm had produced nothing yet.
    latest = math.nan
    for wall_ns, err in zip(trace["wall_ns"], trace["err_inf"], strict=True):
        if wall_ns > budget_ns:
            break
        latest = err

    return latest


def clamped_rate(trace: dict[str, Any]) -> float:
    ## frequent clamping means the run measured the widened eps_effective
    ## rather than the eps its policy named, which would void the comparison.
    rows = trace["clamped"]
    return sum(1 for c in rows if c) / len(rows) if rows else 0.0


def median_ci(
    diffs: list[float], resamples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    ## Percentile bootstrap on the median of the paired differences. Seeded, so
    ## the interval is a property of the data rather than of when it was run.
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(diffs), size=(resamples, len(diffs)), replace=True)
    lo, hi = np.percentile(np.median(draws, axis=1), [2.5, 97.5])

    return float(lo), float(hi)


def contrast(
    rows: list[dict[str, Any]], treatment: str, control: str, prediction: str
) -> list[dict[str, Any]]:
    ## Differences are formed per seed and summarized afterwards, per 6.3 -- a
    ## CI around each arm separately would discard the pairing that makes the
    ## comparison sensitive.
    by_arm = {
        name: {r["seed"]: r for r in rows if r["arm"] == name}
        for name in (treatment, control)
    }
    seeds = sorted(set(by_arm[treatment]) & set(by_arm[control]))

    out = []
    for label, key in [(f"{b}ms", str(b)) for b in BUDGETS_MS] + [("final", None)]:
        def err(row: dict[str, Any], key: str | None = key) -> float:
            return row["err_inf"] if key is None else row["budget_err"][key]

        diffs = [
            err(by_arm[treatment][s]) - err(by_arm[control][s]) for s in seeds
        ]
        if any(math.isnan(d) for d in diffs):
            continue

        lo, hi = median_ci(diffs)
        out.append(
            {
                "treatment": treatment,
                "control": control,
                "prediction": prediction,
                "budget": label,
                "primary": label == f"{PRIMARY_BUDGET_MS}ms",
                "n": len(seeds),
                "median": statistics.median(diffs),
                "mean": statistics.fmean(diffs),
                "ci_lo": lo,
                "ci_hi": hi,
                ## the whole interval inside the region, not merely a point estimate that lands there.
                "within_null_region": abs(lo) < NULL_REGION
                and abs(hi) < NULL_REGION,
                "diffs": diffs,
            }
        )

    return out


def summarize_arms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name in ARMS:
        at_arm = [r for r in rows if r["arm"] == name]
        if not at_arm:
            continue
        primary = [r["budget_err"][str(PRIMARY_BUDGET_MS)] for r in at_arm]
        out.append(
            {
                "arm": name,
                "n": len(at_arm),
                **spread([r["err_inf"] for r in at_arm], "err"),
                **spread([r["policy_loss"] for r in at_arm], "policy_loss"),
                "primary_budget_err_median": statistics.median(primary)
                if not any(math.isnan(p) for p in primary)
                else math.nan,
                "num_groups_mean": statistics.fmean(
                    [r["num_groups"] for r in at_arm]
                ),
                "clamped_rate_mean": statistics.fmean(
                    [r["clamped_rate"] for r in at_arm]
                ),
            }
        )

    return out


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


def run_arms(
    base: RunCfg, vary: str, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    ## Three-arm comparison. Kept separate from the eps-grid path above
    ## because that path is Gate 3's maze reproduction and must stay runnable
    ## and unchanged.
    rows: list[dict[str, Any]] = []
    budgets_ns = {str(b): b * 1e6 for b in BUDGETS_MS}

    for name, epsilon in ARMS.items():
        for seed in SEEDS:
            cfg = with_epsilon(base, epsilon, seed, vary)
            try:
                doc = execute(
                    cfg, args.root, trace_policy_loss=args.policy_loss_curve
                )
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                return None

            final = doc["final"]
            row: dict[str, Any] = {
                "arm": name,
                "seed": seed,
                "epsilon": epsilon.model_dump(mode="json"),
                "wall_ns": doc["wall_ns"],
                "budget_err": {
                    k: err_at_budget(doc["trace"], ns) for k, ns in budgets_ns.items()
                },
                "clamped_rate": clamped_rate(doc["trace"]),
                **final,
            }
            if args.curves:
                row["trace"] = doc["trace"]
            rows.append(row)
            print(
                f"{name:<15} seed={seed:<3} "
                f"err_inf={final['err_inf']:.6g} "
                f"@{PRIMARY_BUDGET_MS}ms={row['budget_err'][str(PRIMARY_BUDGET_MS)]:.6g} "
                f"K={final['num_groups']}"
            )

    contrasts = [
        c for t, ctl, pred in CONTRASTS for c in contrast(rows, t, ctl, pred)
    ]

    return rows, {
        "arms": {k: v.model_dump(mode="json") for k, v in ARMS.items()},
        "budgets_ms": list(BUDGETS_MS),
        "primary_budget_ms": PRIMARY_BUDGET_MS,
        "null_region": NULL_REGION,
        "summary": summarize_arms(rows),
        "contrasts": contrasts,
    }


def report_arms(payload: dict[str, Any]) -> None:
    print("\narms, final iterate:")
    for s in payload["summary"]:
        print(
            f"  {s['arm']:<15} err {s['err_median']:.4g} "
            f"[{s['err_q1']:.4g}, {s['err_q3']:.4g}]  "
            f"@{payload['primary_budget_ms']}ms {s['primary_budget_err_median']:.4g}  "
            f"loss {s['policy_loss_median']:.4g}  K~{s['num_groups_mean']:.0f}  "
            f"clamped {s['clamped_rate_mean']:.1%}"
        )

    print(f"\npreregistered contrasts (null region +/-{payload['null_region']}):")
    for c in payload["contrasts"]:
        if not c["primary"] and c["budget"] != "final":
            continue
        verdict = "NULL" if c["within_null_region"] else "outside null"
        mark = " <- PRIMARY" if c["primary"] else ""
        print(
            f"  {c['treatment']} - {c['control']:<15} @{c['budget']:<6} "
            f"median {c['median']:+.4g}  CI [{c['ci_lo']:+.4g}, {c['ci_hi']:+.4g}]  "
            f"{verdict}{mark}"
        )
        if c["primary"]:
            print(f"       predicted: {c['prediction']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/sweep.py")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--vary", choices=("both", "sampling"), default=None)
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--policy-loss-curve", action="store_true")
    ## Preregistered three-arm comparison, as against the eps grid.
    parser.add_argument("--arms", action="store_true")
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

    if args.arms:
        result = run_arms(base, vary, args)
        if result is None:
            return 1
        rows, payload = result

        out = args.out or RESULTS_ROOT / f"arms_{args.config.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "config": base.model_dump(mode="json"),
                    "vary": vary,
                    "seeds": list(SEEDS),
                    "rows": rows,
                    **payload,
                },
                indent=2,
            )
        )
        report_arms(payload)
        print(f"\nwrote {out}")
        return 0

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
