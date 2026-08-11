import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOTSTRAP = 10000
SEED = 0


# Percentile bootstrap rather than a t interval: n is 5 on the sensitivity arm,
# and nothing here justifies assuming the differences are normal. No scipy
# dependency either, which keeps a clean checkout able to reproduce this.
def ci(values, level=0.95, resamples=BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    means = draws.mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def load(paths):
    by = defaultdict(dict)
    meta = {}
    for path in paths:
        doc = json.loads(Path(path).read_text())
        meta = {"threads": doc["threads"], "target": doc["target"]}
        for row in doc["rows"]:
            by[(row["config"], row["arm"])][row["problem_seed"]] = row
    return by, meta


# p* is preregistered as the thread count with the lowest median time, chosen
# per solver. Taken across seeds so one unlucky seed cannot pick it.
def best_threads(rows):
    counts = next(iter(rows.values()))["timings"].keys()
    return min(
        counts,
        key=lambda p: float(np.median([r["timings"][p]["median_s"] for r in rows.values()])),
    )


def at(rows, p):
    return {seed: row["timings"][p]["median_s"] for seed, row in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/scaling_summary.json"))
    args = parser.parse_args()

    by, meta = load(args.results)
    configs = sorted({config for config, _ in by}, key=lambda c: (("sensitivity" in c), c))

    out = []
    print(f"target err_inf <= {meta['target']}\n")
    print(f"{'config':32s} {'|S|':>8s} {'p*VI':>5s} {'p*AG':>5s} "
          f"{'ratio':>7s} {'95% CI on paired diff (s)':>30s} {'n':>3s}")

    for config in configs:
        vi_rows, ag_rows = by[(config, "vi")], by[(config, "adaptive")]
        seeds = sorted(set(vi_rows) & set(ag_rows))

        p_vi, p_ag = best_threads(vi_rows), best_threads(ag_rows)
        vi_t, ag_t = at(vi_rows, p_vi), at(ag_rows, p_ag)

        diffs = [ag_t[s] - vi_t[s] for s in seeds]
        ratios = [ag_t[s] / vi_t[s] for s in seeds]
        mean_d, lo, hi = ci(diffs)
        mean_r, r_lo, r_hi = ci(ratios)

        # Preregistered: supported only if the interval excludes zero.
        supported = lo > 0

        eff_vi = [vi_rows[s]["efficiency"]["10"] for s in seeds]
        eff_ag = [ag_rows[s]["efficiency"]["10"] for s in seeds]
        ser_r = [
            ag_rows[s]["timings"]["1"]["median_s"] / vi_rows[s]["timings"]["1"]["median_s"]
            for s in seeds
        ]

        out.append({
            "config": config,
            "num_states": vi_rows[seeds[0]]["num_states"],
            "gamma": vi_rows[seeds[0]]["gamma"],
            "seeds": len(seeds),
            "p_star": {"vi": p_vi, "adaptive": p_ag},
            "paired_difference_s": {"mean": mean_d, "ci95": [lo, hi]},
            "ratio_at_p_star": {"mean": mean_r, "ci95": [r_lo, r_hi]},
            "serial_ratio_mean": float(np.mean(ser_r)),
            "efficiency_10": {
                "vi": float(np.mean(eff_vi)),
                "adaptive": float(np.mean(eff_ag)),
            },
            "primary_supported": bool(supported),
        })

        name = config.replace("configs/", "").replace(".json", "")
        mark = "" if supported else "   <-- CI includes 0"
        print(f"{name:32s} {out[-1]['num_states']:8d} {p_vi:>5s} {p_ag:>5s} "
              f"{mean_r:6.3f}x  [{lo:+.4f}, {hi:+.4f}]{mark}")

    print(f"\n{'config':32s} {'serial ratio':>12s} {'p* ratio':>10s} "
          f"{'effVI(10)':>10s} {'effAG(10)':>10s}")
    for row in out:
        name = row["config"].replace("configs/", "").replace(".json", "")
        print(f"{name:32s} {row['serial_ratio_mean']:11.3f}x "
              f"{row['ratio_at_p_star']['mean']:9.3f}x "
              f"{row['efficiency_10']['vi']:10.2f} {row['efficiency_10']['adaptive']:10.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"target": meta["target"], "rows": out}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
