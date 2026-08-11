import argparse
import json
import time
from pathlib import Path

import numba
import numpy as np

from mdpagg.adaptive import _aggregate_sweep, _aggregate_sweep_parallel
from mdpagg.maze import make_standard_maze
from mdpagg.mdp import unpack
from mdpagg.partition import allocate, rebin_by_value
from mdpagg.types import VALUE

GAMMA = 0.95
ALPHA = 0.5


# v = arange(|S|) % k gives exactly k distinct values, so the real binning code
# produces exactly k groups. Members of a group are then spread across the
# state space, which is what value-based binning does on a maze anyway: states
# at a similar distance from the goal are not neighbours.
def partition_with(k, part, num_states):
    v = (np.arange(num_states) % k).astype(VALUE)
    rebin_by_value(v, 0.5, k, part)

    if part.num_groups != k:
        raise RuntimeError(f"wanted {k} groups, binning produced {part.num_groups}")


def time_kernel(kernel, args, min_seconds, min_calls):
    calls = 0
    start = time.perf_counter_ns()

    while True:
        kernel(*args)
        calls += 1
        elapsed = time.perf_counter_ns() - start
        if calls >= min_calls and elapsed >= min_seconds * 1e9:
            return elapsed / calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=500)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    parser.add_argument("--min-seconds", type=float, default=0.1)
    parser.add_argument("--min-calls", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/aggregate_grain.json"))
    args = parser.parse_args()

    mdp = make_standard_maze((args.dims, args.dims), 0.92, seed=0)
    arrays = unpack(mdp)
    n = mdp.num_states

    grid = [k for k in (1 << i for i in range(2, 17)) if k <= n]
    part = allocate(n, max(grid))
    rng = np.random.default_rng(0)

    top = max(grid)
    w = rng.random(top).astype(VALUE)
    out_serial = np.empty(top, dtype=VALUE)
    out_parallel = np.empty(top, dtype=VALUE)
    draws = rng.random(top).astype(VALUE)

    def call(kernel, out, k):
        return (
            *arrays, w, out, part.group_of, part.members, part.offset,
            k, draws, ALPHA, GAMMA,
        ), kernel

    # Warm both kernels at a size nothing below depends on, so no compile lands
    # inside a timed loop.
    partition_with(grid[0], part, n)
    for kernel, out in ((_aggregate_sweep, out_serial), (_aggregate_sweep_parallel, out_parallel)):
        kernel(*call(kernel, out, grid[0])[0])

    rows = []
    mismatches = []

    for k in grid:
        partition_with(k, part, n)

        serial_args, _ = call(_aggregate_sweep, out_serial, k)
        serial_ns = time_kernel(_aggregate_sweep, serial_args, args.min_seconds, args.min_calls)
        _aggregate_sweep(*serial_args)

        threaded = {}
        for p in args.threads:
            numba.set_num_threads(p)
            parallel_args, _ = call(_aggregate_sweep_parallel, out_parallel, k)
            threaded[p] = time_kernel(
                _aggregate_sweep_parallel, parallel_args, args.min_seconds, args.min_calls
            )

            _aggregate_sweep_parallel(*parallel_args)
            if not np.array_equal(out_parallel[:k], out_serial[:k]):
                worst = float(np.max(np.abs(out_parallel[:k] - out_serial[:k])))
                mismatches.append({"k": k, "threads": p, "max_abs_diff": worst})

        best_p = min((p for p in threaded if p > 1), key=lambda p: threaded[p], default=None)
        rows.append({
            "k": k,
            "serial_ns": serial_ns,
            "threaded_ns": threaded,
            "best_threads": best_p,
            "speedup": None if best_p is None else serial_ns / threaded[best_p],
        })

        best = "" if best_p is None else f"  best {best_p:>2d} threads  {serial_ns / threaded[best_p]:5.2f}x"
        print(f"K = {k:>6d}  serial {serial_ns / 1e3:9.1f} us{best}")

    paying = [r for r in rows if r["speedup"] is not None and r["speedup"] > 1.0]
    crossover = paying[0]["k"] if paying else None

    doc = {
        "num_states": n,
        "dims": [args.dims, args.dims],
        "threads": args.threads,
        "numba_threading_layer": str(numba.threading_layer()),
        "max_threads": numba.config.NUMBA_NUM_THREADS,
        "crossover_k": crossover,
        "bitwise_mismatches": mismatches,
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))

    print(f"\nthreading layer      {doc['numba_threading_layer']}")
    print(f"bitwise mismatches   {len(mismatches)}")
    if crossover is None:
        print("crossover            none -- threading never pays at any tested K")
    else:
        print(f"crossover            K = {crossover}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
