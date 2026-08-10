import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numba
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mdpagg.config import RunCfg, load
from mdpagg.mdp import TabularMdp, unpack
from mdpagg.partition import allocate, rebin_by_value
from mdpagg.solve import CACHE_ROOT, load_ground_truth

EPS_GRID = (0.05, 0.1, 0.5)
SEEDS = tuple(range(20))


@numba.njit
def _support_distances(sa_begin, succ_begin, succ_state, goal):
    """Shortest support-graph distance to goal, computed on reversed edges."""
    num_states = sa_begin.shape[0] - 1
    indegree = np.zeros(num_states, dtype=np.int64)

    for state in range(num_states):
        for pair in range(sa_begin[state], sa_begin[state + 1]):
            for edge in range(succ_begin[pair], succ_begin[pair + 1]):
                indegree[succ_state[edge]] += 1

    offset = np.empty(num_states + 1, dtype=np.int64)
    offset[0] = 0
    for state in range(num_states):
        offset[state + 1] = offset[state] + indegree[state]

    cursor = offset[:-1].copy()
    predecessor = np.empty(offset[-1], dtype=np.int32)
    for state in range(num_states):
        for pair in range(sa_begin[state], sa_begin[state + 1]):
            for edge in range(succ_begin[pair], succ_begin[pair + 1]):
                successor = succ_state[edge]
                predecessor[cursor[successor]] = state
                cursor[successor] += 1

    distance = np.full(num_states, -1, dtype=np.int64)
    queue = np.empty(num_states, dtype=np.int32)
    distance[goal] = 0
    queue[0] = goal
    head = 0
    tail = 1

    while head < tail:
        state = queue[head]
        head += 1
        for edge in range(offset[state], offset[state + 1]):
            previous = predecessor[edge]
            if distance[previous] == -1:
                distance[previous] = distance[state] + 1
                queue[tail] = previous
                tail += 1

    return distance


def distances(mdp: TabularMdp) -> np.ndarray:
    sa_begin, succ_begin, succ_state, _, _ = unpack(mdp)
    out = _support_distances(sa_begin, succ_begin, succ_state, 0)
    if np.any(out < 0):
        raise ValueError(f"support graph has {int(np.count_nonzero(out < 0))} states that cannot reach the goal")
    return out


def partition_metrics(v: np.ndarray, eps: float) -> dict[str, Any]:
    raw_capacity = int(np.ceil((float(np.max(v)) - float(np.min(v))) / eps)) + 1
    part = allocate(v.shape[0], min(v.shape[0], max(raw_capacity, 1)))
    rebin_by_value(v, eps, part.capacity, part)
    sizes = np.diff(part.offset[: part.num_groups + 1]).astype(np.int64)
    weights = sizes / sizes.sum()

    return {
        "eps": eps,
        "within_eps_of_max": float(np.mean(float(np.max(v)) - v <= eps)),
        "num_groups": int(part.num_groups),
        "largest_group": int(np.max(sizes)),
        "largest_group_fraction": float(np.max(weights)),
        "median_group_size": float(np.median(sizes)),
        "singleton_groups": int(np.count_nonzero(sizes == 1)),
        "singleton_group_fraction": float(np.mean(sizes == 1)),
        "effective_groups": float(1.0 / np.sum(weights * weights)),
        "group_sizes": sizes.tolist(),
    }


def audit_one(cfg: RunCfg, seed: int, root: Path, eps_grid: tuple[float, ...]) -> dict[str, Any]:
    problem = cfg.problem.model_copy(update={"seed": seed})
    truth, mdp = load_ground_truth(problem, root)
    v = truth.v_star
    distance = distances(mdp)
    quantiles = np.quantile(distance, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))

    return {
        "dims": list(problem.dims),
        "num_states": int(v.shape[0]),
        "seed": seed,
        "gamma": problem.gamma,
        "effective_horizon": 1.0 / (1.0 - problem.gamma),
        "v_min": float(np.min(v)),
        "v_max": float(np.max(v)),
        "distinct_exact": int(np.unique(v).shape[0]),
        "distinct_rounded_6": int(np.unique(np.round(v, 6)).shape[0]),
        "distance_quantiles": {
            name: float(value)
            for name, value in zip(("min", "q25", "median", "q75", "q90", "q99", "max"), quantiles, strict=True)
        },
        "partitions": [partition_metrics(v, eps) for eps in eps_grid],
    }


def mean_stdev(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "stdev": float(np.std(array, ddof=1)) if array.shape[0] > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize(rows: list[dict[str, Any]], eps_grid: tuple[float, ...]) -> list[dict[str, Any]]:
    summaries = []
    dimensions = sorted({tuple(row["dims"]) for row in rows}, key=np.prod)
    for dims in dimensions:
        selected = [row for row in rows if tuple(row["dims"]) == dims]
        partitions = []
        for eps in eps_grid:
            eps_rows = [next(part for part in row["partitions"] if part["eps"] == eps) for row in selected]
            partitions.append(
                {
                    "eps": eps,
                    "num_groups": mean_stdev([part["num_groups"] for part in eps_rows]),
                    "largest_group_fraction": mean_stdev([part["largest_group_fraction"] for part in eps_rows]),
                    "median_group_size": mean_stdev([part["median_group_size"] for part in eps_rows]),
                    "singleton_group_fraction": mean_stdev([part["singleton_group_fraction"] for part in eps_rows]),
                    "effective_groups": mean_stdev([part["effective_groups"] for part in eps_rows]),
                }
            )

        summaries.append(
            {
                "dims": list(dims),
                "num_states": selected[0]["num_states"],
                "gamma": selected[0]["gamma"],
                "effective_horizon": selected[0]["effective_horizon"],
                "n": len(selected),
                "distance_median": mean_stdev([row["distance_quantiles"]["median"] for row in selected]),
                "distance_q99": mean_stdev([row["distance_quantiles"]["q99"] for row in selected]),
                "distance_max": mean_stdev([row["distance_quantiles"]["max"] for row in selected]),
                "partitions": partitions,
            }
        )
    return summaries


def plot(doc: dict[str, Any], path: Path) -> None:
    summaries = doc["summary"]
    rows = doc["rows"]
    eps_grid = doc["eps_grid"]
    states = np.asarray([summary["num_states"] for summary in summaries])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=180)
    for eps in eps_grid:
        groups = [next(part for part in summary["partitions"] if part["eps"] == eps)["num_groups"]["mean"] for summary in summaries]
        axes[0, 0].plot(states, groups, marker="o", label=f"ε={eps:g}")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("states |S|")
    axes[0, 0].set_ylabel("occupied groups K")
    axes[0, 0].set_title("Aggregation width stops exposing new work")
    axes[0, 0].legend(frameon=False)

    largest = [
        next(part for part in summary["partitions"] if part["eps"] == 0.5)["largest_group_fraction"]["mean"]
        for summary in summaries
    ]
    axes[0, 1].plot(states, np.asarray(largest) * 100.0, marker="o", color="#b44b38")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_ylim(0.0, 101.0)
    axes[0, 1].set_xlabel("states |S|")
    axes[0, 1].set_ylabel("largest group (% of states)")
    axes[0, 1].set_title("One value bin absorbs almost every state")

    seed_zero = sorted((row for row in rows if row["seed"] == 0), key=lambda row: row["num_states"])
    for row in seed_zero:
        sizes = next(part for part in row["partitions"] if part["eps"] == 0.5)["group_sizes"]
        ordered = np.sort(np.asarray(sizes))[::-1]
        axes[1, 0].plot(np.arange(1, ordered.shape[0] + 1), ordered, label=f"{row['dims'][0]}²")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("group rank")
    axes[1, 0].set_ylabel("states in group")
    axes[1, 0].set_title("Group-size skew at ε=0.5, seed 0")
    axes[1, 0].legend(frameon=False)

    median_distance = [summary["distance_median"]["mean"] for summary in summaries]
    q99_distance = [summary["distance_q99"]["mean"] for summary in summaries]
    axes[1, 1].plot(states, median_distance, marker="o", label="median distance")
    axes[1, 1].plot(states, q99_distance, marker="o", label="99th percentile")
    axes[1, 1].axhline(summaries[0]["effective_horizon"], linestyle="--", color="#777777", label="1/(1−γ)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("states |S|")
    axes[1, 1].set_ylabel("support-graph steps")
    axes[1, 1].set_title("Maze distances exceed the discount horizon")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Value-space saturation in the paper-faithful standard maze", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/value_space_audit.json"))
    parser.add_argument("--figure", type=Path, default=Path("results/figures/value_space_saturation.png"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--eps", type=float, nargs="+", default=list(EPS_GRID))
    args = parser.parse_args()

    eps_grid = tuple(args.eps)
    rows = []
    for path in args.configs:
        cfg = load(path)
        for seed in args.seeds:
            print(f"audit dims={cfg.problem.dims} seed={seed}", flush=True)
            rows.append(audit_one(cfg, seed, args.root, eps_grid))

    doc = {
        "configs": [str(path) for path in args.configs],
        "seeds": args.seeds,
        "eps_grid": list(eps_grid),
        "rows": rows,
        "summary": summarize(rows, eps_grid),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    plot(doc, args.figure)
    print(f"wrote {args.out}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
