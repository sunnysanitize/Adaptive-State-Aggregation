import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mdpagg.config import InventoryProblem, load
from mdpagg.inventory import (
    NUM_ACTIONS,
    decode,
    do_nothing_policy,
    equicorrelated,
    immediate_cost_policy,
    linear_hedge_policy,
    num_states,
)
from mdpagg.mdp import TabularMdp
from mdpagg.policy import greedy_policy, policy_value
from mdpagg.solve import CACHE_ROOT, load_ground_truth
from mdpagg.types import IndexArray, ValueArray


def baselines(problem: InventoryProblem) -> dict[str, IndexArray]:
    sigma = equicorrelated(problem.num_assets, problem.rho)
    fill = np.asarray(problem.fill)
    spread = np.asarray(problem.spread)

    return {
        "do_nothing": do_nothing_policy(problem.num_assets, problem.q_max),
        "immediate_cost": immediate_cost_policy(
            problem.num_assets, problem.q_max, fill, problem.lam, sigma, spread
        ),
        "linear_hedge": linear_hedge_policy(problem.num_assets, problem.q_max),
    }


def score(
    mdp: TabularMdp,
    policy: IndexArray,
    v_star: ValueArray,
    gamma: float,
    exposure: np.ndarray,
) -> dict[str, Any]:
    gap = policy_value(mdp, policy, gamma) - v_star
    used = np.bincount(policy, minlength=NUM_ACTIONS)

    return {
        "sup_gap": float(gap.max()),
        "mean_gap": float(gap.mean()),
        "min_gap": float(gap.min()),
        "median_gap": float(np.median(gap)),
        "actions_used": used.tolist(),
        "action_by_exposure": [
            float(policy[exposure == k].mean()) for k in range(exposure.max() + 1)
        ],
    }


def check(problem: InventoryProblem, root: Path) -> dict[str, Any]:
    truth, mdp = load_ground_truth(problem, root)
    states = np.arange(num_states(problem.num_assets, problem.q_max))
    exposure = np.abs(decode(states, problem.num_assets, problem.q_max)).max(axis=1)

    rows = {
        name: score(mdp, policy, truth.v_star, problem.gamma, exposure)
        for name, policy in baselines(problem).items()
    }
    rows["optimal"] = score(
        mdp,
        greedy_policy(mdp, truth.v_star, problem.gamma),
        truth.v_star,
        problem.gamma,
        exposure,
    )

    return {
        "num_assets": problem.num_assets,
        "q_max": problem.q_max,
        "num_states": int(truth.v_star.shape[0]),
        "gamma": problem.gamma,
        "scale": float(truth.scale),
        "v_star_norm": float(np.abs(truth.v_star).max()),
        "policies": rows,
    }


def dominance(doc: dict[str, Any]) -> tuple[bool, list[str]]:
    admissible = 3e-10 / (1.0 - doc["gamma"])
    verdicts = []
    passed = True

    for name, row in doc["policies"].items():
        if name == "optimal":
            continue
        beats = row["min_gap"] <= -admissible
        ties = row["sup_gap"] <= admissible
        passed &= not (beats or ties)
        verdicts.append(
            f"{name:14s} sup {row['sup_gap']:9.4f}  mean {row['mean_gap']:8.4f}  "
            f"min {row['min_gap']:11.3e}  "
            + ("BEATS OPTIMAL" if beats else "ties optimal" if ties else "worse, as required")
        )

    return passed, verdicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, default=Path("configs/inventory_n3.json"), nargs="?")
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("results/inventory_baselines.json"))
    args = parser.parse_args()

    problem = load(args.config).problem
    if not isinstance(problem, InventoryProblem):
        raise SystemExit(
            f"{args.config} is a {problem.kind!r} problem; this check is inventory-only"
        )

    doc = check(problem, args.root)
    passed, verdicts = dominance(doc)
    doc["baselines_dominated"] = passed

    for line in verdicts:
        print(line)
    print(f"baselines dominated: {'PASS' if passed else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
