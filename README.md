# Residual-Driven State Aggregation for Multi-Asset Inventory Control

Does adapting the aggregation width to the Bellman residual beat simply shrinking
it on a schedule?

**No — not on this problem.** Over 120 paired runs, a residual-driven rule and a
two-line geometric schedule with no feedback at all converge to the same answer,
differing by `2.9e-06` on a scale where `‖V*‖∞ = 100`. Whatever the residual rule
buys comes from *how fast* it anneals, not from watching the residual.

This repository is the preregistered experiment behind that null result, built on
the verified numerical core tagged `shared-core-v1` (`0dce46b`).

---

## The rule under test

Adaptive state aggregation solves a discounted tabular MDP by alternating full
Bellman sweeps with cheap sweeps over groups of states binned by value. The bin
width `ε` sets the trade-off: coarse groups are cheap and inaccurate, fine groups
are expensive and accurate. The standard algorithm fixes `ε` up front.

The proposed modification makes it adaptive, re-evaluated when each aggregate
phase begins:

```
ε ← max(ε_min, c · span(TV − V))
```

The intuition: while the Bellman residual is large the value estimate is crude,
so fine groups are wasted precision; as the residual contracts, refine. It is a
feedback rule, and the question is whether the feedback earns its place.

## Why it doesn't

Three arms, identical in every respect but `ε`, on the same 20 sampling seeds:

| arm | `ε` rule |
|:--|:--|
| **fixed** | constant, at each of `{0.05, 0.1, 0.5}` |
| **residual** | `max(ε_min, c · span(TV − V))` |
| **geometric** | a preset decay from `ε₀` to `ε_min`, ignoring the residual |

The geometric arm is the one that matters. Residual-`ε` does two things at once —
it shrinks `ε` over time, *and* it shrinks it in response to the residual.
Comparing against fixed `ε` alone cannot separate them. So the geometric arm was
run twice: once on the schedule the plan specified, and once **rate-matched** to
anneal at the same speed the residual rule was projected to, leaving feedback as
the only remaining difference.

Median `err_inf` over 20 seeds, by wall-clock budget:

| arm | 20 ms | 50 ms | 100 ms | 400 ms | final |
|:--|--:|--:|--:|--:|--:|
| fixed, `ε = 0.05` | 0.7802 | 0.1342 | 0.1479 | 0.1403 | 0.1379 |
| residual | 0.6314 | 0.1344 | 0.1486 | 0.1407 | 0.1388 |
| geometric, rate-matched | 0.6762 | 0.1344 | 0.1486 | 0.1405 | 0.1388 |

From 50 ms onward — about 6% of a full run — the three agree to three or four
significant figures. The paired difference between residual and the rate-matched
geometric arm at the final iterate is **−2.9e-06**, with a bootstrap CI that
comfortably contains zero.

Two things the experiment found that were not predicted, and one mistake in it,
are written up in [`docs/residual_epsilon_note.md`](docs/residual_epsilon_note.md) —
including a preregistered endpoint that turned out to be badly chosen, and why
that is reported rather than quietly replaced.

## The problem

Multi-asset market-making inventory control, chosen because its value function is
*not* saturated — unlike the standard maze benchmark, where most states share a
value and aggregation looks good for the wrong reason.

| | |
|:--|:--|
| State | signed inventory `(q₁, …, q_N)`, each `q_i ∈ {−Q, …, Q}` |
| Size | `N = 3`, `Q = 10` → `21³ = 9,261` states |
| Actions | **5 quote-aggressiveness levels, independent of `N`** |
| Dynamics | one fill per period, `2N + 1` successors, clipped at `±Q` |
| Cost | `λ · qᵀΣq` risk penalty, minus spread captured on fills |
| Discount | `γ = 0.95`, costs rescaled so `‖V*‖∞ = 100` |

The fixed action space is the load-bearing design choice: per-asset levels would
give `5^N` actions, growing as fast as the state space, and aggregation — which
compresses *states* — would buy nothing. The formulation, its frozen constants
and the evidence that correlation actually matters are in
[`docs/inventory_design.md`](docs/inventory_design.md).

## What makes the comparison trustworthy

The result is a null, so the machinery that would have detected a real effect has
to be shown working:

- **Exact ground truth.** `V*` solved to `1e-10` and cached; a run measured
  against a `V*` from a different problem fails loudly rather than looking
  plausible.
- **The policy path is validated end to end.**
  `‖policy_value(greedy(V*)) − V*‖∞ = 1.7e-09`, inside the `3·tol/(1−γ)` the
  span-seminorm stopping rule allows.
- **Three policy baselines are all strictly worse than optimal** — pointwise, not
  just on average. A baseline beating `V*` would mean the sign convention,
  dynamics or evaluator was wrong.
- **The fixed arm is bit-identical** before and after the experimental arms were
  added, so the control is uncontaminated.
- **The baseline resolves what it measures.** Seed spread is 27–180× smaller than
  the gap between adjacent `ε`, so the experiment could have seen an effect a
  fraction of that size.
- **Constants were frozen before any result existed**, with the reasoning
  recorded — including a stability bound on `c` found by measurement.

## Reproduce

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
make all                              # lint, 3 run modes, 108 tests
scripts/reproduce_residual_epsilon.sh # ground truth, baselines, the experiment
```

`make all` runs linting, type checking, compiled tests, pure-Python tests, and a
cold bounds-checked Numba run. Two exactness gates are compiled-mode claims and
skip themselves under `make debug`, printing the reason.

The experiment takes a few minutes and writes to `results/`. Wall-clock budget
columns are machine-dependent and will differ from the published table; the
final-iterate column is deterministic given the seeds and will not.

## Layout

| Path | |
|:--|:--|
| `src/mdpagg/` | solver core — MDP, VI, partitioning, adaptive loop, ε policies |
| `configs/inventory_n3.json` | the frozen instance |
| `scripts/sweep.py --arms` | the three-arm paired experiment |
| `docs/residual_epsilon_note.md` | preregistration, results, limitations |
| `docs/inventory_design.md` | MDP formulation and validation |
| `docs/metrics.md` | every measured number, terse |

## Provenance

Built on `shared-core-v1` (`0dce46b`), a verified numerical core shared with a
separate study on parallel value iteration. The solver, maze benchmark and
correctness gates are shared engineering; the inventory MDP, the ε policies, the
preregistration and every result here are this project's.
