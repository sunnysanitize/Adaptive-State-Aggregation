# Residual-Driven State Aggregation for Multi-Asset Inventory Control

This repository asks whether state aggregation works better when its resolution
responds to the Bellman residual. More specifically, it tests whether a solver
benefits from watching the residual as it refines its state groups, or whether a
fixed refinement schedule works just as well.

On the inventory-control problem studied here, the answer is no. Once the solver
has converged, the residual-driven rule and a rate-matched geometric schedule
produce effectively the same result. Their final paired difference is
`-2.9e-06` on a scale where `||V*||inf = 100`. The practical conclusion is that
the benefit comes from how quickly the aggregation width shrinks, not from using
the residual as feedback.

This is a deliberately narrow conclusion from a preregistered experiment. The
study covers one multi-asset inventory problem, one solver schedule, and 20
paired seeds. It also includes a preregistered timing endpoint that turned out
to be poorly chosen. The full record, including that mistake and its effect on
the interpretation, is in
[`docs/residual_epsilon_note.md`](docs/residual_epsilon_note.md).

## What the study tests

Adaptive state aggregation alternates full Bellman sweeps over every state with
cheaper sweeps over groups of states that have similar values. The aggregation
width, written here as `epsilon`, controls how coarse those groups are. A larger
width produces fewer groups and cheaper updates, but it loses precision. A
smaller width produces more precise groups at a higher computational cost.

The residual-driven method updates the width at the beginning of each aggregate
phase according to this rule:

```text
epsilon = max(epsilon_min, c * span(TV - V))
```

The intuition is straightforward. When the Bellman residual is large, the value
estimate is still crude, so fine groups may be unnecessary. As the residual
contracts, the method gradually creates finer groups. The experiment tests
whether that feedback is actually useful.

## How the comparison works

Every run uses the same MDP, sampling seeds, sweep schedule, iteration count,
trace schedule, and group limit. Only the rule for `epsilon` changes. The fixed
baseline holds `epsilon` at `0.05`, `0.1`, or `0.5`. The residual configuration
uses the feedback rule above. Two geometric configurations shrink `epsilon` on
a preset timetable without reading the residual.

The fast geometric configuration is the most important control. It starts and
finishes at the same widths as the residual rule and refines at approximately
the same rate. The only meaningful difference is that one configuration reads
the Bellman residual while the other follows a schedule. This separates the
effect of feedback from the more ordinary benefit of annealing.

The slow geometric configuration follows the literal schedule in the original
plan and spreads its decay over all 1,429 aggregate cycles. It is useful for
showing why matching the annealing rate matters, but it is not the main test of
the feedback hypothesis. Altogether, the experiment contains six configurations
across 20 paired seeds, for 120 runs.

## Results

The table below reports median infinity-norm error over the 20 seeds for the
three configurations that form the central comparison.

| Configuration | 20 ms | 50 ms | 100 ms | 400 ms | Final |
|:--|--:|--:|--:|--:|--:|
| Fixed, `epsilon = 0.05` | 0.7802 | 0.1342 | 0.1479 | 0.1403 | 0.1379 |
| Residual | 0.6314 | 0.1344 | 0.1486 | 0.1407 | 0.1388 |
| Geometric fast | 0.6762 | 0.1344 | 0.1486 | 0.1405 | 0.1388 |

From 50 ms onward, the three configurations agree to three or four significant
figures. At the final iterate, the median paired difference between the residual
and fast geometric configurations is `-2.9e-06`, with a 95 percent bootstrap
confidence interval of `[-4.116e-04, 2.127e-07]`. That interval is entirely
inside the preregistered null region of `+/-0.02`.

The preregistration selected 20 ms as the primary endpoint, but that endpoint
falls in the steep part of the solver's descent rather than near its final error
floor. Small timing changes therefore select noticeably different iterates.
Even repeated runs on the same machine changed the reported confidence interval
at that budget. The residual-versus-geometric comparison was indeterminate at
20 ms, while every comparison from 50 ms onward and the deterministic final
iterate were null.

The 20 ms result remains in the repository because it was preregistered. It has
not been silently replaced with a more favorable endpoint. The conclusion is
instead based on the stable portion of the experiment, where observing
`span(TV - V)` adds no measurable value over a matched timetable.

## The inventory problem

The benchmark is a discounted multi-asset market-making inventory MDP. A state
is a signed inventory vector `(q1, ..., qN)`, with each position bounded between
`-Q` and `Q`. The primary instance uses three assets and `Q = 10`, which produces
`21^3 = 9,261` states. Each state has five quote-aggressiveness actions, and one
period has `2N + 1` possible successors representing no fill or a fill on either
side of one asset.

The cost combines an inventory-risk penalty, `lambda * q^T Sigma q`, with the
spread captured from fills. The discount is `gamma = 0.95`, and costs are scaled
so that `||V*||inf = 100`. The covariance matrix is non-diagonal, which makes
the value of an inventory depend on how positions across assets are correlated.

The five-action space stays fixed as the number of assets grows. This is an
important part of the design. Giving every asset five independent action levels
would make the action count grow as `5^N`, limiting the usefulness of compressing
the state space. The complete formulation, frozen constants, and validation
evidence are in [`docs/inventory_design.md`](docs/inventory_design.md).

## Reproducing the study

The project requires Python 3.11 or newer and a Unix-like shell. Create a virtual
environment and install the development and plotting dependencies with:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
```

Run the full validation suite with:

```bash
make all
```

This command runs linting, type checking, compiled tests, pure-Python tests, and
a cold bounds-checked Numba run. Two exactness checks apply only to compiled
mode and explain why they are skipped in debug mode.

To reproduce the ground truth, policy baselines, and arm comparison, run:

```bash
scripts/reproduce_residual_epsilon.sh
```

The experiment takes a few minutes and writes its output to `results/`. The main
comparison is stored in `results/arms_inventory_n3.json`, while
`results/arms_inventory_n3_curves.json` includes the per-iteration traces. The
fixed-width reference is stored in `results/inventory_fixed_eps.json`, and the
policy baselines are stored in `results/inventory_baselines.json`.

Wall-clock columns depend on the machine and may differ from the values reported
above. Final-iterate results are deterministic for the configured seeds.

## Code and documentation

The solver implementation lives in `src/mdpagg/`, which contains the MDP
representation, value-iteration routines, state partitioning, and adaptive
solver. The frozen primary configuration is
`configs/inventory_n3.json`. Experiment orchestration lives in
`scripts/sweep.py`, and the end-to-end reproduction command is collected in
`scripts/reproduce_residual_epsilon.sh`.

The `tests/` directory contains the numerical, behavioral, and regression
checks. [`docs/residual_epsilon_note.md`](docs/residual_epsilon_note.md) contains
the preregistration, complete results, limitations, and conclusions.
[`docs/inventory_design.md`](docs/inventory_design.md) explains the benchmark
design, while [`docs/metrics.md`](docs/metrics.md) defines the recorded metrics.

## Validation and provenance

The exact value function is solved to `1e-10` and tied to the correct problem
instance. The greedy policy evaluator is checked end to end, three policy
baselines are verified to be pointwise worse than optimal, and the fixed control
reproduces bit for bit after the experimental configurations are added. Seed
variation is smaller than the gaps between adjacent fixed aggregation widths,
which shows that the experiment had enough resolution to detect a meaningful
difference. The experimental constants were also frozen before any
residual-configuration result was generated.

The project builds on the verified numerical core tagged `shared-core-v1`
(`0dce46b`). The solver, maze benchmark, and correctness checks are shared
engineering. The inventory MDP, epsilon policies, preregistration, and results
belong to this study.

The repository is released under the [MIT License](LICENSE).
