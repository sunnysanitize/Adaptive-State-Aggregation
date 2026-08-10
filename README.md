# Adaptive state aggregation shared core

Verified numerical foundation for experiments with adaptive state aggregation in
discounted tabular Markov decision processes. The implementation reproduces the
standard-maze setup from [Chen et al. (2021)](https://arxiv.org/abs/2107.11053)
and provides the common base for two independent studies.

The shared core contains:

- sparse tabular MDP construction and validation;
- exact Jacobi value iteration;
- value-based state partitioning;
- alternating global and aggregate updates;
- deterministic random streams, operation accounting, traces, and cached ground truth;
- paper-faithful standard-maze configurations at `200²`, `500²`, and `1000²`.

Experimental result files and ground-truth caches are deliberately regenerated
and ignored by Git. Private planning documents and working reproduction notes
under `docs/` and `docs_private/` are also intentionally untracked.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
make all
```

`make all` runs linting, type checking, compiled tests, pure-Python tests, and a
cold bounds-checked Numba run. On the frozen development environment the suite
contains 52 tests.

## Run one config

Ground truth is never computed implicitly:

```bash
.venv/bin/python -m mdpagg.solve configs/maze_200.json
.venv/bin/python -m mdpagg.run configs/maze_200.json
```

Run the full paper baseline, value-saturation audit, and figures with:

```bash
scripts/reproduce_baseline.sh
```

The full reproduction solves 20 maze instances per configuration and can take a
substantial amount of time. Outputs are written beneath `results/`.

## Research split

Once the baseline and benchmark audit are frozen, the tag `shared-core-v1` is
the common ancestor of:

- `parallel-study`: when multithreaded VI becomes preferable to adaptive aggregation;
- `residual-epsilon`: residual-driven aggregation evaluated on inventory control.

Each publication artifact records the shared tag and commit while keeping its
research question, experiments, and writeup independent.
