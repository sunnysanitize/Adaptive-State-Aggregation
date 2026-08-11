# When multithreading makes value iteration better than state aggregation

Does multithreaded full-state value iteration reach a target error faster than
a multithreaded adaptive state-aggregation solver, and does its advantage grow
as tabular MDPs get large?

**Measured answer: yes to the first, no to the second.** On standard-maze MDPs
from `4×10⁴` to `10⁶` states, exact Jacobi value iteration reaches the target
sooner at every size and thread count tested, by `1.3×` to `2.2×`. But the
advantage does **not** grow monotonically with size at the original paper's
parameters — it peaks at `500²` and falls again at `1000²`. Multithreading
widens value iteration's lead; it does not create it, because value iteration
was already ahead on a single thread.

Both solvers are limited by memory bandwidth long before they run out of
parallel work, reaching only `0.11–0.36` parallel efficiency on ten cores.

Full report: [`docs/parallel_note.md`](docs/parallel_note.md). What was
predicted before any of it was measured, written and dated in advance:
[`docs/predictions.md`](docs/predictions.md).

## Headline result

Time to `err_inf ≤ 2.0`, each solver at its own best thread count, 20 paired
maze instances at `γ = 0.95` and 5 at `γ = 0.999`. Intervals are percentile
bootstrap on the per-seed differences.

| Config | `|S|` | Ratio at best threads | 95% CI on paired difference (s) |
|:--|--:|--:|:--|
| `γ=0.95` `200²` | `40k` | `1.317×` | `[+0.0041, +0.0044]` |
| `γ=0.95` `500²` | `250k` | `1.552×` | `[+0.0224, +0.0254]` |
| `γ=0.95` `1000²` | `1M` | `1.429×` | `[+0.0624, +0.0658]` |
| `γ=0.999` `200²` | `40k` | `2.098×` | `[+0.7342, +0.7636]` |
| `γ=0.999` `500²` | `250k` | `2.158×` | `[+2.5245, +2.5608]` |

Above `1` means value iteration is faster. Every interval excludes zero.

## Why the advantage stops growing

The benchmark's value function saturates. With `γ = 0.95` the effective horizon
is `20` steps while typical maze distances are thousands, so `V*` is flat almost
everywhere: `|S|` grows `25×` from `200²` to `1000²` while the number of
occupied aggregation groups stays near `65`. The aggregate phase — the one the
algorithm is named for — is `0.1–1.5%` of runtime at those parameters.

What actually limits the aggregation solver is **rebinning**, which cannot be
threaded without changing which states get sampled, and which the original
operation count does not bill at all. At `200²`, `99.67%` of the operations
performed go uncounted by that metric.

A second, non-saturated regime (`γ = 0.999`, `ε = 0.005`, `K = 4156`) is
measured alongside the paper's parameters, so the comparison is not only ever
made against a value function with nothing to aggregate.

## Trusting the comparison

- **Threading changes speed and nothing else.** Serial and threaded kernels are
  compared for *exact* equality — not a tolerance — at every thread count, for
  the global sweep, the aggregate sweep, lifting, value iteration, and the
  complete end-to-end solve including operation counts.
- **Kernel choice is measured, not preferred.** Threading the aggregate sweep
  only pays above about `8192` groups; both measured regimes fall below that, so
  threaded runs call the *serial* aggregate kernel. Forcing the threaded one at
  `K ≈ 65` would have handed aggregation a `50×` slower loop.
- **Timed runs carry no observer.** The iteration reaching the target is found
  in a separate untimed pass; the timed run then executes exactly that many
  iterations with nothing measuring inside it.
- **Ground truth is never computed implicitly.** A missing cache is an error, so
  no run can be scored against a `V*` from a different problem.
- **Thread counts are set explicitly** and both requested and observed counts
  are recorded in every result file.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
make all
```

`make all` runs linting, type checking, compiled tests, pure-Python tests, and a
cold bounds-checked run.

## Reproducing

```bash
scripts/reproduce_parallel.sh          # about 70 minutes, idle machine
QUICK=1 scripts/reproduce_parallel.sh  # reduced matrix, checks the pipeline only
```

Outputs land under `results/`: headline numbers in `results/scaling_summary.json`,
figures in `results/figures/`. Raw files retain every individual trial time, not
only the medians, so any interval can be recomputed without rerunning.

Timings are only meaningful on an otherwise idle machine.

## Scope

Standard-maze MDPs at two parameter settings, `|S|` from `4×10⁴` to `10⁶`, one
Apple M4 Pro, one threading layer, and the fixed-`ε` aggregation algorithm as
implemented here. The memory-bandwidth explanation is inferred from scaling
behaviour rather than read from hardware counters. `docs/parallel_note.md`
records the limitations in full, including what was planned and not measured.

The benchmark reproduces [Chen et al. (2021)](https://arxiv.org/abs/2107.11053);
[`docs/reproduction_note.md`](docs/reproduction_note.md) covers what reproduced,
what did not, and why.
