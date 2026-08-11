# What this study will measure, and what would prove it wrong

**Written `2026-08-10`, before any parallel code existed in this branch.**
`grep -rn prange src/` returns nothing today. That is the point: these choices
are only worth something if they were fixed before the first measurement.


## 1 The question

> As tabular MDPs get large, does multithreaded value iteration reach a target
> error faster than a well-optimized multithreaded aggregation solver?

Three results are kept separate. The first is the one that matters:

1. **Which solver is practically better.** VI reaches the target error in less
   wall-clock time than aggregation, each solver using whatever thread count
   suits it best.
2. **Why.** VI gains more from added threads, because it has `|S|` independent
   backups to hand out where the aggregate phase has only `K`.
3. **How far the answer goes.** It applies to the maze sizes, parameters,
   hardware and threading layer measured here, and no further.

## 2 Terms

| Term | Meaning |
|:--|:--|
| `err_inf(t)` | `max_norm(V_t − V*)`, the worst-case error of the current values |
| `T(target)` | Solver time at the first iteration where `err_inf ≤ target` |
| `speedup(p)` | `T₁ / T_p` — how much faster `p` threads are than one |
| `efficiency(p)` | `speedup(p) / p` — how much of each added thread is useful |
| `p*` | The thread count with the lowest median `T(target)`, picked per solver |

`T` is solver time only. It leaves out JIT warm-up, and the code that records
error runs outside the clock, so measuring does not inflate the measurement.
Policy evaluation never runs during a timed trial.

Error stays flat during an aggregate phase and only steps when values are
lifted back to states. First crossing is therefore read off a recorded
iteration and never interpolated.

## 3 What the serial code already does

Produced by `scripts/baseline_targets.py`, one seed per configuration. Both
solvers run through the same loop, with aggregation switched off to get exact
value iteration, so their timing is measured the same way.

### The error each solver can actually reach

| Parameters | Size | VI final | Aggregation final | Aggregation best |
|:--|--:|--:|--:|--:|
| `γ = 0.95`, `ε = 0.5` | `200²` | `1.8e-09` | `1.848` | `1.565` |
| `γ = 0.95`, `ε = 0.5` | `500²` | `1.8e-09` | `1.848` | `1.565` |
| `γ = 0.95`, `ε = 0.5` | `1000²` | `1.8e-09` | `1.848` | `1.565` |
| `γ = 0.999`, `ε = 0.005` | `200²` | `1.3e-09` | `1.227` | `1.225` |

Aggregation bottoms out at `1.848` in the first three rows. **Any target below
about `1.6` is therefore out of reach for one of the two solvers**, and picking
one would decide the comparison by definition instead of by measurement.

### Time to reach `err_inf ≤ 2.0`, one thread

| Parameters | Size | VI iters | VI time | Agg. iters | Agg. time | Ratio |
|:--|--:|--:|--:|--:|--:|--:|
| `γ = 0.95` | `200²` | `76` | `0.0189 s` | `210` | `0.0224 s` | `1.185×` |
| `γ = 0.95` | `500²` | `76` | `0.1290 s` | `210` | `0.1448 s` | `1.122×` |
| `γ = 0.95` | `1000²` | `76` | `0.5133 s` | `210` | `0.5442 s` | `1.060×` |
| `γ = 0.999` | `200²` | `3910` | `0.9850 s` | `13203` | `1.6814 s` | `1.707×` |

Ratio is aggregation time divided by VI time. Above `1` means VI is faster.

Two things here shape the whole study, and both are written down now rather
than found later:

**VI already wins on one thread, at every size.** The question as posed — does
VI reach the target faster — is already answered yes, with no threading at all.
So this experiment cannot claim to have established it. What threads can show
is whether the gap gets *wider*.

**The gap narrows as the maze grows: `1.185 → 1.122 → 1.060`.** That points
against the "as MDPs get large" part of the question. On one thread,
aggregation is catching up as size grows, and continuing that trend has it
passing VI somewhere beyond `1000²`. The reason is visible in the iteration
counts, which are the same at all three sizes (`76` and `210`): both solvers
need a fixed number of iterations regardless of size, so the comparison comes
down to cost per iteration, and aggregation's grows more slowly.

That makes the real question sharper: **do threads reverse a trend that, on one
thread, runs the other way?**

## 4 The plan

### Configurations

| Parameters | `max_groups` | Sizes | Iterations |
|:--|--:|:--|--:|
| `γ = 0.95`, `ε = 0.5` | `4096` | `200²`, `500²`, `1000²` | `1000` |
| `γ = 0.999`, `ε = 0.005` | `32768` | `200²`, `500²` | `20000` |

The second row is the setting where values are spread out instead of bunched
near the maximum, so aggregation has many groups to work with. Its iteration
count is raised from `15000` to `20000` **for timed runs only**, because
aggregation first reaches the target at iteration `13203` — `88%` of a `15000`
budget. A target reached that close to the end turns any slowdown into "never
reached". Nothing else changes: same algorithm, same instance, same parameters,
just a longer run.

`1000²` at `γ = 0.999` is **not** part of the plan. It needs a fresh exact
solve of about `20000` sweeps over `10⁶` states for every seed. If it turns out
to be affordable it will be reported as an extra, and labelled as one.

### Thread counts

`1, 2, 4, 8, 10`. The machine is an Apple M4 Pro with `14` cores, `10` of them
performance cores. Stopping at `10` keeps every measurement on cores of the
same speed. A `14`-thread run is recorded for interest only and left out of
speedup and efficiency, since the four slower cores would move both numbers for
reasons that have nothing to do with the algorithms.

Thread count is set explicitly for every run, never left to a machine default.
Both the requested and the actual count go into each result file.

### Repetitions

- At least `7` timed runs per combination of parameters, size, solver and
  thread count.
- **Median** is the headline number, with the interquartile range beside it. A
  single scheduling hiccup should not move the result.
- Runs are batched so each timed block lasts at least `1.0 s`. At `γ = 0.95`,
  `200²` reaches the target in about `20 ms`, which is too short to measure
  against thread startup cost and timer noise, so those points are timed as
  repeated whole solves and divided.
- Maze and sampling seeds are paired across every configuration: `20` seeds at
  `γ = 0.95`, `5` at `γ = 0.999`.
- Kernels and the thread pool are warmed before every timed configuration.
- No policy evaluation inside a timed run. Error and policy loss are computed
  from the final values after the clock stops. Detailed curves come from
  separate untimed runs.

### The main comparison

At the largest size of each parameter set, each solver on its own best thread
count:

> **Supported** if the median `T_VI(2.0)` is below the median `T_agg(2.0)`, and
> the `95%` confidence interval on the per-seed difference does not include `0`.

### The others

- **Error at equal time.** Compare `err_inf` at `0.25, 0.5, 1.0, 2.0 s` for
  `γ = 0.95`, and `0.5, 1.0, 2.0, 4.0 s` for `γ = 0.999`.
- **Threading gain.** `speedup(10)` and `efficiency(10)` for each solver.
- **Trend with size.** The time ratio at `p*` as the maze grows.

## 5 What would prove each claim wrong

| # | Claim | Wrong if |
|--:|:--|:--|
| 1 | VI is practically better | At the largest size, median `T_VI(2.0)` is not below median `T_agg(2.0)`, or the confidence interval on the difference includes `0`. An interval containing `0` is reported as "no difference shown", not as support |
| 2 | "As MDPs get large" | The time ratio at `p*` **falls** as size grows, as it already does on one thread. VI winning at every size does not save this claim — the trend *is* the claim |
| 3 | Threads explain it | `efficiency(10)` for aggregation is at least as high as for VI |
| 4 | Error at equal time | Aggregation matches or beats VI's error at most of the listed time budgets |
| 5 | The comparison is fair | Any parallel kernel fails to reproduce its serial output exactly, at any thread count |

Claim 2 is the one most likely to fail, for the reason given in section 3. That
would still be worth publishing: it would say aggregation gets *more*
competitive as problems grow, and that the case for preferring VI on threaded
hardware is strongest at middling sizes.

## 6 Prediction, on the record

VI wins the main comparison at every size, by a wider margin than serial at
`200²` and a narrower one at `1000²`. Specifically: `efficiency(10)` between
`0.55` and `0.80` for VI, and between `0.10` and `0.35` for aggregation at
`γ = 0.95`, where the aggregate phase has only about `61` group updates to
spread across threads while VI has `|S|`. Claim 2 fails at `γ = 0.95`. It
should hold at `γ = 0.999`, where `4156` groups give the aggregate kernel
enough work to be worth handing to threads.

If the aggregate kernel turns out slower on many threads than on one, that is
recorded as evidence for this explanation, not thrown away. The end-to-end
aggregation runs then use whichever of the two is faster, so the comparison is
between two threaded solvers rather than one being handed its slower kernel.

## 7 Limits

This applies to standard mazes at these two parameter settings, `|S|` from
`4×10⁴` to `10⁶`, one Apple M4 Pro, the threading layer recorded in each result
file, and the fixed-`ε` aggregation algorithm as implemented here.

At `γ = 0.95` the values bunch up near the maximum and the number of groups
barely grows with size. That is **part of the explanation and is reported as
such**, not left out in favour of a story about hardware alone.

A clean negative result is a complete result.
