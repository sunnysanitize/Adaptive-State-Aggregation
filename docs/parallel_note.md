# When multithreading makes value iteration better than state aggregation

A measured comparison of two solvers for large discounted tabular MDPs: exact
full-state Jacobi value iteration, and the fixed-`ε` adaptive state aggregation
algorithm. Both are multithreaded, and both were given comparable effort.

**The short answer.** Value iteration reaches a fixed error target faster than
aggregation at every size and thread count tested, by `1.3×` to `2.2×`.
Multithreading widens that lead but does not create it — value iteration was
already ahead on one thread. The advantage does **not** grow monotonically with
problem size at the paper's parameters: it peaks at `500²` and falls again at
`1000²`. Both solvers are limited by memory bandwidth long before they run out
of parallel work, reaching only `0.11–0.36` parallel efficiency on ten cores.

Everything below regenerates from the commands in
[Reproducing](#reproducing). What was predicted before any of it was measured
is in [`predictions.md`](predictions.md), written and dated in advance.

---

## 1 The benchmark

The standard maze from [Chen et al. (2021)](https://arxiv.org/abs/2107.11053):
a maze with a unique path from every tile to the goal, uniform step cost, and
an intended move taken with probability `p = 0.92`, otherwise a uniformly
random available direction. Costs are rescaled so `‖V*‖∞ = 100`.

| Parameter | Value |
|:--|:--|
| Discount `γ` | `0.95` (paper), `0.999` (second regime) |
| Aggregation width `ε` | `0.5` (paper), `0.005` (second regime) |
| Global / aggregate schedule | `2` global, `5` aggregate |
| Step size | `1/√t` |
| Sizes | `200²`, `500²`, `1000²` — `4×10⁴` to `10⁶` states |

### The reproduction this rests on

Before any timing, the implementation was checked against the paper's reported
error at fixed `ε`:

| Size | Paper ℓ∞ error | Reproduced |
|:--|--:|--:|
| `200²` | `1.39 ± 0.15` | `1.8481 ± 0.0012` |
| `500²` | `1.11 ± 0.16` | `1.8481 ± 0.0012` |
| `1000²` | `1.40 ± 0.16` | `1.8481 ± 0.0012` |

Within a factor of `1.7`, and error scales cleanly with `ε` — `0.1933`,
`0.3626`, `1.8481` at `ε = 0.05, 0.1, 0.5`, close to linear. The reproduction is
sound enough to build timings on.

Two things did not reproduce, and both are load-bearing here.

**The error does not vary between instances.** The paper reports `± 0.16`;
measured spread is `± 0.0012`, over two hundred times smaller. The cause is
structural: the maze generator puts the goal at the end of a single-file
corridor, so exactly one state sits at each near-goal distance, the near-goal
value ladder is identical in every maze at every size, and the worst-case error
lands on the same value every time. The paper's generator is not identified —
its one quantitative clue is a sentence about typical distances in a `10 × 10`
maze — so this is recorded as an unexplained gap rather than resolved.

**The value function is saturated.** With `γ = 0.95`, the effective horizon is
`20` steps while typical maze distances are thousands. `V*` is therefore flat
almost everywhere:

| Size | States within `0.5` of max | Occupied groups `K` at `ε = 0.5` | Largest group |
|:--|--:|--:|--:|
| `200²` | `99.698%` | `65.3` | `99.575%` |
| `500²` | `99.952%` | `65.35` | `99.932%` |
| `1000²` | `99.988%` | `65.35` | `99.983%` |

**`|S|` grows `25×` while `K` stays near `65`.** This is not a defect of the
implementation; it is a property of the benchmark, and it is central to the
result. It is why a second, non-saturated regime (`γ = 0.999`, `ε = 0.005`,
`K = 4156`) was measured alongside the paper's parameters — otherwise the study
would only ever describe a value function with almost nothing to aggregate.

## 2 What the paper's operation count leaves out

The paper bills work in backups: `|S|` per global sweep, `K` per aggregate
sweep. Two `|S|`-sized passes per cycle are not billed at all:

- **rebinning**, when an aggregate phase starts;
- **lifting** group values back to states, when it ends.

At `200²`, `ε = 0.5`:

| Measure | Value |
|:--|--:|
| Operations billed | `11,477,479` |
| Operations actually performed | `22,917,479` |
| Uncounted share | **`99.67%`** |

The unbilled work is almost exactly equal to the billed work. Under the paper's
accounting the aggregate phase looks nearly free; measured, the two `|S|`-sized
passes it forces per cycle cost about as much as the entire billed workload.
Section 4 shows this is not merely an accounting curiosity — rebinning is the
part of the solver that cannot be threaded.

## 3 How this was measured

**The target is `err_inf ≤ 2.0`**, where `err_inf = max_norm(V − V*)` — the
single worst state, not an average. The target was not free to choose:
aggregation bottoms out at `1.848` at the paper parameters, so anything below
about `1.6` is unreachable for one of the two solvers and would settle the
comparison by definition rather than by measurement. It was fixed before any
parallel timing existed.

**Both arms run through the same loop.** Setting the aggregate phase length to
zero turns the adaptive solver into exact value iteration — verified equal bit
for bit — so the two arms share their timing instrumentation rather than being
compared across different code paths.

**Timing is observer-free.** The iteration at which each arm reaches the target
is found in a separate, untimed pass. The timed run then executes exactly that
many iterations with no observer, so no `|S|`-sized error computation lands
inside the measurement. `err_inf` is verified after the clock stops, and a run
that failed to reach the target is an error rather than a data point. No policy
evaluation happens inside a timed run.

**Repetition.** At least `7` trials per configuration, more when a single trial
is short, up to `51`, so each set of timings covers at least a second of work.
Medians are reported; every individual trial time is kept in the raw files. JIT
compilation and thread-pool startup are warmed before the clock starts.

**Seeds are paired.** `20` maze instances at `γ = 0.95`, `5` at `γ = 0.999`,
with the same instances and sampling streams used for both arms. Confidence
intervals are percentile bootstrap on the **per-seed differences**, not
computed separately around each arm.

**Thread counts are set explicitly**, never inherited from the machine, and
both the requested and observed counts are recorded in every result file. The
ladder is `1, 2, 4, 8, 10`, stopping at the ten performance cores so every
point is measured on cores of the same speed.

### Threading changes speed and nothing else

Every threaded kernel is the same source compiled twice — the parallel loop
construct degrades to an ordinary loop when threading is off — so the two forms
cannot drift apart. Each splits a loop whose iterations write distinct outputs
and read only inputs, so no arithmetic is reordered and no sum is
reassociated.

This is tested rather than asserted: serial and threaded forms are compared for
**exact equality**, at every thread count on the ladder, for the global sweep,
the aggregate sweep, lifting, exact value iteration, and the complete
end-to-end solve including operation counts. Confirmed again outside the test
suite at production scale — a `40,000`-state maze run serially and on eight
threads produced identical error traces across all `1000` iterations.

**Rebinning is deliberately left serial.** The aggregate step samples a state
by its *position* within a group, so a threaded scatter that let writes
interleave would still produce a valid partition, still pass every set-based
check, and silently run a different algorithm. No error tolerance could detect
that, so the ordering property is asserted directly instead.

### Choosing kernels fairly

Threading the aggregate sweep is a loss when there are few groups. Measured
across group counts from `4` to `65,536`:

| Groups `K` | Serial | Best threaded | Speedup |
|--:|--:|--:|--:|
| `64` | `1.3 µs` | 2 threads | `0.02×` |
| `1024` | `19.9 µs` | 2 threads | `0.30×` |
| `4096` | `82.4 µs` | 4 threads | `0.97×` |
| **`8192`** | `127.0 µs` | 4 threads | **`1.21×`** |
| `32768` | `491.2 µs` | 8 threads | `3.01×` |
| `65536` | `810.4 µs` | 10 threads | `3.99×` |

Threading pays only above `K ≈ 8192`. **Both measured regimes fall below that**
— `K ≈ 65` at `γ = 0.95`, `K = 4156` at `γ = 0.999` — so the threaded runs call
the *serial* aggregate kernel in both. Forcing the threaded kernel at `K ≈ 65`
would have handed aggregation a loop `50×` slower and rigged the comparison.
The crossover is a measured constant in the code, not a preference.

## 4 Results

### Primary endpoint: time to `err_inf ≤ 2.0`

Each solver at its own best thread count `p*`:

| Config | `|S|` | `p*` VI | `p*` agg | Ratio at `p*` | 95% CI on paired difference (s) |
|:--|--:|--:|--:|--:|:--|
| `γ=0.95` `200²` | `40k` | `4` | `4` | `1.317×` | `[+0.0041, +0.0044]` |
| `γ=0.95` `500²` | `250k` | `10` | `10` | `1.552×` | `[+0.0224, +0.0254]` |
| `γ=0.95` `1000²` | `1M` | `10` | `10` | `1.429×` | `[+0.0624, +0.0658]` |
| `γ=0.999` `200²` | `40k` | `4` | `4` | `2.098×` | `[+0.7342, +0.7636]` |
| `γ=0.999` `500²` | `250k` | `10` | `10` | `2.158×` | `[+2.5245, +2.5608]` |

**Every interval excludes zero and every difference is positive.** Value
iteration reaches the target sooner than aggregation at every configuration
measured.

Note that `p*` itself depends on size: four threads is best at `40,000` states,
ten at `250,000` and above. Past four threads, the smallest maze gets actively
*worse* — a sweep over `40,000` states is not enough work to pay for the thread
launch.

### Multithreading widens the lead but does not create it

| Config | Serial ratio | Ratio at `p*` |
|:--|--:|--:|
| `γ=0.95` `200²` | `1.183×` | `1.317×` |
| `γ=0.95` `500²` | `1.114×` | `1.552×` |
| `γ=0.95` `1000²` | `1.056×` | `1.429×` |
| `γ=0.999` `200²` | `1.757×` | `2.098×` |
| `γ=0.999` `500²` | `1.425×` | `2.158×` |

Value iteration was already faster on a single thread in every configuration.
**The honest claim is that multithreading widens an existing advantage, not
that it produces one.** This was measured and written down before the parallel
work began, precisely so the result could not later be presented as stronger
than it is.

### The advantage does not grow with size at the paper's parameters

Ratio at `p*` with bootstrap intervals:

| Config | Ratio at `p*` (95% CI) |
|:--|:--|
| `γ=0.95` `200²` | `1.317×` `[1.307, 1.326]` |
| `γ=0.95` `500²` | `1.552×` `[1.502, 1.593]` |
| `γ=0.95` `1000²` | `1.429×` `[1.417, 1.440]` |
| `γ=0.999` `200²` | `2.098×` `[2.080, 2.116]` |
| `γ=0.999` `500²` | `2.158×` `[2.147, 2.170]` |

At `γ = 0.95` the ratio **rises then falls**, and the intervals separate at both
steps. This is a measured peak at `500²`, not noise. The hypothesis that value
iteration's advantage grows as MDPs get large is **false at the paper's
parameters**.

At `γ = 0.999` it rises, intervals separating, while the *serial* ratio falls
over the same two sizes. Threading reversed the single-thread trend there. That
rests on two sizes only.

**Why the peak.** The ratio at `p*` is approximately the serial ratio times the
efficiency ratio, and those move in opposite directions:

| Config | Serial ratio | `effVI / effAgg` | Product | Observed |
|:--|--:|--:|--:|--:|
| `200²` | `1.183` | `1.17` | `1.38` | `1.317` |
| `500²` | `1.114` | `1.36` | `1.52` | `1.552` |
| `1000²` | `1.056` | `1.38` | `1.46` | `1.429` |

Aggregation's per-iteration cost keeps improving with size, while value
iteration's threading advantage improves and then flattens. Their product peaks
in the middle. **The most interesting finding here is arguably that
aggregation becomes *more* competitive per iteration as problems grow**, and
only its worse parallel scaling keeps value iteration ahead.

### Both solvers scale poorly, for the same reason

Parallel efficiency on ten cores:

| Config | VI | Aggregation |
|:--|--:|--:|
| `γ=0.95` `200²` | `0.14` | `0.12` |
| `γ=0.95` `500²` | `0.30` | `0.22` |
| `γ=0.95` `1000²` | `0.36` | `0.26` |
| `γ=0.999` `200²` | `0.14` | `0.11` |
| `γ=0.999` `500²` | `0.31` | `0.20` |

Ten cores buy at most `3.6×`. Two limits stack, and the order matters:

**Memory bandwidth binds first, and binds both arms.** Value iteration has
essentially no serial fraction — it is one sweep repeated, every state
independent — yet it reaches only `0.36`. Nothing about the algorithm explains
that. A backup reads scattered successor values and writes one output, so the
sweep moves far more memory than it does arithmetic, and adding cores past four
mostly adds contention. The isolated sweep kernel reaches `0.54` at `500²`
while the complete solve reaches `0.30`.

**A serial fraction binds aggregation on top of that.** Where the time goes on
one thread:

| Config | Global | Aggregate | Rebin | Lift |
|:--|--:|--:|--:|--:|
| `γ=0.95` `200²` | `65.9%` | `1.5%` | `22.0%` | `10.5%` |
| `γ=0.95` `500²` | `71.7%` | `0.3%` | `17.9%` | `10.2%` |
| `γ=0.95` `1000²` | `75.0%` | `0.1%` | `14.3%` | `10.6%` |
| `γ=0.999` `200²` | `54.7%` | `16.7%` | `19.7%` | `8.9%` |

Global sweeps and lifting thread cleanly. Rebinning cannot, for the sampling
reason in section 3. With rebinning and the serial aggregate kernel together,
aggregation's ceiling is `3.2×`, `3.8×` and `4.3×` at the three paper sizes
regardless of thread count — and `2.3×` at `γ = 0.999`, where the aggregate
phase is a much larger share. Measured efficiency reaches roughly `60–65%` of
those ceilings, which is about what applying value iteration's bandwidth limit
on top would predict.

**At `γ = 0.95` the aggregate phase is `0.1–1.5%` of runtime.** The phase the
algorithm is named for is a rounding error in its own cost. There are about
`65` group updates against `|S|` global backups and two `|S|`-sized maintenance
passes per cycle. This is the saturation of section 1 showing up directly in the
runtime.

## 5 What was predicted, and what the predictions got wrong

Predictions were written and dated before any solver had been timed on more
than one thread.

| Prediction | Outcome |
|:--|:--|
| Value iteration wins the endpoint | **Right** — all five configurations, every interval clear of zero |
| The advantage fails to grow with size at `γ = 0.95` | **Right** — ratio peaks at `500²`, intervals separate |
| The advantage grows with size at `γ = 0.999` | **Right, for the wrong reason** — see below |
| `efficiency_agg(10)` in `0.10–0.35` | **Right** — measured `0.11–0.26` |
| `efficiency_VI(10)` in `0.55–0.80` | **Wrong** — measured `0.14–0.36` |

Two failures are worth stating plainly rather than burying.

**The efficiency prediction was badly wrong.** `0.55–0.80` assumed value
iteration would scale nearly linearly because it exposes `|S|` independent
backups. It exposes them, and it still cannot use them: the limit is bandwidth,
not available parallel work. The whole framing of "value iteration has more
parallel work, therefore it scales better" is only half right — it does scale
better than aggregation, but far worse than the available work suggests.

**The right answer for the wrong reason.** The prediction that the advantage
would grow with size at `γ = 0.999` was justified by the claim that `K = 4156`
gives the aggregate kernel enough work to be worth threading. The
microbenchmark then put the crossover at `K ≈ 8192`, so the aggregate kernel
stays serial in that regime too. The prediction held; its stated cause did not.
The cause is instead the one in section 4 — the balance between per-iteration
cost and threading advantage — and it would have been easy, and wrong, to bank
the correct call without noticing.

## 6 Limitations

**Scope.** Standard-maze MDPs at two parameter settings, `|S|` from `4×10⁴` to
`10⁶`, one machine, one threading layer, and the fixed-`ε` aggregation
algorithm as implemented here. Nothing here speaks to other MDP families,
other hardware, or other aggregation rules.

**Saturation is part of the result, not a footnote.** At the paper's
parameters, `K` stays near `65` while `|S|` grows `25×`. Conclusions about
aggregation's parallel scaling there are conclusions about a value function
with almost nothing to aggregate. The `γ = 0.999` regime exists to give the
other case, and it is where aggregation looks relatively better.

**The second regime has only two sizes.** "Grows with size" at `γ = 0.999`
rests on `200²` and `500²`. A `1000²` point needs about `20,000` exact sweeps
over `10⁶` states per seed and was excluded in advance on cost grounds.

**Bandwidth was inferred, not instrumented.** The claim that both solvers are
memory-bandwidth-bound is the best explanation for value iteration reaching
`0.36` with no serial fraction, supported by the kernel-versus-solve gap
(`0.54` against `0.30`). No hardware counters were read. A cache-miss or
bandwidth measurement would settle it properly.

**Equal wall-clock error was not measured.** The secondary endpoint — compare
error at fixed time budgets rather than time at a fixed error — was
planned in advance and has not been run. Everything here is time-to-target.

**One machine.** All timings are from a single Apple M4 Pro. Parallel
efficiency is a property of the memory system as much as the algorithm, so
these numbers should be expected to move on different hardware — particularly
anything with substantially different memory bandwidth per core.

**Seed counts differ between regimes.** `20` paired seeds at `γ = 0.95` but
only `5` at `γ = 0.999`, so the second regime's intervals rest on less data.

**Instance-to-instance variance is near zero**, for the structural reason in
section 1. The confidence intervals here are consequently narrow, and they
describe repeatability on this benchmark rather than robustness across a
diverse family of problems.

## 7 Machine and software

| Item | Value |
|:--|:--|
| CPU | Apple M4 Pro |
| Cores | `14` logical, `10` performance |
| Threading layer | `workqueue` |
| Thread counts measured | `1, 2, 4, 8, 10` |
| CPython | `3.14.6` |
| NumPy | `2.4.6` |
| Numba | `0.66.0` |
| pydantic | `2.13.4` |
| OS | macOS `26.5.2`, arm64 |

The `14`-thread case was recorded only as a diagnostic and excluded from every
speedup and efficiency figure: the four efficiency cores run at a different
rate, and including them would move both statistics for reasons unrelated to
the algorithms.

## Reproducing

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
make all                      # lint, then compiled, pure-Python and bounds-checked tests

# Ground truth. Never computed implicitly by a timing run.
.venv/bin/python -m mdpagg.solve configs/maze_200.json
for s in 0 1 2 3 4; do
  .venv/bin/python -m mdpagg.solve configs/maze_500_sensitivity.json --problem-seed $s
done

# One-thread characterization
.venv/bin/python scripts/baseline_targets.py configs/maze_200.json
.venv/bin/python scripts/phase_split.py configs/maze_200.json
.venv/bin/python scripts/aggregate_grain.py

# The measurement
.venv/bin/python scripts/scaling.py configs/maze_200.json configs/maze_500.json \
  configs/maze_1000.json --seeds $(seq 0 19) --threads 2 4 8 10 \
  --out results/scaling_paper.json
.venv/bin/python scripts/scaling.py configs/maze_200_sensitivity.json \
  configs/maze_500_sensitivity.json --seeds 0 1 2 3 4 --threads 2 4 8 10 \
  --out results/scaling_sensitivity.json
.venv/bin/python scripts/analyze_scaling.py \
  results/scaling_paper.json results/scaling_sensitivity.json
```

The full measurement takes roughly `70` minutes and should run on an otherwise
idle machine. Every point regenerates from the raw JSON, which retains each
individual trial time rather than only the medians.
