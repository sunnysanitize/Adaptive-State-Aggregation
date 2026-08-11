| Parameter | Value | Source |
|:--|:--|:--|
| Discount factor, `γ` | `0.95` | Paper |
| Aggregate interval, `‖Aᵢ‖` | `5` | Paper |
| Global interval, `‖Bᵢ‖` | `2` | Paper |
| Step size, `αₜ` | `1 / √t` | Paper |
| `ε` | `0.5`; sweep over `{0.05, 0.1, 0.5}` | Paper |
| Cost normalization | `‖V*‖∞ = 100` | Paper |
| Initial value, `V₀` | `0` | Paper |
| Repetitions per configuration | `20`, paired seeds | Paper |
| VI tolerance for ground truth | `1e-10` | **Custom** |

---

## Design choices

| Choice | What the paper says | What I chose |
|:--|:--|:--|
| Action tie-break | Nothing | Lowest action index. Strict `<` on the running min. Arbitrary, but it has to be fixed or seeds do not reproduce. |
| Top-bin edge, `v == b2` | Algorithm 2's bins are half-open, `[b1 + (i−1)ε, b1 + iε)`. When `ε` divides `b2 − b1` evenly, `b2` falls in no bin at all. | Clamp the bin **index** to the last raw bin. Distance to that centre is `ε/2`, so the `2ε/(1−γ)` bound is untouched, and nothing is recorded. Dropping the state instead would leave a hole in the partition. |
| Constant `V`, `b1 == b2` | `Δ = 0`, so `⌈Δ⌉ = 0` and the loop makes no bins — `K = 0`, which is degenerate. | One group, centre `= b1` **exactly**, not the formula's `b1 + ε/2`. A constant `V` then lifts back to itself instead of picking up an `ε/2` offset. Not an edge case — it is iteration 1 of every run, since `V₀ = 0`. |
| More groups than `max_groups` | Nothing — the paper never caps `K`. | **Custom.** Clamp `K` to `max_groups`, which means widening to `ε' = (b2 − b1)/max_groups` and re-binning; sets `Partition.groups_clamped`. The ε-closeness guarantee stays true of the `ε'` actually used, which `eps_effective` reports. Merging the overflow into a final group would break the guarantee silently. Both this and the top-bin row are clamps; only this one changes the answer, so only this one is flagged. §5.3 reports how often it fires. |
| Scope of "bit for bit" | **Custom** — an implementation choice, not a paper ambiguity | Compiled mode only. The two exactness tests skip under `make debug`, reason printed. Numba fuses mul+add into an FMA where CPython will not. Belongs in the README. |
| Which spanning tree the maze is | §4.1 says only "there is a unique path from each position to the terminal state" — a spanning tree, but not *which distribution over* spanning trees. No algorithm is named. | Randomized depth-first search (`_carve`, a recursive backtracker). Chosen by default rather than by argument — see below. Every spanning-tree algorithm satisfies the stated spec, and `test_maze`'s two checks (reachability, seed-determinism) pass for all of them, so nothing in the project distinguishes them. **This choice measurably changes the headline number.** |

---

## Environment

| Item | Recorded value |
|:--|:--|
| CPython | `3.14.6` |
| NumPy | `2.4.6` |
| Numba | `0.66.0` |
| OS / CPU | macOS `26.5.2`, arm64, Apple M4 Pro |
| Cold start: first `make test` with a clean `__pycache__` | `0.48 s` (median of 3) |
| Warm `make test` | `0.47 s` (median of 3) |
| Does `cache=True` persist for closure kernels? | **No** |

### Numba findings

All three found while building the backup kernel.

**1. `cache=True` does nothing for these kernels.** Files are written, never
read back. Both kernels also share one cache file — same closure, same source
line. No cross-loading seen, and a test covers it.

| First call | Cold cache | Warm cache |
|:--|--:|--:|
| `backup_direct` | `103 ms` | `103 ms` |
| `backup_lifted` | `52 ms` | `51 ms` |

Not splitting the template over `155 ms`. The consequence: the adaptive loop
**must** warm the kernels before starting any timer.

**2. A warm cache silently disables `NUMBA_BOUNDSCHECK=1`.** Only freshly
compiled code gets the checks.

| Reading `a[99]` on a length-3 array | Result |
|:--|:--|
| Bounds checking on, warm cache | returns `2.5e-313` |
| Bounds checking on, cold cache | raises `IndexError` |

`make bounds` wipes `__pycache__` first. Without that the mode passes while
checking nothing.

**3. Read-only and writable arrays are different types to Numba.** The builder
freezes the model arrays; `V` and `W` stay writable. A flag mismatch triggers a
second full compile.

| Signatures compiled | Count |
|:--|--:|
| After a writable `V` | `1` |
| After a read-only `V` | `2` |

Warm with the arrays the loop actually passes, or the compile lands inside the
timed section anyway.

---

## Maze reproduction

### Main comparison

Paper values are Table 1, *Std.* rows. **The paper's `±` is a 95% CI over 20
runs, not a standard deviation** — `±0.16` implies a run-to-run `σ ≈ 0.37`. Mine
is a standard deviation, so the two columns are not directly comparable and the
spread gap below is larger than it looks.

| Problem | Size | Paper's ℓ∞ error | Reproduced (σ) | Ratio |
|:--|--:|--:|--:|--:|
| Standard maze | `100²` | `1.43 ± 0.16` | *not run* | — |
| Standard maze | `200²` | `1.39 ± 0.15` | `1.8481 ± 0.0012` | `1.33×` |
| Standard maze | `300²` | `1.42 ± 0.20` | *not run* | — |
| Standard maze | `500²` | `1.11 ± 0.16` | `1.8481 ± 0.0012` | `1.66×` |
| Standard maze | `1000²` | `1.40 ± 0.16` | `1.8481 ± 0.0012` | `1.32×` |

At `ε = 0.5`, `1000` iterations, `20` maze instances per size (seed *i* gives
maze *i* and sampling stream *i*, paired across arms). Both sizes are **inside
the factor-of-2 pass band**, and the `1000²` figure is close.

Workload scales as it should — `11.5M`, `71.5M`, `286M` billed operations for
`40k`, `250k`, `1M` states — so these are genuine runs at each size:

| Size | `\|S\|` | Billed ops | Median timed loop |
|:--|--:|--:|--:|
| `200²` | `40,000` | `11,477,479` | `0.114 s` |
| `500²` | `250,000` | `71,537,479` | `0.696 s` |
| `1000²` | `1,000,000` | `286,037,479` | `2.695 s` |

### The error does not move with maze size

`1.8481 ± 0.0012` at **all three sizes**, to five significant figures. That is
not a copy-paste and not a stale cache — see the workload table above. It is a
property of this benchmark family, and it is the single most important thing on
this page.

With uniform step cost, `V(d) = c(1−γᵈ)/(1−γ)` depends only on the distance `d`
to the goal. Every size saturates at the same `c/(1−γ)`, so the `‖V*‖∞ = 100`
rescaling lands on the same constant — measured scale `5.0000000005` at every
size — and the near-goal value ladder comes out **identical**:

```
V* ascending:  −4.558705,  0,  1.086698,  6.42729,  11.47953,  16.258986,  …
```

byte-identical at `200²` and `500²`. Raw floating-point uniqueness overstates
the meaningful diversity in the saturated tail; at `ε = 0.5`, the exact `V*`
partitions have only `65.30`, `65.35`, and `65.35` occupied groups on average at
`200²`, `500²`, and `1000²`. The ℓ∞ error is attained on the near-goal ladder,
which does not depend on size or instance, and so neither does the error.

**The paper's error is flat with size too** — `1.43`, `1.39`, `1.42`, `1.11`,
`1.40` across `100²` … `1000²`, where `1.11` is the low outlier rather than the
start of a trend. So size-invariance **reproduces**; it is a property of the
benchmark that the paper's own numbers also show, and the mechanism above
explains why. What does not reproduce is the *spread*, not the trend.

### ε sweep

`scripts/sweep.py`, `20` paired maze instances per arm, `500²`, `γ = 0.95`,
`1000` iterations. Figure:
`results/figures/sweep_maze_500_both_{light,dark}.png`.

| `ε` | ℓ∞ error | Policy loss | `K` | Bound `2ε/(1−γ)` | Error ÷ bound |
|--:|--:|--:|--:|--:|--:|
| `0.05` | `0.1933 ± 0.0000` | `0.210 ± 0.069` | `106.65` | `2` | `0.097` |
| `0.1` | `0.3626 ± 0.0000` | `0.416 ± 0.052` | `93.00` | `4` | `0.091` |
| `0.5` | `1.8481 ± 0.0012` | `2.304 ± 0.608` | `61.10` | `20` | `0.092` |

Error scales with `ε`, and near-linearly: a `2×` increase in
`ε` gives `1.88×` the error, a `5×` increase gives `5.10×`. The measured error
sits at a near-constant `9%` of the bound across the whole grid, so the bound is
roughly **11× looser** than what actually happens — consistent with the ~15×
estimated below from the paper's own numbers.

Median timed loop: `0.73 s` across the grid (JIT warm-up and policy evaluation
excluded).

**Reported `±` is a standard deviation over 20 maze instances, and it is
essentially zero for ℓ∞ error at every `ε`.** Policy loss over the same 20
instances varies substantially, which is the evidence that the seeds are live
and the instances genuinely differ. See *Saturated `V*`* under Limitations for
why the two metrics behave so differently.

> **Reference point:** At these parameters, the theoretical bound
> `2ε / (1 − γ) = 20`. Compared with the paper's observed error of about `1.3`,
> the bound is roughly **15× looser**.

---

## Pre-freeze benchmark adequacy audit

`scripts/audit_value_space.py` audits all 20 paired maze seeds. At the paper's
`γ = 0.95`, the effective discount horizon is `1/(1−γ) = 20` support-graph
steps, while typical maze distances are thousands to hundreds of thousands of
steps. The value function therefore saturates near its maximum long before maze
size is exhausted.

| Size | `K` at `ε=0.5` | Within `0.5` of max | Largest group | Median group | Median distance |
|:--|--:|--:|--:|--:|--:|
| `200²` | `65.30` | `99.698%` | `99.575%` | `1` | `7,712` |
| `500²` | `65.35` | `99.952%` | `99.932%` | `1` | `38,018` |
| `1000²` | `65.35` | `99.988%` | `99.983%` | `1` | `129,758` |

The headline figure is
`results/figures/value_space_saturation.png`. It makes the mechanism explicit:
`|S|` grows by `25×`, but the aggregate phase continues to expose only about 65
independent group updates. This does not prove that multithreaded VI is better;
it establishes the benchmark property that Project I must account for.

An untimed joint parameter audit selected a non-saturated sensitivity arm before
parallel measurements: `γ = 0.999`, `ε = 0.005`, `max_groups = 32768`, and
`15000` iterations. On the `200²`, seed-0 feasibility instance, the exact `V*`
partition has `K = 4156`, its largest group contains `15.38%` of states, and no
group clamp fires. At iteration 15000, serial VI has `err_inf < 2e-9` and serial
adaptive aggregation has `err_inf = 1.2272`; both therefore reach the intended
`err_inf < 2` comparison target. Paired-seed confirmation and all parallel
timings remain Project I work.

---

## Uncounted work in the paper's metric

The paper's backup accounting excludes two operations:

1. **Rebinning:** `|S|` operations when entering an aggregate phase.
2. **Lifting:** `|S|` operations when leaving an aggregate phase.

| Measurement | Result at `200²`, `ε = 0.5` |
|:--|:--|
| `K` | `61` |
| Operations billed by the paper's metric | `11,477,479` |
| Operations actually performed | `22,917,479` |
| Uncounted share | **`99.67%`** |
| `K` at `500²`, `ε = 0.5` | `61.1` mean maximum occupied groups |
| Share of total **runtime** | Deferred to Project I's phase-level parallel accounting |

The uncounted work is almost exactly equal to the billed work. The arithmetic:
`143` cycles × `40,000` for rebinning, plus `143` × `40,000` for lifting, is
`11.4M` operations against `286` global sweeps × `40,000` = `11.44M` billed —
while aggregation contributes only about `37k` backups at `K ≈ 61`. Under the
paper's accounting the aggregate phase looks nearly free; measured, the two
`|S|`-sized passes it forces per cycle cost as much as the entire billed
workload.

This shows up already on the maze; Project I measures its runtime consequence.

The count above is an **operation count, not a runtime split**. The standalone
parallel study owns the phase-level runtime measurement.

JIT warm-up is excluded from all reported wall-clock measurements. Confirm this
before publishing the overhead percentage.

---

## Limitations and notes

### What I could not reproduce

**The paper's `± 0.16` spread.** Not reproducible from either source of
randomness, and both were tried:

| What varies | ℓ∞ error spread over 20 runs |
|:--|--:|
| Sampling stream only, one fixed maze | `0.000000` (bitwise identical) |
| Maze instance **and** sampling stream | `0.001249` |
| Paper | `0.16` |

`run.py` and `mdpagg.solve` both take `--problem-seed`, so 20 seeds now means 20
mazes with 20 ground-truth solves. It moved the spread from exactly zero to
`0.0012` — still two orders of magnitude short of the paper's. The cause is
*Saturated `V*`* below.

The paper's `σ ≈ 0.37` (from its 95% CI) against my `0.0012` is a factor of
`300`. This is the one genuinely unexplained gap in the maze reproduction. The
error level itself reproduces to within `1.3–1.7×`, and the size-invariance
reproduces; only the variance does not.

Note also that the paper measures the same configuration twice and disagrees
with itself: Table 1 gives `1.11 ± 0.16` for the `500²` standard maze, while
Table 2 (`p = 0.92`, `σ = 0.00`) gives `1.39 ± 0.19` for what reads as the same
setup. The two intervals barely overlap. My `1.848` is `1.33×` the Table 2
value.

### Notes

#### The maze generator is the likeliest source of the remaining gap

The paper gives exactly one quantitative handle on its mazes (§4.1): *"In a
10 × 10 standard maze, the initial tile … is often between 25 and 30 units away
from the destination tile."* Measured over `200` seeds, corner-to-goal distance
on a `10 × 10`:

| Generator | Median | IQR | `P(25 ≤ d ≤ 30)` |
|:--|--:|:--|--:|
| **Randomized DFS (ours)** | `44` | `[34, 54]` | `10%` |
| Randomized Prim | `18` | `[18, 20]` | `0%` |
| Wilson (uniform spanning tree) | `24` | `[20, 26]` | `28%` |
| Paper | *"often 25–30"* | — | — |

**What this does and does not establish.** Prim is effectively excluded; our DFS
is a poor fit at roughly double the paper's figure. But Wilson's median is `24`,
*below* the quoted band, and it lands inside only `28%` of the time — it is the
least-bad of three candidates, not a match. **The paper's generator is not
identified.** Aldous–Broder produces a distribution identical to Wilson's, so no
statistic can separate them; Kruskal, Eller, hunt-and-kill and recursive
division were not tested; and the whole comparison rests on one sentence of
prose, where "often" is not a statistic and both "units" and "10 × 10" admit
more than one reading.

**What is established, by direct measurement of our own code**, is that the
`_carve` DFS puts the goal at the end of a single-file corridor. States at BFS
distance `1…12` from the goal, `200²`, six seeds:

```
seed 0: 1 1 1 1 1 1 1 1 1 1 1 1
seed 1: 1 1 1 1 1 2 2 1 1 1 1 1
seed 3: 1 1 1 1 1 1 1 1 1 1 1 1
```

Exactly one state at nearly every near-goal distance. That is why the near-goal
value ladder is identical in every maze at every size, why those states form
size-one groups, and why the ℓ∞ error has no variance. It holds regardless of
what the authors did. A branchier tree — Wilson at `100²` gives `1 1 2 3 4 7 5
3 3 4 4 7`, varying per seed — would put several states at each near-goal
distance and let the extremum move between instances.

`maze.py` is frozen after Gate 3 by design, so changing `_carve` invalidates the
ε sweep and both size comparisons. That re-run is scripted and takes roughly 40
minutes; the decision is deliberately not taken here.

#### Saturated `V*` makes the `200²` maze a weak test instance

Measured on the cached ground truth:

| Property of `V*` at `200²` | Value |
|:--|--:|
| `min` / `max` | `−4.56` / `100.00` |
| Mean | `99.95` |
| Fraction of states within `0.5` of `max` | **`99.698%`** mean over 20 seeds |
| Largest group at `ε = 0.5` | `39,830` states mean (`99.575%`) |
| Singleton groups at `ε = 0.5` | `70.5%` of occupied groups |

With uniform step cost and `γ = 0.95`, `V(s) → c/(1−γ)` for any state more than
about `100` steps from the goal, and in a `200²` maze nearly every state is. So
`V*` is flat almost everywhere and only `65.3` ε-sized bins are occupied on
average at `ε = 0.5`.

Three consequences, all of which showed up in the runs:

1. **`K` is tens, not the low hundreds** the `‖V*‖∞ = 100`, `ε = 0.5 → K ≤ 200`
   estimate suggests. `K = 61` is not a binning bug; there is nothing else to
   bin.
2. **Near-zero variance in ℓ∞ error, at any size.** The max error is attained at
   state `1`, which sits in a **size-one group**. A group of one has nothing to
   sample, so the draw cannot change which state is backed up, and that state's
   trajectory — and hence `‖V − V*‖∞` — is identical for every sampling seed.
   Only `30` states differ between sampling seeds at all, by at most `0.51`.
   Changing the *maze* does not help either: the near-goal value ladder is
   size- and instance-independent (see *The error does not move with maze size*),
   so the error lands on the same value. Policy loss does vary — `± 0.699` at
   `200²`, `± 0.832` at `1000²` — even though it is *also* a sup-norm
   (`max_norm(V^π − V*)`), because the greedy policy it evaluates is decided by
   the whole value vector rather than by the extremum alone.
3. **Aggregation is flattered.** One group covering more than `99.5%` of the state space
   is nearly free to represent accurately, so this instance makes the method
   look better than an instance with a spread-out `V*` would.

None of this invalidates Gate 3 — error still scales cleanly with `ε`. But it
does mean **ℓ∞ error on the standard maze is very nearly a deterministic
function of `ε` and `γ`, not a measurement of the algorithm on an instance.**
The quantity the paper reports with a `± 0.16` is, in this implementation, a
constant. Policy loss does respond to the instance and is the more informative
metric here, though it is a sup-norm too and inherits some of the same
weakness.

Confirmed at `500²` and `1000²`: saturation gets *more* severe with size, not
less. The separate residual-ε project calibrates its inventory grid from the
inventory `V*` histogram for exactly this reason—the inventory value function
is expected to be roughly convex rather than maze-saturated.
