# Project II — measured metrics

Running log of numbers worth tracking. Detail and reasoning live in
`inventory_design.md`; this file is just the values.

Base: `shared-core-v1`. Last updated after 6.1.

---

## Instance sizes

| | `N=3, Q=10` | `N=4, Q=10` |
|:--|--:|--:|
| States | 9,261 | 194,481 |
| Pairs (`× 5` actions) | 46,305 | 972,405 |
| Successors per pair (`2N+1`) | 7 | 9 |
| CSR entries | 324,135 | 8,751,645 |
| Memory | 3.9 MB | **105.0 MB** |

`N=4` memory splits `35.0 MB` succ_state `int32` + `70.0 MB` succ_prob
`float64`. Matches the plan's estimate; confirms no `int64` promotion.

---

## Frozen parameters (5.3)

| Parameter | Value |
|:--|:--|
| `γ` | `0.95` |
| `λ` | `0.02` |
| `Σ` | unit diagonal, `ρ = 0.5` off-diagonal |
| `f` (fill probability) | `[0.1, 0.2, 0.4, 0.6, 0.8]` |
| `δ` (half-spread per fill) | `[1.0, 0.8, 0.5, 0.3, 0.15]` |
| `δ · f` (expected revenue) | `[0.10, 0.16, 0.20, 0.18, 0.12]` — peaks at `a=2` |
| Cost normalization | `‖V*‖∞ = 100` |

---

## Value distribution (`N=3, Q=10`, rescaled)

| ε | `K` | Largest group | Clamped |
|:--|--:|--:|:--|
| 0.05 | 533 | 0.91% | no |
| 0.1 | 397 | 1.04% | no |
| 0.5 | 145 | 2.92% | no |
| 1.0 | 88 | 4.73% | no |
| 2.0 | 50 | 9.26% | no |

| | |
|:--|--:|
| `V*` range | `[−0.89, 100.00]` |
| States within 0.5 of max | 0.022% |
| States within 1.0 of max | 0.022% |
| States within 5.0 of max | 0.086% |

**Saturation comparison** — maze sensitivity arm (`γ=0.999`, `ε=0.005`, `200²`):
largest group `15.38%`, `K = 4156`. Inventory largest group is `2.92%` at
`ε = 0.5`.

**Frozen ε grid (5.4): `{0.05, 0.1, 0.5}`** → `K = 533, 397, 145`.

---

## Correlation effect

| | `ρ = 0.5` | `ρ = 0` |
|:--|--:|--:|
| `V*(+7,+7,+7)` | 59.70 | 59.23 |
| `V*(+7,−7,+7)` | 19.50 | 59.23 |
| `K(ε=0.05)` | 533 | 261 |
| `K(ε=0.5)` | 145 | 149 |

`max |V*(ρ=0) − V*(ρ=0.5)| = 66.7` on a 100 scale. At `ρ = 0` the aligned and
hedged states are identical — the problem separates.

---

## Optimal policy (`ρ = 0.5`)

| `‖q‖₁` | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 |
|:--|:--|:--|:--|:--|:--|:--|
| Median action | 1 | 1 | 1 | 2 | 4 | 4 |
| States | 129 | 1,030 | 2,666 | 3,160 | 1,828 | 440 |

Action shares: `[0.000, 0.422, 0.236, 0.107, 0.236]` — 4 of 5 used, max 42%.

With a flat `δ` ladder instead: `acts = 1` at every `λ` and `γ` tried.

---

## Ground truth (5.4)

| | |
|:--|--:|
| Config | `configs/inventory_n3.json` |
| Sweeps to `1e-10` | 454 |
| Cost rescale factor | `0.528982` |
| `‖V*‖∞` after rescale | `100` |
| `V*` range | `[−0.89, 100.00]` |

**Policy path check:** `‖policy_value(greedy(V*)) − V*‖∞ = 1.746e-09`

Admissible bound is `≈ 3·tol/(1−γ) = 6e-09`, not `tol` — span-seminorm stopping
amplifies by `1/(1−γ) = 20` at `γ = 0.95`. Passes.

`V*` histogram, 10 equal bins over `[−0.89, 100.00]`:

| bin | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| states | 1979 | **3482** | 1826 | 890 | 512 | 246 | 158 | 92 | 56 | 20 |

## Policy baselines (5.5) — Gate 5

Gaps `V^π − V*`, rescaled `‖V*‖∞ = 100` scale, `results/inventory_baselines.json`:

| Baseline | sup | mean | median | min | actions used |
|:--|--:|--:|--:|--:|:--|
| Do-nothing (`a = 0`) | 20.213 | 1.574 | 0.802 | 2.33e-01 | `[9261, 0, 0, 0, 0]` |
| Cheapest this period (`a = 2`) | 8.537 | 0.566 | 0.316 | 7.87e-02 | `[0, 0, 9261, 0, 0]` |
| Linear hedge | **0.933** | 0.459 | 0.455 | 1.63e-02 | `[27, 316, 1854, 2716, 4348]` |
| *greedy(V\*), control* | *−3.7e-11* | *−0.000* | *−0.000* | — | — |

Every min gap is positive by nine or more orders of magnitude over solver
tolerance. Gate 5 passes on the formulation, not on numerics.

---

## Fixed-ε baseline (5.6)

`N = 3`, 10,000 iterations, 20 sampling seeds, final iterate,
`results/inventory_fixed_eps.json`:

| ε | `err_inf` median | IQR | mean | tail mean | `policy_loss` median | IQR | `K` |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0.05 | 0.1379 | 0.0001 | 0.1376 | 0.1335 | 0.0128 | 0.0000 | 531 |
| 0.1 | 0.3017 | 0.0009 | 0.2979 | 0.2886 | 0.0518 | 0.0000 | 401 |
| 0.5 | 1.5108 | 0.0441 | 1.5130 | 1.4657 | 0.3734 | 0.1131 | 144 |

| | |
|:--|--:|
| `groups_clamped` firings, all 60 runs | **0** |
| Separation, 0.05 → 0.1 | gap 0.1638 vs IQR 0.0009 — 180× |
| Separation, 0.1 → 0.5 | gap 1.2091 vs IQR 0.0441 — 27× |
| Wall per run (solver only) | 0.78–0.92 s |
| Distinct `policy_loss` values over 20 seeds | 6 / **1** / 20 at ε = 0.05 / 0.1 / 0.5 |
| Policy-eval cache hits (of 160 per ε) | 76 / 41 / 0 |

`policy_loss` is quantised — at ε = 0.1 every seed recovers the same greedy
policy, so its IQR is structurally zero. `err_inf` is the better primary endpoint
for the Phase 6 paired test.

---

## First fixed-ε run (ε = 0.5, 1000 iterations)

| | |
|:--|--:|
| `err_inf` | 1.5169 |
| `policy_loss` | 0.3337 |
| `K` (max) | 144 |
| Billed / actual backups | 2,747,737 / 5,396,383 |
| Overhead | 96.4% |
| Wall | 0.077 s |

---

## Residual-ε preregistration (6.1)

Frozen 2026-08-12 before any residual-arm result. Rationale in
`residual_epsilon_note.md`.

| Parameter | Value |
|:--|:--|
| `c` | `0.084` |
| `ε₀ = c · span₀` | `0.4985` |
| `ε_min` | `0.05` |
| Geometric arms | `C = 1429` (plan's), `C = 14` (rate-matched) |
| Primary endpoint | `err_inf` at 20 ms wall-clock |
| Budget ladder (ms) | 5, 10, 20, 50, 100, 200, 400, 800 |
| Null region | `\|median paired difference\| < 0.02` |

### Residual span

| | |
|:--|--:|
| `span₀` at first aggregate entry (t = 2) | **5.93429** — deterministic, arm-independent |
| Aggregate entries per 10,000-iteration run | 1,429 |
| Floor crossing, `c·span ≤ ε_min` | entry 14 (t = 100) via ε = 0.5; entry 12 (t = 86) via ε = 0.05 |

**Span plateaus at `κ·ε`, it does not decay to zero** (fixed arm, seed 0, 6,000
iterations, mean over entries after t = 3000):

| ε | 0.02 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| plateau span | 0.00786 | 0.01988 | 0.04202 | 0.10703 | 0.22968 | 0.47282 | 0.94495 |
| `κ = span/ε` | 0.393 | 0.398 | 0.420 | 0.428 | 0.459 | 0.473 | 0.473 |
| `err_inf` | 0.0525 | 0.1295 | 0.2798 | 0.7379 | 1.397 | 3.700 | 8.555 |

**Stability bound: `c < 1/κ_max = 2.11`.** Above it `ε ← c·span` diverges and
coarsens until `max_groups` clamps. `c = 0.084` is 25× below.

### Wall-clock and error vs iteration (fixed arm, seed 0)

| iteration | 100 | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ε = 0.5 wall (ms) | 8.0 | 15.7 | 39.4 | 78.6 | 157.5 | 417.4 | 810.1 |
| ε = 0.5 `err_inf` | 6.969 | 1.386 | 1.258 | 1.292 | 1.271 | 1.291 | 1.300 |
| ε = 0.05 wall (ms) | 9.3 | 18.4 | 46.0 | 91.9 | 184.6 | 480.4 | 952.6 |
| ε = 0.05 `err_inf` | 6.868 | 0.961 | 0.135 | 0.129 | 0.125 | 0.121 | 0.119 |

Coarse ε saves only **18%** wall-clock (810 vs 953 ms) despite `K = 145` against
`531`: global sweeps are ~90% of the work at `global_len = 2`, `agg_len = 5`.
ε = 0.05 dominates ε = 0.5 at every budget measured.

---

## Three-arm comparison (6.3)

`results/arms_inventory_n3.json`. 6 arms × 20 seeds × 10,000 iterations = 120
runs, 105 s solver time. `groups_clamped` fired **0** times.

Median `err_inf` by wall-clock budget:

| arm | 10 ms | **20 ms** | 50 ms | 100 ms | 400 ms | final |
|:--|--:|--:|--:|--:|--:|--:|
| fixed_0.05 | 5.795 | 0.7802 | 0.1342 | 0.1479 | 0.1403 | 0.1379 |
| fixed_0.1 | 5.806 | **0.5680** | 0.3075 | 0.3040 | 0.3018 | 0.3017 |
| fixed_0.5 | 4.413 | 1.505 | 1.504 | 1.501 | 1.502 | 1.511 |
| residual | 5.797 | 0.6314 | 0.1344 | 0.1486 | 0.1407 | 0.1388 |
| geometric_slow | 4.591 | 1.466 | 1.274 | 1.199 | 0.5275 | **0.1287** |
| geometric_fast | 5.799 | 0.6762 | 0.1344 | 0.1486 | 0.1405 | 0.1388 |

Preregistered contrasts:

| contrast | budget | median | 95% CI | verdict |
|:--|:--|--:|:--|:--|
| residual − fixed_0.05 | 20 ms | −0.1486 | [−0.1502, −0.0999] | outside null |
| residual − fixed_0.05 | final | +0.00081 | [+0.00013, +0.00090] | null |
| residual − geometric_slow | 20 ms | −0.8273 | [−0.8489, −0.8186] | outside null |
| residual − geometric_slow | final | +0.0101 | [+0.0093, +0.0101] | null |
| residual − geometric_fast | 20 ms | −0.0417 | [−0.0459, +0.0011] | indeterminate |
| residual − geometric_fast | final | **−2.9e-06** | [−0.00041, +0.0000002] | null |

**The 20 ms primary was badly chosen** — it sits in the steep transient, and the
`residual − fixed_0.05` sign flips across adjacent budgets: **+0.0197** at 10 ms
(residual worse), **−0.1486** at 20 ms (better), **−1.2e-05** at 50 ms (null).
From 50 ms on, residual / geometric_fast / fixed_0.05 agree to 3–4 significant
figures.

**Conclusion: observing `span(TV − V)` adds nothing over a matched geometric
timetable.** Detail in `residual_epsilon_note.md`.

### Clean-tree reproduction (Phase 7)

`scripts/reproduce_residual_epsilon.sh` from a `git archive` tree with no
ground-truth and no Numba cache:

| | |
|:--|:--|
| Ground truth | 454 sweeps, matches 5.4 |
| Gate 5 | PASS — 20.2125 / 8.5368 / 0.9332, matches 5.5 exactly |
| Fixed-ε baseline | 0.1379 / 0.3017 / 1.511, matches 5.6 |
| Final iterate, all 6 arms | **bit-identical** to the published run |
| Headline contrast @ final | `−2.94e-06`, CI `[−0.0004116, +2.127e-07]` — identical |
| 20 ms primary contrast | **moved**: CI `[−0.1502, −0.0999]` → `[−0.1530, −0.0100]` |

The last row is the measured case for the wall-clock endpoint being the wrong
primary: same command, same machine, CI upper bound an order of magnitude closer
to zero.

---

## Suite

| After task | `make test` | `make debug` |
|:--|--:|--:|
| 5.1 | 64 | 56 + 8 skipped |
| 5.2 | 70 | 62 + 8 skipped |
| 5.3 | 77 | 69 + 8 skipped |
| 5.4 | 80 | 72 + 8 skipped |
| 5.5 | 87 | 79 + 8 skipped |
| 5.6 | 95 | 87 + 8 skipped |
| 6.2 | 108 | 99 + 9 skipped |
| 6.3 | 108 | 99 + 9 skipped |

Baseline before Project II: 60. Lint and mypy clean throughout.

**Gate 6 (6.2).** The fixed arm is bit-identical to the frozen 5.6 baseline after
the geometric policy and the aggregate-cycle counter landed — `err_inf` matches to
every digit at ε ∈ {0.05, 0.1, 0.5} × seeds {0, 7}:

| ε | 0.05 | 0.1 | 0.5 |
|:--|--:|--:|--:|
| seed 0 | 0.137889711313 | 0.302296550095 | 1.506512254315 |
| seed 7 | 0.137903763191 | 0.301314078422 | 1.541478112631 |

---

## Still to fill in

| Task | Metric |
|:--|:--|
| 6.3 | Paired residual vs geometric vs fixed differences; `groups_clamped` rate |
