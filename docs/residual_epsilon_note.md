# Residual-driven state aggregation — preregistration (6.1)

**Preregistered 2026-08-12, before any residual-arm result exists.** No run of
`ResidualSpanEpsilon` on the inventory instance has been executed or inspected.
Every number below comes from the frozen 5.6 fixed-ε baseline or from the
arm-independent opening of the run; §"Where the numbers came from" says exactly
which, and how to regenerate each.

Built on `shared-core-v1`. The instance, the solver and the protocol are 5.1–5.6
unchanged — only the ε policy varies.

---

## Frozen parameters

| Parameter | Value | Fixed by |
|:--|:--|:--|
| `c` | **0.084** | §Calibrating `c` |
| `ε₀` = `c · span₀` | **0.4985** | Derived; `span₀ = 5.93429` is deterministic |
| `ε_min` | **0.05** | §Choosing `ε_min` |
| Stability bound on `c` | `c < 2.11` | §The stability constraint — **new** |
| Geometric arms | `C = 1429` and `C = 14` | §Why the geometric arm runs twice |
| Primary endpoint | `err_inf` at a **20 ms** wall-clock budget | §Endpoint |
| Budget ladder | 5, 10, 20, 50, 100, 200, 400, 800 ms | §Endpoint |
| Null region | `|median paired difference| < 0.02` | §Null region |
| Seeds | `0 … 19`, sampling stream only | 5.6, unchanged |
| Iterations | `10000` | 5.6, unchanged |
| Instance | `N = 3`, `Q = 10`, 9,261 states | 5.1–5.4, unchanged |

All ε values are on the rescaled `‖V*‖∞ = 100` scale.

## The four arms

Same MDP, same seeds, same schedule (`global_len = 2`, `agg_len = 5`), same
`max_groups = 4096`, same trace and checkpoint schedule. Only ε differs.

| Arm | ε rule | Role |
|:--|:--|:--|
| **Fixed** | `ε ∈ {0.05, 0.1, 0.5}`, constant | Control. Frozen at 5.6; not re-run. |
| **Residual** | `ε ← max(ε_min, c · span(TV − V))` at aggregate entry | Treatment. |
| **G-slow** | `ε_i = max(ε_min, ε₀·(ε_min/ε₀)^(i/(C−1)))`, `C = 1429` | The plan's literal open-loop control. |
| **G-fast** | Same formula, `C = 14` | Rate-matched open-loop control. |

`i` indexes aggregate cycles from 0. A 10,000-iteration run at cycle length 7
contains **1,429** aggregate entries, so `C = 1429` in G-slow spreads the decay
across the entire run.

All four share the same endpoints, `ε₀ = 0.4985` and `ε_min = 0.05`. The primary
contrast is therefore about *how* ε moves between two fixed endpoints, not about
where it starts or stops.

---

## Endpoint

**Primary: paired difference in `err_inf` at a 20 ms wall-clock budget.**
Secondary: the full budget ladder as an error-vs-time curve, and the final
iterate at 10,000 iterations.

Wall-clock rather than iteration count, for a measured reason. Coarse ε is
cheaper per aggregate sweep but barely cheaper per run: the full 10,000
iterations take 810 ms at ε = 0.5 (`K = 145`) and 953 ms at ε = 0.05
(`K = 531`) — 18% apart, not the 3.7× the group counts suggest. The alternating
schedule spends 2 of every 7 iterations on a full `|S| = 9,261` global sweep, so
global backups are roughly 90% of the work and the aggregate sweep's width
hardly moves the total. An iteration-count endpoint would credit coarse ε with a
saving it does not deliver.

**Why 20 ms.** Under the frozen constants, all ε motion is finished by
aggregate entry 12–14, which is iteration 86–100, which is **under 10 ms** — about
1% of the run. Beyond that every arm except G-slow is sitting at `ε_min`, and a
late budget compares four arms that have been identical for 99% of their elapsed
time. But the error has *not* resolved by then either (`err_inf ≈ 6.9` at
iteration 100, still mid-descent), so differences created inside the ε window
surface slightly later as a head start or a handicap. 20 ms is iteration ≈215–250,
past the initial descent and well above the plateau, where any such difference is
still legible. The ladder brackets it on both sides so the choice can be audited
rather than taken on faith.

Everything regenerates from the existing `wall_ns` trace column. No solver
change is needed for the measurement.

## Null region

**A paired difference is null when the 95% CI on its median lies entirely within
`±0.02` `err_inf`.**

Anchored on the frozen grid rather than guessed. One step on the 5.4 ε grid —
ε = 0.05 to ε = 0.1 — moves final `err_inf` by 0.1638. The threshold 0.02 is
about 12% of that step, and about 20× the largest seed IQR the fixed arm shows at
fine ε (0.0009 at ε = 0.1). So a difference inside `±0.02` is both smaller than an
eighth of the crudest available alternative — just picking a finer fixed ε — and
large enough not to be seed noise. An improvement below it is not worth a method
claim.

Confidence intervals go on the **paired per-seed differences**, not separately
around each arm, per 6.3.

---

## Predictions, 2026-08-12

Stated before any residual-arm result. Two of the three are pessimistic; the
plan explicitly permits a null or negative result as a complete one, and
recording the pessimism in advance is what stops a later null from being
reported as a surprise.

**1. Residual vs fixed-0.05 → null or negative.** Residual will not beat the
finest frozen fixed arm at any budget. On this instance coarse ε is worse at
*every* budget measured, not merely at convergence — at iteration 200 the fixed
arm reads 1.386 at ε = 0.5 against 0.961 at ε = 0.05 — and coarsening buys only
18% wall-clock. There is no window in which starting coarse pays for itself.
*Falsified if* residual beats fixed-0.05 by more than 0.02 at any preregistered
budget, CI clear of the null region.

**2. Residual vs G-slow → residual wins, and the win means nothing.** G-slow
anneals across all 1,429 cycles while residual reaches its floor at cycle ~13, so
residual is finer for essentially the whole run. This is a difference in
annealing rate, not in feedback. Recorded here so it cannot later be presented as
support for the hypothesis. *Falsified if* residual fails to beat G-slow — which
would indicate something wrong with the residual arm, not a finding about ε.

**3. Residual vs G-fast → null. This is the real test.** G-fast reaches `ε_min`
at cycle 13, matching residual's projected schedule, so the only remaining
difference is that one arm looks at `span(TV − V)` and the other reads a
timetable. Prediction: observing the residual adds nothing measurable.
*Falsified if* residual beats G-fast by more than 0.02 at the primary budget with
the CI clear of the null region — which would be the publishable result.

The hypothesis as written in `docs_private/overview.md` — "fixed ε reaches an
approximation plateau while residual-driven ε continues descending" — cannot hold
literally under any stable `c`; see the stability constraint below. Residual ε
descends to `ε_min` and stops. Its asymptote is the fixed-`ε_min` floor, already
measured at 0.1379. The revised, testable form of the hypothesis is prediction 3.

---

## Calibrating `c`

The plan requires the initial residual-driven ε to be comparable to the frozen
fixed-ε baseline. That is arithmetic here rather than estimation, because **the
initial span is deterministic**:

`span(TV − V)` at the first aggregate entry (t = 2) is **5.93429**, identical
across every ε and every seed. `V₀ = 0`, and with `global_len = 2` two global
sweeps run before ε is consulted for the first time, so the opening of the run is
arm-independent by construction.

`c = 0.084` therefore gives `ε₀ = 0.084 × 5.93429 = 0.49848`, within 0.3% of the
frozen baseline's ε = 0.5, with no seed variance. Two significant figures, chosen
once, not tuned per seed, dimension or covariance matrix.

A consequence worth stating: because `ε₀` is fixed by the first two global
sweeps, the residual and geometric arms are *identical* to each other and to
fixed-0.5 for iterations 0 and 1 of every run. Any difference between arms begins
at t = 2 exactly.

### The non-finite guard never fires

`ResidualSpanEpsilon` returns `eps_min` when `state.residual_span` is not finite,
and `test_residual_epsilon` covers that path. Under this schedule it is dead
code: `global_len ≥ 1` guarantees a global sweep sets `residual_span` before the
first `rebin()` at t = 2. The guard is retained as defence against a future
schedule change, and the test with it, but no run in this study reaches it. Its
behaviour is also backwards for a coarse-to-fine rule — it would start at the
*finest* ε — which is another reason not to let it fire silently.

## The stability constraint

**New, and not anticipated by the plan.** `c` is not only a scale factor; it
decides whether the rule converges at all.

The residual span does not decay to zero. It plateaus, at a level roughly
proportional to ε itself — because the adaptive iterate never reaches the true
fixed point, and the distance it settles at is set by the aggregation error.
Measured on the fixed arm across two decades (seed 0, 6,000 iterations, mean over
aggregate entries after t = 3000):

| ε | 0.02 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| plateau span | 0.00786 | 0.01988 | 0.04202 | 0.10703 | 0.22968 | 0.47282 | 0.94495 |
| span / ε | 0.393 | 0.398 | 0.420 | 0.428 | 0.459 | 0.473 | 0.473 |
| `err_inf` | 0.0525 | 0.1295 | 0.2798 | 0.7379 | 1.397 | 3.700 | 8.555 |

So `span ≈ κ·ε` with `κ ∈ [0.39, 0.48]`, and the rule `ε ← c·span` has a
self-consistent fixed point at `ε = c·κ·ε`. That gives a threshold:

- `c·κ < 1` — ε contracts each cycle and is driven to `ε_min`. **Stable.**
- `c·κ > 1` — ε grows each cycle, coarsening without bound until `max_groups`
  clamps and the run measures `eps_effective` rather than the ε named by the
  policy.

Taking the largest measured `κ = 0.473` gives a divergence threshold of
`c ≈ 2.11`. **`c = 0.084` sits 25× below it.** This is recorded as a constraint
on the parameter, not as a tuning result: any future `c` for this instance must
be justified against 2.11, and a `c` near or above it would invalidate the arm
rather than merely change it.

## Choosing `ε_min`

**`ε_min = 0.05`**, the finest value on the frozen 5.4 grid, a decade below `ε₀`.

Three reasons, in order of weight:

1. **Its floor is already known.** Fixed-0.05 was measured at 5.6: `err_inf`
   median 0.1379, IQR 0.0001, `K = 531`, never clamped. The residual arm's
   asymptote is therefore predicted *in advance* rather than discovered, which is
   what makes prediction 1 falsifiable rather than vague.
2. **It keeps the primary contrast clean.** An `ε_min` below the frozen grid
   would let residual finish finer than any fixed arm ever ran, and it would then
   beat fixed for a trivial reason — it ended somewhere else — rather than
   because feedback helped. Matched endpoints across residual, G-slow and G-fast
   remove that confound from the primary contrast entirely.
3. It is on the grid, so `K` and the clamping behaviour are known quantities and
   `groups_clamped` is expected to stay at zero, as it did across all 60 runs at
   5.6.

## Why the geometric arm runs twice

The plan adds a non-adaptive arm to "separate feedback from ordinary annealing" —
to answer whether watching `span(TV − V)` beats simply shrinking ε on a
timetable. Its literal schedule does not do that job here.

With `C = 1429` the decay is spread over the whole run, while residual reaches
its floor at cycle ~13. The two arms would differ by roughly two orders of
magnitude in annealing rate, and residual would win on that alone. The comparison
would be real but would answer a different question than the one the arm was
added for.

So both are preregistered:

- **G-slow, `C = 1429`** — the plan's arm, exactly as written. Kept because it
  was frozen before any of this was measured, and dropping a control because a
  later measurement makes it look unflattering is the move preregistration exists
  to prevent.
- **G-fast, `C = 14`** — reaches `ε_min` at cycle 13, matching residual's
  projected floor arrival, isolating feedback from rate.

`C = 14` comes from a projection, and that is disclosed rather than buried.
Applying `c = 0.084` to the span trajectories of the two frozen fixed arms puts
the floor crossing at aggregate entry **14** (ε = 0.5 trajectory, t = 100) and
entry **12** (ε = 0.05 trajectory, t = 86). `C = 14` places G-fast's arrival at
entry 13, the midpoint of that bracket. The residual arm's own trajectory will
differ from both once ε starts moving, so entry 13 is an estimate, not a
guarantee of an exact match.

This is a weaker form of preregistration than fixing a constant from the frozen
baseline alone: G-fast's parameter is set from a prediction about the treatment.
It is disclosed here, before the fact, and G-slow is retained precisely so one
control remains that was fixed with no knowledge of the treatment at all.

---

## Where the numbers came from

Every measurement above is from the **fixed** arm or from the arm-independent
opening of the run. Nothing was computed from `ResidualSpanEpsilon`.

| Number | Source |
|:--|:--|
| Final `err_inf`, IQR, `K`, clamping at ε ∈ {0.05, 0.1, 0.5} | `results/inventory_fixed_eps.json` (5.6, frozen) |
| `span₀ = 5.93429` | First aggregate entry, arm-independent |
| Plateau span / ε table | Fixed arm, seed 0, 6,000 iterations |
| Wall-clock 810 / 953 ms, error-at-iteration table | Fixed arm, seed 0, 10,000 iterations |
| Floor-crossing entries 12 and 14 | `c = 0.084` applied to fixed-arm span trajectories |

The last four are regenerated by instrumenting `run_adaptive` with an observer
that records `state.residual_span` at aggregate entries (`t % 7 == 2`) under
`FixedEpsilon`, on the cached `configs/inventory_n3.json` ground truth. They are
diagnostic measurements of the frozen baseline, not new experimental arms, and
they add no result to the record that 5.6 did not already contain.

## Scope

The claim this preregistration can support is **lower `err_inf` at equal
wall-clock over the measured budget range, on the `N = 3`, `Q = 10` inventory
instance**, and nothing wider. Specifically not claimed: behaviour at other `N`
(4 was cut at 5.6, deliberately), other `γ`, other covariance structures, other
schedules, or on the maze — where §"transfer check" in 6.3 may look, but which
cannot substitute for the inventory evidence.

The stability threshold `c < 2.11` and the ratio `κ ≈ 0.4` are properties of this
instance and this schedule. They are not claimed to hold generally, and a
different MDP would need its own `κ` measured before `c` could be transferred.

---

# Results

Everything below was produced after the constants above were frozen. Raw:
`results/arms_inventory_n3.json`, regenerated by

```bash
.venv/bin/python scripts/sweep.py configs/inventory_n3.json --arms --policy-loss-curve
```

120 runs — 6 arms × 20 seeds — each the full 10,000 iterations, 105 s of solver
time. `t_sa = 7143` on all 120, so no run terminated early.
**`groups_clamped` never fired**, so every arm measured the ε its policy named
rather than a widened `eps_effective`.

## Gate 6

The fixed arm reproduces the frozen 5.6 baseline bit-for-bit after the geometric
policy and the aggregate-cycle counter landed — `err_inf` identical to every
digit at ε ∈ {0.05, 0.1, 0.5} × seeds {0, 7}, e.g. `1.506512254315` at ε = 0.5,
seed 0. The counter increments on the fixed arm's aggregate entries too, so this
is a real claim rather than a tautology. **Gate 6 passes**; the three-arm control
is uncontaminated.

## The headline: the arms are indistinguishable from 50 ms onward

Median `err_inf` over 20 seeds, by wall-clock budget:

| arm | 5 ms | 10 ms | **20 ms** | 50 ms | 100 ms | 200 ms | 400 ms | 800 ms | final |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fixed_0.05 | 20.32 | 5.795 | 0.7802 | 0.1342 | 0.1479 | 0.1434 | 0.1403 | 0.1382 | 0.1379 |
| fixed_0.1 | 20.31 | 5.806 | 0.5680 | 0.3075 | 0.3040 | 0.2805 | 0.3018 | 0.3011 | 0.3017 |
| fixed_0.5 | 17.05 | 4.413 | 1.505 | 1.504 | 1.501 | 1.340 | 1.502 | 1.300 | 1.511 |
| residual | 20.35 | 5.797 | 0.6314 | 0.1344 | 0.1486 | 0.1342 | 0.1407 | 0.1386 | 0.1388 |
| geometric_slow | 16.93 | 4.591 | 1.466 | 1.274 | 1.199 | 0.9683 | 0.5275 | 0.1800 | 0.1287 |
| geometric_fast | 16.85 | 5.799 | 0.6762 | 0.1344 | 0.1486 | 0.1436 | 0.1405 | 0.1330 | 0.1388 |

From 50 ms — about 6% of a full run — residual, geometric_fast and fixed_0.05
agree to three or four significant figures and stay that way to the end. The
paired difference `residual − geometric_fast` at the final iterate is
**−2.9e-06**, on a scale where `‖V*‖∞ = 100`.

## The primary endpoint was badly chosen

**This is the most important thing in this section, and it is a criticism of the
preregistration, not a finding about the method.**

The 20 ms budget lands inside the steep part of the descent, where every arm is
falling by an order of magnitude between adjacent budgets. The measurement there
is dominated by exactly where the budget slices the curve, and the sign of the
primary contrast flips between adjacent budgets:

| `residual − fixed_0.05` | 10 ms | **20 ms** | 50 ms |
|:--|--:|--:|--:|
| median | **+0.0197** | **−0.1486** | −1.2e-05 |
| 95% CI | [+0.0031, +0.0318] | [−0.1502, −0.0999] | [−0.0001, +0.0002] |
| verdict | outside null — residual **worse** | outside null — residual **better** | null |

Taken literally, prediction 1's falsification condition — "residual beats
fixed-0.05 by more than 0.02 at any preregistered budget, CI clear of the null
region" — **fires at 20 ms**. But the identically-worded condition fires in the
*opposite direction* at 10 ms. A criterion that both confirms and refutes
depending on which adjacent budget is read is not measuring a durable property.

The error is mine and it is diagnosable. §Endpoint argued for 20 ms because it is
"past the initial descent" — it is not. At 20 ms the arms sit at ~0.6–0.8 against
a floor of ~0.13, still an order of magnitude out. The budget ladder should have
been chosen to straddle the *plateau*, not the transient, and the primary should
have been the first budget at which the arms have converged — 50 ms on this
machine.

**What I am claiming, therefore:** no durable difference between residual and
either control. The 20 ms result is reported because it was preregistered, not
because it is believed — and a clean-tree rerun of the same command moved its CI
upper bound from `−0.0999` to `−0.0100`, which is documented under Limitations.

## The three preregistered contrasts

At the preregistered 20 ms primary, and at the final iterate:

| contrast | budget | median | 95% CI | verdict | predicted |
|:--|:--|--:|:--|:--|:--|
| residual − fixed_0.05 | 20 ms | −0.1486 | [−0.1502, −0.0999] | outside null | null or negative |
| | final | +0.00081 | [+0.00013, +0.00090] | **null** | ✓ |
| residual − geometric_slow | 20 ms | −0.8273 | [−0.8489, −0.8186] | outside null | ✓ residual wins |
| | final | +0.0101 | [+0.0093, +0.0101] | **null** | — |
| residual − geometric_fast | 20 ms | −0.0417 | [−0.0459, **+0.0011**] | indeterminate | null |
| | final | −2.9e-06 | [−0.00041, +0.0000002] | **null** | ✓ |

**Prediction 1** (residual will not beat fixed-0.05): fires as falsified at 20 ms
under its own wording, holds at every other budget, and is decisively null at the
final iterate. Given the sign flip documented above, the honest verdict is that
the test was not decisive in either direction.

**Prediction 2** (residual beats geometric_slow, and it means nothing): confirmed
at every budget through 800 ms — and it does mean nothing, as recorded in advance.
Note the reversal at the final iterate, where geometric_slow is the *best* arm of
the six at 0.1287.

**Prediction 3, the real test** (residual ≈ geometric_fast): the primary-budget CI
is [−0.0459, +0.0011] — it crosses zero, so "no difference" cannot be rejected,
and it extends past −0.02, so the strict null criterion is not met either.
**Indeterminate at 20 ms.** At every budget from 50 ms on, and at the final
iterate, it is null to five or six decimal places.

## What this study concludes

**Observing `span(TV − V)` adds nothing over annealing ε on a matched timetable,
on this instance.** Residual and geometric_fast are indistinguishable at every
budget where the solver has converged, and identical to −2.9e-06 at the final
iterate. The benefit of the residual rule is entirely in *how fast ε anneals* —
which a two-line geometric schedule with no feedback reproduces exactly.

Two secondary observations, neither preregistered, both worth recording:

- **A well-chosen constant ε beats the adaptive rule at short budgets.**
  fixed_0.1 reaches 0.5680 at 20 ms against residual's 0.6314. The preregistered
  contrast was against fixed_0.05, which residual does beat there — but the
  broader row does not support a claim that adaptivity wins early.
- **geometric_slow ends best of all six arms** (0.1287 vs fixed_0.05's 0.1379),
  which nothing predicted. It is still annealing when the run ends, so it spends
  the whole run coarser than the others and arrives at `ε_min` only at the last
  cycle. Whether slow annealing genuinely helps at the plateau, or this is the
  wobble described at 5.6, is not resolved by this experiment.

## Limitations

- **Scope.** `N = 3`, `Q = 10`, one instance, `γ = 0.95`, one schedule, one
  machine. `N = 4` was cut at 5.6 as a scope decision. The maze transfer check
  was not run, so nothing here speaks to saturated value distributions.
- **The primary endpoint is wall-clock, so it is machine-dependent and not
  bit-reproducible** — and this was measured, not merely anticipated.
  `scripts/reproduce_residual_epsilon.sh`, run from a clean tree with no
  ground-truth or Numba cache on the same machine, reproduces **every
  final-iterate value bit-for-bit** and the headline contrast exactly
  (`−2.94e-06`, CI `[−0.0004116, +2.127e-07]`). The 20 ms primary contrast did
  not hold up:

  | `residual − fixed_0.05` @ 20 ms | median | 95% CI |
  |:--|--:|:--|
  | first run | −0.1486 | [−0.1502, −0.0999] |
  | clean reproduction | −0.1210 | [−0.1530, **−0.0100**] |

  The CI upper bound moved an order of magnitude toward zero between two runs of
  the same command on the same hardware. The falsification of prediction 1 rests
  on a quantity that is not stable across reruns, which is the strongest
  available evidence that the endpoint — not the method — is what failed here.
  The final-iterate column is deterministic and is the reproducible part of the
  result.
- **20 seeds proved underpowered for the primary contrast.** Prediction 3's CI
  crosses zero at 20 ms. More seeds would narrow it, but choosing `n` after
  seeing the result changes what the interval means; the underpowered result is
  reported as it stands rather than extended.
- **`err_at_budget` is a step function over traced rows** at `fine_stride = 10`.
  In the steep transient, adjacent rows differ substantially, so budget columns
  below ~50 ms carry a resolution error that the CIs do not model.
- **The stability threshold `c < 2.11` was measured, not derived**, on this
  instance and schedule. It is not claimed to transfer.

## Environment

| Item | Value |
|:--|:--|
| CPython | `3.14.6` |
| NumPy | `2.4.6` |
| Numba | `0.66.0` |
| pydantic | `2.13.4` |
| Platform | macOS `27.0`, arm64, Apple M4 Pro, 14 cores |
| Provenance | `shared-core-v1` = `0dce46b` |

Numba kernels are warmed inside `run_adaptive` before any timer starts, so
compilation is outside every wall-clock number above.
