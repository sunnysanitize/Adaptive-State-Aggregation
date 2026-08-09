"""Task 2.2: the invariant Algorithm 2 exists to make true.

    |v[s] - centers[group_of[s]]| <= eps   for every s

Every state is within eps of the number standing in for its group. That is the
whole promise of value-based aggregation, and it is what Theorem 1's
2*eps/(1-gamma) bound rests on: the aggregated iterate can only drift from the
exact one by what the grouping threw away. If this holds, the partition is
doing its job.

The other invariants that suggest themselves -- every state in exactly one
group, `members` a permutation, no empty groups, K <= ceil((b2-b1)/eps) -- are
restatements of the counting sort's structure. A bug that broke one would break
this one too, and this is the one whose failure means something.

Two details in the property test:

`eps_effective`, not the drawn eps. When the drawn eps wants more groups than
`max_groups`, `rebin_by_value` widens eps until K fits and reports the widened
value. Asserting against the drawn eps would either fail on a design decision
or force `max_groups` high enough that the clamp path is never exercised.
Asserting against `eps_effective` covers both paths -- but on its own it would
be vacuous, since a broken widening could satisfy it by inflating eps without
limit. The second assertion is what closes that: the clamp may only ever widen.
The two capacities are what make the first path actually get walked; see below.

Bounds on the draws. eps below 1e-3 gives a raw bin count in the millions and
the test dies on memory rather than on a bug. Values are held to +-100 for the
same reason, and because the paper rescales costs so that ||V*||inf = 100 --
the range the implementation actually runs at.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mdpagg.partition import allocate, rebin_by_value
from mdpagg.types import VALUE

NUM_STATES = 64

# Both capacities are measured, not guessed. `arrays` of floats draws mostly
# low-cardinality vectors -- a median of 5 distinct values out of 64 -- so at a
# roomy capacity the clamp never fires and asserting against `eps_effective`
# would cover nothing that asserting against the drawn eps did not. At 4 it
# fires on roughly a third of examples; at 64 it cannot fire at all, since K is
# bounded by |S|. One tight, one roomy, both regimes reached every run.
TIGHT, ROOMY = 4, NUM_STATES

values = arrays(
    dtype=VALUE,
    shape=NUM_STATES,
    elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
)


@pytest.mark.parametrize("max_groups", [TIGHT, ROOMY], ids=["tight", "roomy"])
@settings(max_examples=200, deadline=None)
@given(v=values, eps=st.floats(1e-3, 10.0))
def test_every_state_is_within_eps_of_its_group_center(max_groups, v, eps):
    part = allocate(NUM_STATES, max_groups)
    rebin_by_value(v, eps, max_groups, part)

    assert part.eps_effective >= eps
    assert np.max(np.abs(v - part.centers[part.group_of])) <= part.eps_effective


@pytest.mark.parametrize("constant", [0.0, -3.5, 100.0])
def test_constant_values_collapse_to_one_group(constant):
    """Not an edge case -- iteration 1 of every experiment, since V0 = 0.

    The centre is the value itself, not the b1 + eps/2 the formula in Algorithm
    2 would give, so a constant V lifts back to itself instead of picking up an
    eps/2 offset on the first iteration. See the ambiguity log.
    """
    v = np.full(NUM_STATES, constant, dtype=VALUE)
    part = allocate(NUM_STATES, ROOMY)
    rebin_by_value(v, 0.5, ROOMY, part)

    assert part.num_groups == 1
    assert part.centers[0] == constant
