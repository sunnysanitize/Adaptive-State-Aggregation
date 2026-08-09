"""Task 1.5: the Bellman operator is a gamma-contraction in the sup norm.

    ||TV - TW||inf <= gamma * ||V - W||inf

The highest-value test in the project per line written. It is one property, but
almost every way of corrupting a transition matrix violates it, and violates it
loudly: a row summing to more than 1 inflates the expectation and the measured
contraction factor climbs above gamma. Mis-sliced CSR offsets, a successor read
from the wrong pair, probabilities that do not normalize -- all land here.

The MDPs are random by construction, which is the point: a hand-built fixture
tests the transition structure you thought of, and this tests the ones you did
not. `make_random_mdp` lives in `tests/`, never in the package, so that `run.py`
cannot import it and quietly use it as a benchmark -- see its docstring.

Elements are bounded to +-1e3 deliberately. Unbounded floats hand you 1e308,
the subtraction overflows to inf, and the assertion fails on arithmetic rather
than on a real bug.
"""

import pytest
from fixtures import make_random_mdp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mdpagg.norms import max_abs_diff
from mdpagg.types import VALUE
from mdpagg.vi import bellman

NUM_STATES = 12
GAMMA = 0.95

# The slack absorbs float rounding only. At |v| <= 1e3 the backup's own error is
# a few ulps, ~2e-13, so 1e-12 covers it while staying far below any real
# violation -- a row summing to 1.01 shows up as a factor of 0.96, not 0.95.
SLACK = 1e-12

MDPS = {f"seed_{s}": make_random_mdp(NUM_STATES, seed=s) for s in (0, 1, 2)}

values = arrays(
    dtype=VALUE,
    shape=NUM_STATES,
    elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
)


@pytest.mark.parametrize("mdp", MDPS.values(), ids=MDPS.keys())
@settings(max_examples=100, deadline=None)
@given(v=values, w=values)
def test_bellman_operator_contracts_by_gamma(mdp, v, w):
    contracted = max_abs_diff(bellman(mdp, v, GAMMA), bellman(mdp, w, GAMMA))

    assert contracted <= GAMMA * max_abs_diff(v, w) + SLACK
