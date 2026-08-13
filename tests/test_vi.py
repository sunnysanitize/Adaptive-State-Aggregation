"""Exact value iteration against the two hand-computed fixtures.

This is the anchor `test_adaptive` pins the adaptive layer *to*. If these numbers are
wrong, `agg_len = 0` reproducing them bit for bit means only that both are
wrong in the same way.

The first test is also the regression test for the stopping rule. VI here stops
on the sup norm, `max_abs_diff(V_k+1, V_k) < tol`, not on the span seminorm.
Swap in `span_seminorm` and `two_state_chain` fails at 0.30 -- the span is blind
to an additive constant, and under the optimal policy the difference vector
converges to a multiple of the all-ones vector, so span-stopping halts as soon
as the *shape* is right and leaves the offset behind. `tiny_gridworld` will not
catch that: its zero-cost absorbing goal pins the difference at the goal to 0
forever, which removes the constant freedom. Both fixtures are needed, and only
one of them is load-bearing here.

Span is the correct rule for average-cost relative VI, where `h` is defined only
up to a constant. It lives on in `norms.py`.
"""

import numpy as np
import pytest
from fixtures import ALL_FIXTURES

from mdpagg.vi import value_iteration

# Tight enough that the a-posteriori bound below lands well under the 1e-9 the
# task asks for: at gamma = 0.9 the bound is 9 * tol.
TOL = 1e-12


@pytest.mark.parametrize("make", ALL_FIXTURES, ids=lambda f: f.__name__)
def test_matches_the_hand_computed_value_function(make):
    f = make()

    result = value_iteration(f.mdp, f.gamma, tol=TOL)

    assert result.v == pytest.approx(f.v_star, abs=1e-9), f.name


@pytest.mark.parametrize("make", ALL_FIXTURES, ids=lambda f: f.__name__)
def test_error_obeys_the_a_posteriori_bound(make):
    """||V_k - V*||inf <= gamma / (1 - gamma) * ||V_k - V_k-1||inf.

    The guarantee sup-norm stopping buys and span stopping does not. Asserting
    it, rather than just "close enough", is what makes `tol` a knob with a
    meaning: it says how far from V* the answer may be, up to the horizon
    factor. `solve.py` picks its 1e-10 ground-truth tolerance on this basis.
    """
    f = make()

    result = value_iteration(f.mdp, f.gamma, tol=TOL)

    horizon = f.gamma / (1.0 - f.gamma)
    assert np.max(np.abs(result.v - f.v_star)) <= horizon * TOL + 1e-15, f.name


@pytest.mark.parametrize("make", ALL_FIXTURES, ids=lambda f: f.__name__)
def test_terminates_in_a_sane_number_of_iterations(make):
    """Finite and sane: hundreds at worst, not the iteration cap.

    Hitting the cap raises, so this is really a check that the geometric rate
    is the one the discount implies -- roughly log(tol) / log(gamma) sweeps.
    """
    f = make()

    result = value_iteration(f.mdp, f.gamma, tol=TOL)

    assert 0 < result.iterations < 1000, f"{f.name}: {result.iterations}"


@pytest.mark.parametrize("make", ALL_FIXTURES, ids=lambda f: f.__name__)
def test_bills_one_backup_per_state_per_sweep(make):
    """The backup count the overhead figure is a share *of*.

    A sweep is |S| backups by construction, so this pins the counter to the
    work done rather than to a number someone incremented by hand.
    """
    f = make()

    result = value_iteration(f.mdp, f.gamma, tol=TOL)

    assert result.backups == result.iterations * f.mdp.num_states, f.name
