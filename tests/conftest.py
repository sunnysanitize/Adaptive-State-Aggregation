"""Shared test setup: the jit-mode skip marker.

`make debug` runs the suite under NUMBA_DISABLE_JIT=1 so that "the logic is
wrong" and "it won't compile" stay separable failures. A few claims, though,
are true only of compiled code, and running them interpreted would either fail
for an uninteresting reason or -- worse -- pass and be mistaken for evidence.
Those tests carry `requires_jit`, and the reason prints in the skip output so
the scoping stays visible rather than living in someone's head.
"""

import os

import pytest

JIT_DISABLED = os.environ.get("NUMBA_DISABLE_JIT", "0") not in ("", "0")

requires_jit = pytest.mark.skipif(
    JIT_DISABLED,
    reason=(
        "njit-mode claim: exactness holds within one run mode, not across all "
        "three. Numba may contract a multiply-add into a single FMA where "
        "CPython will not, so bit-for-bit equality is asserted under `make "
        "test` only. See overview 1.3 and the ambiguity log."
    ),
)
