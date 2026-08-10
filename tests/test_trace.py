"""Task 3.3: the trace survives a round-trip through JSON.

Two things here are fragile in ways that are easy to miss until a result file
is unreadable weeks later.

`NaN` is not JSON. The standard has no literal for it, and Python's `json`
emits the bare token `NaN` anyway -- a non-standard extension that its own
loader happens to accept. The trace depends on that: `policy_loss` is NaN
everywhere off the coarse stride, which is most of the file. It works today,
and it is exactly the kind of thing that works until a library version changes
or another tool reads the file, so the round-trip is asserted rather than
assumed, and the literal itself is checked in the raw text.

numpy scalars are not JSON either. `json` refuses `np.float64` and `np.int32`
outright, so every column is converted with `.tolist()` on write. A column that
skipped the conversion would raise at serialization time -- loud, but only if
something actually serializes it, which is why this runs on a filled trace
rather than an empty one.

Exact equality on the floats, not approximate. Python writes floats with a
repr that round-trips, so a value that comes back merely close means the
conversion went through something lossy on the way.
"""

import json
import math

import numpy as np
import pytest

from mdpagg import trace as trace_module
from mdpagg.trace import allocate, document, read, write

ITERATIONS = 20
FINE, COARSE = 2, 10


def filled_trace():
    """A trace whose values are all awkward to serialize in some way."""
    t = allocate(ITERATIONS, fine_stride=FINE, coarse_stride=COARSE)

    for i in range(0, ITERATIONS, FINE):
        loss = 0.5 + i if t.wants_policy_loss(i) else math.nan
        t.record(
            t=i,
            phase=i % 2,
            err_inf=1.0 / 3.0 + i,
            residual_span=1e-17 * (i + 1),
            num_groups=200 + i,
            eps=0.5,
            clamped=bool(i == 4),
            billed=2_500_000_000 + i,
            actual=2_600_000_000 + i,
            wall_ns=123_456_789 + i,
            policy_loss=loss,
        )

    return t


def test_policy_loss_is_nan_off_the_coarse_stride():
    t = filled_trace()

    recorded = t.iteration[: t.rows]
    finite = np.isfinite(t.policy_loss[: t.rows])

    assert list(recorded) == list(range(0, ITERATIONS, FINE))
    assert list(recorded[finite]) == list(range(0, ITERATIONS, COARSE))
    assert finite.sum() < t.rows, "the coarse stride is not sparser than the fine one"


def test_nan_survives_the_round_trip(tmp_path):
    path = tmp_path / "run.json"
    write(path, document(filled_trace(), {"k": 1}, {"master": 0}, "abc", 5))

    raw = path.read_text()
    assert "NaN" in raw, "Python stopped emitting the non-standard NaN literal"

    loss = read(path)["trace"]["policy_loss"]
    assert math.isnan(loss[1]), loss
    assert not math.isnan(loss[0]), loss


def test_numpy_scalars_do_not_reach_the_serializer():
    columns = filled_trace().columns()

    for name, values in columns.items():
        for x in values:
            assert type(x) in (int, float), f"{name} leaked {type(x).__name__}"


def test_every_column_round_trips_exactly(tmp_path):
    t = filled_trace()
    path = tmp_path / "run.json"
    write(path, document(t, {"k": 1}, {"master": 0}, "abc", 5))

    back = read(path)["trace"]

    for name, values in t.columns().items():
        for i, (before, after) in enumerate(zip(values, back[name], strict=True)):
            if isinstance(before, float) and math.isnan(before):
                assert math.isnan(after), f"{name}[{i}]"
            else:
                assert before == after, f"{name}[{i}]: {before!r} != {after!r}"


def test_counts_above_the_int32_ceiling_survive():
    """500^2 states times a few thousand sweeps overruns int32."""
    t = filled_trace()

    assert t.billed[0] > 2**31
    assert t.columns()["billed"][0] == 2_500_000_000


def test_document_carries_what_a_reader_needs(tmp_path):
    path = tmp_path / "run.json"
    write(path, document(filled_trace(), {"gamma": 0.95}, {"master": 7}, "deadbeef", 42))

    doc = read(path)

    assert doc["config"] == {"gamma": 0.95}
    assert doc["seeds"] == {"master": 7}
    assert doc["v_star_hash"] == "deadbeef"
    assert doc["strides"] == {"fine": FINE, "coarse": COARSE}

    env = doc["environment"]
    for key in ("cpython", "numpy", "numba", "platform", "machine"):
        assert env[key], key


def test_recording_past_the_allocation_raises():
    t = allocate(4, fine_stride=2)

    for i in (0, 2):
        t.record(i, 0, 0.0, 0.0, 1, 0.5, False, 0, 0, 0)

    with pytest.raises(IndexError, match="trace is full"):
        t.record(4, 0, 0.0, 0.0, 1, 0.5, False, 0, 0, 0)


def test_bad_strides_are_rejected():
    with pytest.raises(ValueError, match="strides must be >= 1"):
        allocate(10, fine_stride=0)


def test_environment_names_the_three_versions_the_repro_note_wants():
    env = trace_module.environment()

    assert env["cpython"].count(".") == 2
    assert env["numpy"] == np.__version__
    assert json.dumps(env)
