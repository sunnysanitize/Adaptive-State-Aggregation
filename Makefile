PY ?= .venv/bin/python
# -rs prints skip reasons. The bitwise gates skip themselves under `make
# debug`, and a silent `s` would hide which mode the claim is scoped to.
PYTEST := $(PY) -m pytest -q -rs

.PHONY: all lint test debug bounds clean-cache

# Lint first because it is the cheapest failure, then
# the three run modes; `bounds` goes last because it wipes __pycache__ and so
# pays for a full recompile.
all: lint test debug bounds

lint:
	$(PY) -m ruff check .
	$(PY) -m mypy

# The three run modes. Numba's compilation errors are cryptic, so keeping
# "the logic is wrong" and "it won't compile" as separable failures is worth
# the extra run.

# Compiled. The mode every number is measured in, and the only mode the
# bitwise equality tests claim anything about.
test:
	$(PYTEST)

# Pure Python. Readable tracebacks, and no FMA contraction, which is why the
# bitwise tests skip themselves here rather than failing.
debug:
	NUMBA_DISABLE_JIT=1 $(PYTEST)

# Trap out-of-range CSR reads, which are silent once jitted.
#
# Depends on clean-cache for a measured reason: a kernel cached without
# boundscheck is reused when boundscheck is on, so a warm cache turns this
# mode into a no-op that passes. Measured: the same read returns
# 2.5e-313 with a warm cache and raises IndexError with a cold one.
bounds: clean-cache
	NUMBA_BOUNDSCHECK=1 $(PYTEST)

clean-cache:
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
