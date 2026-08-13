#!/usr/bin/env bash
#
# Ground truth, policy baselines, and the preregistered arm comparison.
# Assumes no cached ground truth and no Numba cache. Takes a few minutes;
# outputs land under results/.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG=configs/inventory_n3.json

# V* is never computed implicitly, so it is solved here first.
"${PYTHON_BIN}" -m mdpagg.solve "${CONFIG}"

# All three baselines must be strictly worse than exact optimal; one beating V*
# means the sign convention, dynamics or evaluator is wrong.
"${PYTHON_BIN}" scripts/check_baselines.py "${CONFIG}"

# The fixed-eps reference the arm comparison is measured against.
"${PYTHON_BIN}" scripts/sweep.py "${CONFIG}" \
    --out results/inventory_fixed_eps.json --curves --policy-loss-curve

"${PYTHON_BIN}" scripts/sweep.py "${CONFIG}" --arms --policy-loss-curve

# Again with per-iteration traces, for the error-vs-time curves. Split from the
# headline run because it writes ~24 MB.
"${PYTHON_BIN}" scripts/sweep.py "${CONFIG}" --arms --policy-loss-curve --curves \
    --out results/arms_inventory_n3_curves.json

echo
echo "reproduction complete; outputs are under results/"
echo "  results/inventory_baselines.json        policy baselines"
echo "  results/inventory_fixed_eps.json        fixed-eps reference"
echo "  results/arms_inventory_n3.json          arm comparison"
echo "  results/arms_inventory_n3_curves.json   the same, with traces"
