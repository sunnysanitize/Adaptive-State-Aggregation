#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${REPO_ROOT}/results/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

for config in configs/maze_200.json configs/maze_500.json configs/maze_1000.json; do
    for seed in {0..19}; do
        "${PYTHON_BIN}" -m mdpagg.solve "${config}" --problem-seed "${seed}"
    done
done

"${PYTHON_BIN}" scripts/sweep.py configs/maze_200.json \
    --out results/sweep_maze_200_both.json --vary both --eps 0.05 0.1 0.5
"${PYTHON_BIN}" scripts/sweep.py configs/maze_500.json \
    --out results/sweep_maze_500_both.json --vary both --eps 0.05 0.1 0.5
"${PYTHON_BIN}" scripts/sweep.py configs/maze_1000.json \
    --out results/sweep_maze_1000_both.json --vary both --eps 0.5

"${PYTHON_BIN}" scripts/plot_sweep.py results/sweep_maze_500_both.json
"${PYTHON_BIN}" scripts/audit_value_space.py \
    configs/maze_200.json configs/maze_500.json configs/maze_1000.json
"${PYTHON_BIN}" scripts/audit_gamma.py --eps 0.005
"${PYTHON_BIN}" -m mdpagg.solve configs/maze_200_sensitivity.json
"${PYTHON_BIN}" scripts/check_sensitivity.py configs/maze_200_sensitivity.json

echo "baseline reproduction complete; outputs are under results/"
