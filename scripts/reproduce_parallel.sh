#!/usr/bin/env bash
set -euo pipefail

# Reproduces the parallel study end to end: ground truth, one-thread
# characterization, the paired scaling measurement, and every figure.
#
# QUICK=1 runs the same pipeline on a reduced matrix (two sizes, three seeds,
# fewer thread counts). It exists to check the script works from a clean clone
# without spending an hour, and its numbers are NOT the reported ones.
#
# Timings are only meaningful on an otherwise idle machine. Nothing else should
# be running, including another copy of this script.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}
QUICK=${QUICK:-0}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${REPO_ROOT}/results/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${QUICK}" == "1" ]]; then
    PAPER_CONFIGS=(configs/maze_200.json configs/maze_500.json)
    PAPER_SEEDS=(0 1 2)
    SENS_CONFIGS=(configs/maze_200_sensitivity.json)
    SENS_SEEDS=(0 1 2)
    THREADS=(2 4)
    echo "QUICK mode: reduced matrix, results are a smoke test only"
else
    PAPER_CONFIGS=(configs/maze_200.json configs/maze_500.json configs/maze_1000.json)
    PAPER_SEEDS=($(seq 0 19))
    SENS_CONFIGS=(configs/maze_200_sensitivity.json configs/maze_500_sensitivity.json)
    SENS_SEEDS=(0 1 2 3 4)
    THREADS=(2 4 8 10)
fi

echo "== ground truth =="
# Never computed implicitly by a timing run: a run measured against a V* from a
# different problem would look plausible and be wrong.
for config in "${PAPER_CONFIGS[@]}"; do
    for seed in "${PAPER_SEEDS[@]}"; do
        "${PYTHON_BIN}" -m mdpagg.solve "${config}" --problem-seed "${seed}"
    done
done
for config in "${SENS_CONFIGS[@]}"; do
    for seed in "${SENS_SEEDS[@]}"; do
        "${PYTHON_BIN}" -m mdpagg.solve "${config}" --problem-seed "${seed}"
    done
done

echo "== one-thread characterization =="
"${PYTHON_BIN}" scripts/baseline_targets.py "${PAPER_CONFIGS[@]}" "${SENS_CONFIGS[@]}"
"${PYTHON_BIN}" scripts/phase_split.py "${PAPER_CONFIGS[@]}" "${SENS_CONFIGS[@]}"
"${PYTHON_BIN}" scripts/aggregate_grain.py --threads "${THREADS[@]}"

echo "== paired scaling measurement =="
"${PYTHON_BIN}" scripts/scaling.py "${PAPER_CONFIGS[@]}" \
    --seeds "${PAPER_SEEDS[@]}" --threads "${THREADS[@]}" \
    --out results/scaling_paper.json
"${PYTHON_BIN}" scripts/scaling.py "${SENS_CONFIGS[@]}" \
    --seeds "${SENS_SEEDS[@]}" --threads "${THREADS[@]}" \
    --out results/scaling_sensitivity.json
"${PYTHON_BIN}" scripts/analyze_scaling.py \
    results/scaling_paper.json results/scaling_sensitivity.json

echo "== error against wall-clock =="
# Diagnostic, with an observer. Kept apart from the timed runs above.
"${PYTHON_BIN}" scripts/error_vs_time.py "${PAPER_CONFIGS[@]: -1}" "${SENS_CONFIGS[@]: -1}" \
    --threads "${THREADS[@]: -1}"

echo "== figures =="
"${PYTHON_BIN}" scripts/plot_scaling.py \
    results/scaling_paper.json results/scaling_sensitivity.json
"${PYTHON_BIN}" scripts/plot_grain.py results/aggregate_grain.json
"${PYTHON_BIN}" scripts/plot_error_time.py results/error_vs_time.json

echo
echo "parallel reproduction complete; outputs are under results/"
echo "  headline numbers   results/scaling_summary.json"
echo "  figures            results/figures/"
