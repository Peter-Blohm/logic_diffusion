#!/usr/bin/env bash
#SBATCH -J new-sat-grid
#SBATCH -o experiments/new_sat/logs/new-sat-grid-%A_%a.out
#SBATCH -e experiments/new_sat/logs/new-sat-grid-%A_%a.err
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --array=0-35

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"

PYTHON_CMD="${PYTHON_CMD:-/appl/scibuilder-mamba/aalto-rhel9/prod/software/scicomp-python-env/2025.2/b4b5f8e/bin/python}"
OUTPUT_CSV="experiments/new_sat/output_gamma_1.0.csv"
BS=16384
SIGMA=.5
SEED=1

if [[ ! -f "experiments/new_sat/scripts/run_sat_single.py" ]]; then
  echo "ERROR: Could not find experiments/new_sat/scripts/run_sat_single.py under REPO_ROOT='${REPO_ROOT}'" >&2
  echo "Set REPO_ROOT explicitly in sbatch --export if needed." >&2
  exit 3
fi

NDIMS=(2 5 10)
LAMBS=(10 100)
INSTANCES=(majority xor exactly_one)

mkdir -p experiments/new_sat/logs

NUM_NDIMS=${#NDIMS[@]}
NUM_LAMBS=${#LAMBS[@]}
NUM_INSTANCES=${#INSTANCES[@]}
NUM_VARIANTS=$((NUM_LAMBS + 1))
TOTAL_TASKS=$((NUM_NDIMS * NUM_INSTANCES * NUM_VARIANTS))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "Skipping task ${TASK_ID}: outside effective task count ${TOTAL_TASKS}."
  exit 0
fi

NDIM_IDX=$(( TASK_ID / (NUM_INSTANCES * NUM_VARIANTS) ))
REM=$(( TASK_ID % (NUM_INSTANCES * NUM_VARIANTS) ))
INSTANCE_IDX=$(( REM / NUM_VARIANTS ))
VARIANT_IDX=$(( REM % NUM_VARIANTS ))

NDIM=${NDIMS[$NDIM_IDX]}
INSTANCE=${INSTANCES[$INSTANCE_IDX]}

if (( VARIANT_IDX < NUM_LAMBS )); then
  METHOD="dombi"
  LAMB=${LAMBS[$VARIANT_IDX]}
else
  METHOD="prob"
  LAMB=1
fi

echo "Running task=${TASK_ID} ndim=${NDIM} instance=${INSTANCE} method=${METHOD} lamb=${LAMB}"

"${PYTHON_CMD}" experiments/new_sat/scripts/run_sat_single.py \
  --ndim "${NDIM}" \
  --instance "${INSTANCE}" \
  --method "${METHOD}" \
  --lamb "${LAMB}" \
  --bs "${BS}" \
  --sigma "${SIGMA}" \
  --seed "${SEED}" \
  --output-csv "${OUTPUT_CSV}"
