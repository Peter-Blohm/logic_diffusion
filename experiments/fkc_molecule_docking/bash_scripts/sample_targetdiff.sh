#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define idx pairs
IDXS=(
  "8 226"
  "29 371"
  "200 416"
  "164 287"
  "208 209"
  "31 313"
  "373 398"
  "226 8"
  "371 29"
  "416 200"
  "287 164"
  "209 208"
  "313 31"
  "398 373"
)

# Define molecule sizes
MOLLENS=(15 19 23 27 35)

bs=32
seed=0
EXP_DIR="${EXP_DIR:-$DEFAULT_EXP_DIR}"

cd "$EXP_DIR"

for pair in "${IDXS[@]}"; do
  # split pair into idx1 and idx2
  set -- $pair
  idx1=$1
  idx2=$2

  for numatoms in "${MOLLENS[@]}"; do
    echo "Running with idx1=$idx1 idx2=$idx2 numatoms=$numatoms"

    python scripts/sample_for_pocket.py \
      configs/sampling_targetdiff.yml \
      --data_id $idx1 \
      --result_path ${EXP_DIR}/TEST_outputs/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --num_samples $bs \
      --num_atoms_to_sample $numatoms \
      --experiment_dir ${EXP_DIR} \
      --seed $seed

    python3 scripts/evaluate_targetdiff.py  \
      --result_path ${EXP_DIR}/TEST_outputs_prior/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --experiment_dir ${EXP_DIR} \
      --sample_path ${EXP_DIR}/TEST_outputs/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --idx1 $idx1 \
      --idx2 $idx1

    python3 scripts/evaluate_targetdiff.py \
      --result_path ${EXP_DIR}/TEST_outputs_prior/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --experiment_dir ${EXP_DIR} \
      --sample_path ${EXP_DIR}/TEST_outputs/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --idx1 $idx1 \
      --idx2 $idx2

    python3 scripts/align_protein_ligand_score.py  \
      --results_dir ${EXP_DIR}/TEST_outputs_prior/baseline_SDE_targetdiff_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --save_dir ${EXP_DIR}/TEST_aligned_SDE_prot_pockets_bs${bs}_fixednumatoms${numatoms}_seed${seed} \
      --experiment_dir ${EXP_DIR} \
      --idx1 $idx1 \
      --idx2 $idx2

  done
done
