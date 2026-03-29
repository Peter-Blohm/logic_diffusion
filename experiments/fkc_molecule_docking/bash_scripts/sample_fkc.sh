#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- fixed config ----
INV_TEMP=2
COMPOSITIONS=("dombi_and" "poe")
ELLS_FOR_DOMBI=(0.3 1 3)   # only used when composition=dombi_and
ELL_FOR_POE=1              # safe default so we can always pass --ell

# idx pairs
IDXS=(
  "8 226" "29 371" "200 416" "164 287" "208 209" "31 313" "373 398"
  "226 8" "371 29" "416 200" "287 164" "209 208" "313 31" "398 373"
)

# molecule sizes
MOLLENS=(15 19 23 27 35)

# ---- script defaults from your template ----
resample=1
num_steps=1000
seed=0
sample_num_atoms="fixed"
num_samples=32
exp_note=""
thresh=0.6
EXP_DIR="${EXP_DIR:-$DEFAULT_EXP_DIR}"

cd "$EXP_DIR"

run_job () {
  local idx1="$1" idx2="$2" numatoms="$3" composition="$4" ell="$5"

  local aligned_dir="${EXP_DIR}/TEST_aligned_SDE_prot_pockets_bs${num_samples}_fixednumatoms${numatoms}_seed${seed}"
  local tag="_comp-${composition}_ell-${ell}"
  local result_path="${EXP_DIR}/TEST_outputs/dualtarget_SDE_invtemp${INV_TEMP}_resample${resample}_bs${num_samples}_ns${num_steps}_seed${seed}_fixednumatoms${numatoms}${exp_note}${tag}"

  echo ">>> idx1=${idx1} idx2=${idx2} numatoms=${numatoms} comp=${composition} ell=${ell} inv_temp=${INV_TEMP}"

  python scripts/compose_sample_score.py \
    configs/sampling_fkc.yml \
    --idx1 "$idx1" \
    --idx2 "$idx2" \
    --num_atoms_to_sample "$numatoms" \
    --inv_temp "$INV_TEMP" \
    --num_steps "$num_steps" \
    --seed "$seed" \
    --sample_num_atoms "$sample_num_atoms" \
    --num_samples "$num_samples" \
    --resample "$resample" \
    --resample_thresh "$thresh" \
    --result_path "$result_path" \
    --aligned_prots_path "$aligned_dir" \
    --experiment_dir "$EXP_DIR" \
    --composition "$composition" \
    --ell "$ell"

  python3 scripts/evaluate_compose.py \
    --sample_path "$result_path" \
    --idx1 "$idx1" \
    --idx2 "$idx2" \
    --experiment_dir "$EXP_DIR"
}

for pair in "${IDXS[@]}"; do
  set -- $pair
  idx1=$1; idx2=$2

  for numatoms in "${MOLLENS[@]}"; do
    for composition in "${COMPOSITIONS[@]}"; do
      if [[ "$composition" == "dombi_and" ]]; then
        for ell in "${ELLS_FOR_DOMBI[@]}"; do
          run_job "$idx1" "$idx2" "$numatoms" "$composition" "$ell"
        done
      else
        run_job "$idx1" "$idx2" "$numatoms" "$composition" "$ELL_FOR_POE"
      fi
    done
  done
done
