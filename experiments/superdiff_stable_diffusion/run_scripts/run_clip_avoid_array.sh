#!/bin/bash
#SBATCH --job-name=clip_avoid_array
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# GPUs
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g

#SBATCH --array=0-139%20

set -euo pipefail



OBJ_VALUES=(
  "A night sky with stars and a crescent moon, reminiscent of Van Gogh's 'Starry Night'."
  "A night sky with stars and a crescent moon, reminiscent of Van Gogh's 'Starry Night'."
  "A portrait of a man with a distorted and fragmented face painted in Picasso's Cubist style."
  "A cat and a ball on the shelf"
  "There are a bicycle and a car in front of the house"
  "orange fruit"
  "a banana"
  "an ocean"
  "strawberry"
  "round shape"
)

BG_VALUES=(
  "Van Gogh"
  "Picasso's Cubist style"
  "Picasso's Cubist style"
  "cat, ball"
  "a bicycle and a car"
  "orange color palette"
  "yellow color palette"
  "blue color palette"
  "red color palette"
  "circle"
)

# Grid definitions
DOMBI_GAMMAS=(1 3 10)
DOMBI_TS=(0.1 1 10)
OTHER_METHODS=( "icn" "and_not_superdiff" "without" "sd_ab" "sd_a" )  # order matters for mapping
NUM_TASKS=${#OBJ_VALUES[@]}  # 10
COMBOS_PER_TASK=$((9 + ${#OTHER_METHODS[@]}))  # 9 dombi + 5 others = 14

# Common args
NUM_STEPS=1000
BATCH_SIZE=20
DEFAULT_T=1
DEFAULT_GAMMA=1
ICN_GUIDANCE=15

# ------------- Map array index to (task, method, hyperparams) -------------
IDX=${SLURM_ARRAY_TASK_ID}

TASK_ID=$(( IDX / COMBOS_PER_TASK ))
OFFSET=$(( IDX % COMBOS_PER_TASK ))

if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "Invalid TASK_ID computed: ${TASK_ID} from IDX=${IDX}"
  exit 1
fi

OBJ=${OBJ_VALUES[$TASK_ID]}
BG=${BG_VALUES[$TASK_ID]}

METHOD=""
GAMMA=${DEFAULT_GAMMA}
T=${DEFAULT_T}
EXTRA_ARGS=()

if (( OFFSET < 9 )); then
  # dombi_contrast block (9 combos): gamma index outer, T index inner
  METHOD="dombi_contrast"
  GAMMA_IDX=$(( OFFSET / 3 ))     # 0..2 -> gamma 1,3,10
  T_IDX=$(( OFFSET % 3 ))         # 0..2 -> T 0.1,1,10
  GAMMA=${DOMBI_GAMMAS[$GAMMA_IDX]}
  T=${DOMBI_TS[$T_IDX]}
else
  # other methods (5 combos): one per method, default gamma/T
  METHOD_IDX=$(( OFFSET - 9 ))    # 0..4
  METHOD=${OTHER_METHODS[$METHOD_IDX]}
  # icn needs guidance_scale=15
  if [[ "$METHOD" == "icn" ]]; then
    EXTRA_ARGS+=( --guidance_scale "${ICN_GUIDANCE}" )
  fi
fi

echo "Array IDX=${IDX} -> TASK_ID=${TASK_ID} (${OBJ}) | METHOD=${METHOD} | gamma=${GAMMA} | T=${T} ${EXTRA_ARGS:+| extras: ${EXTRA_ARGS[*]}}"

# ------------- Run -------------
python ../active/clip_eval.py \
  --num_inference_steps "${NUM_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --method "${METHOD}" \
  --obj "${OBJ}" \
  --bg "${BG}" \
  --T "${T}" \
  --gamma "${GAMMA}" \
  "${EXTRA_ARGS[@]}"

