#!/usr/bin/env bash
#SBATCH -J clip-sweep
#SBATCH -p gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-19
#SBATCH -o ../logs/clip-sweep-%A_%a.out
#SBATCH -e ../logs/clip-sweep-%A_%a.err
# #SBATCH --exclude=gpu14

set -euo pipefail
OBJ_VALUES=("a mountain landscape" "a flamingo" "a dragonfly" "dandelion" "a sunflower" "a rocket" "moon" "a snail" "an eagle" "zebra" "chess pawn" "a pineapple" "a spider web" "a waffle cone" "a cat" "a chair" "a donut" "otter" "pebbles on a beach" "teddy bear")
BG_VALUES=("silhouette of a dog" "a candy cane" "a helicopter" "fireworks" "a lemon" "a cactus" "cookie" "a cinnamon roll" "an airplane" "barcode" "bottle cap" "a beehive" "a bicycle wheel" "a volcano" "a dog" "an avocado" "a map" "duck" "a turtle" "panda")
METHODS=("and" "dombi_and" "avg" "sd_ab" "sd_ba")

i=${SLURM_ARRAY_TASK_ID}
OBJ="${OBJ_VALUES[$i]}"; BG="${BG_VALUES[$i]}"

for METHOD in "${METHODS[@]}"; do
  if [[ "$METHOD" == "dombi_and" ]]; then Ts=(0.1 1 10); else Ts=(0.1); fi
  for T in "${Ts[@]}"; do
    srun python ../active/clip_eval.py \
      --num_inference_steps 200 --batch_size 20 \
      --method "$METHOD" --obj "$OBJ" --bg "$BG" --T "$T"
  done
done
