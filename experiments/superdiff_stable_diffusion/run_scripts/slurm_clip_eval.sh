#!/bin/bash
#SBATCH -J clip_eval
#SBATCH -o ../logs/%A.out
#SBATCH -e ../logs/%A.err
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G

mamba activate superdiff-2025

OBJ_VALUES=("a mountain landscape" "a flamingo" "a dragonfly" "dandelion" "a sunflower" "a rocket" "moon" "a snail" "an eagle" "zebra" "chess pawn" "a pineapple" "a spider web" "a waffle cone" "a cat" "a chair" "a donut" "otter" "pebbles on a beach" "teddy bear")
BG_VALUES=("silhouette of a dog" "a candy cane" "a helicopter" "fireworks" "a lemon" "a cactus" "cookie" "a cinnamon roll" "an airplane" "barcode" "bottle cap" "a beehive" "a bicycle wheel" "a volcano" "a dog" "an avocado" "a map" "duck" "a turtle" "panda")
METHODS=("dombi_and" "and" "avg" "sd_ab" "sd_ba" "sd_a" "sd_b")

for i in "${!OBJ_VALUES[@]}"; do
  OBJ=${OBJ_VALUES[$i]}
  BG=${BG_VALUES[$i]}
  for METHOD in "${METHODS[@]}"; do
    echo "Running METHOD=$METHOD OBJ=$OBJ BG=$BG"
    python ../active/clip_eval.py --num_inference_steps 1000 --batch_size 1 --method "$METHOD" --obj "$OBJ" --bg "$BG"
  done
done
