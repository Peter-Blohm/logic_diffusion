#!/bin/bash

OBJ_VALUES=("a mountain landscape" "a flamingo" "a dragonfly" "dandelion" "a sunflower" "a rocket" "moon" "a snail" "an eagle" "zebra" "chess pawn" "a pineapple" "a spider web" "a waffle cone" "a cat" "a chair" "a donut" "otter" "pebbles on a beach" "teddy bear")
BG_VALUES=("silhouette of a dog" "a candy cane"  "a helicopter" "fireworks" "a lemon" "a cactus" "cookie" "a cinnamon roll" "an airplane" "barcode" "bottle cap" "a beehive" "a bicycle wheel" "a volcano" "a dog" "an avocado" "a map" "duck" "a turtle" "panda")

METHODS=("and" "dombi_and" "avg" "sd_ab" "sd_ba")

TASK_ID=$1
OBJ=${OBJ_VALUES[$TASK_ID]}
BG=${BG_VALUES[$TASK_ID]}

METHOD=${METHODS[$2]}

echo "Running job with METHOD=${METHOD}, OBJ=${OBJ}, BG=${BG}"

python ../active/clip_eval.py --num_inference_steps 200 --batch_size 20 --method "${METHOD}" --obj "${OBJ}" --bg "${BG}" --T 0.1
