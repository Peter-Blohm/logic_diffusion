#!/bin/bash

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
    "round shape")
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
    "circle")

METHODS=("dombi_contrast" "icn" "and_not_superdiff" "without" "sd_ab" "sd_a")

TASK_ID=1
OBJ=${OBJ_VALUES[$TASK_ID]}
BG=${BG_VALUES[$TASK_ID]}

METHOD=${METHODS[0]}

echo "Running job with METHOD=${METHOD}, OBJ=${OBJ}, BG=${BG}"

python ../active/clip_eval.py --num_inference_steps 200 --batch_size 1 --method "${METHOD}" --obj "${OBJ}" --bg "${BG}" --T 0.1 --gamma 10
