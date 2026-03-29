# SuperDiff Stable Diffusion Experiment

This experiment evaluates compositional text-guided image generation with Stable Diffusion using CLIP and ImageReward metrics.

The code in this directory is adapted from the SuperDiff repository (`necludov/super-diffusion`, `applications/images`) and extended for this project.

## Directory layout

- `active/`: active experiment code and environment file
- `run_scripts/`: runnable launch scripts (single + SLURM/array)
- `legacy/`: unused/archived scripts and previous README

## Environment setup

From the repository root:

```bash
cd experiments/superdiff_stable_diffusion
mamba env create -f active/superdiff-2025.yml
mamba activate superdiff-2025
```

## Single run

Example single run from `run_scripts/`:

```bash
cd experiments/superdiff_stable_diffusion/run_scripts
python ../active/clip_eval.py \
  --num_inference_steps 200 \
  --batch_size 1 \
  --method and \
  --obj "a cat" \
  --bg "a dog" \
  --T 0.1
```

## Run scripts for paper reproduction

Use these scripts in `run_scripts/`:

- `run_clip_combine.sh`: SLURM array sweep for compositional combine experiments
- `run_clip_avoid_array.sh`: SLURM array sweep for contrast/avoid experiments
- `clip_eval_combine.sh`: local/manual launcher for combine setup
- `clip_eval_avoid.sh`: local/manual launcher for avoid setup
- `slurm_clip_eval.sh`: full task-method sweep on SLURM

Example SLURM launches:

```bash
cd experiments/superdiff_stable_diffusion/run_scripts
sbatch run_clip_combine.sh
sbatch run_clip_avoid_array.sh
```

## Expected outputs

`clip_eval.py` writes results under:

- `outputs/saved_sd_results/<method>/<task_name>/<image_id>.png`
- `outputs/saved_sd_results/metrics_<method>/metrics_<method>_<task_name>.csv`

SLURM stdout/stderr logs are written to `logs/`.

## W&B / offline usage

`clip_eval.py` initializes Weights & Biases logging. For offline runs, set:

```bash
export WANDB_MODE=offline
```

For online logging, make sure you are logged in with `wandb login` in the active environment.

## Notes

- Archived materials (including the old README) are in `legacy/`.
- This folder is organized so active code paths are separate from non-used scripts.


