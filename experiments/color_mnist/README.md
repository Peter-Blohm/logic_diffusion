# Color-MNIST Experiment

This folder contains the Color-MNIST diffusion composition experiment used in this repository.

The implementation is adapted from the diffusion code path in Garipov et al.'s compositional-sculpting repository:

- https://github.com/timgaripov/compositional-sculpting
- Upstream diffusion docs: https://github.com/timgaripov/compositional-sculpting/tree/main/diffusion

## Installation

From repository root:

```bash
cd experiments/color_mnist
mamba env create -f environment.yml
mamba activate color-mnist-composition
```

## Code structure

- `custom_datasets.py`: Color-MNIST dataset definitions (`M1/M2/MN1/MN2/MN3`)
- `models/`: score model and composition modules
- `samplers/`: predictor-corrector sampling code
- `run_scripts/`: wrapper for local sampling runs
- `training/`: training scripts (`train_diffusion.py`, `train_classifier.py`, `train_3way_classifier.py`, `train_3way_conditional_classifier.py`)
- `sample_cnf_compositions.py`: argument-driven CNF composition sampling (main entrypoint)
- `sample_3way_composition.py`: fixed 3-way composition script (legacy)
- `run_formulas.sbatch`: SLURM array launcher for CNF sweeps

## Color-MNIST CNF composition

Single run (from repository root):

```bash
python experiments/color_mnist/sample_cnf_compositions.py \
  --cnf "[[-3,-2],[3,2],[-1]]" \
  --gamma 10 \
  --resample 0 \
  --composition dombi \
  --batch_size 1024 \
  --num_steps 500 \
  --outdir experiments/color_mnist/runs
```

Wrapper run (from repository root):

```bash
./experiments/color_mnist/run_scripts/run_sampling.sh
```

Optional interpreter override:

```bash
export COLOR_MNIST_PYTHON=/path/to/python
```

The script supports:

- CNF formulas via `--cnf`
- composition choice via `--composition {dombi,poe}`
- guidance/temperature controls (`--conj_guidance_scale`, `--disj_guidance_scale`, `--gamma`)
- optional output controls (`--save_fp16`, `--save_individual_pngs`, `--no_colorblind_grid`)

## Batch sweep (SLURM)

From repository root:

```bash
sbatch experiments/color_mnist/run_formulas.sbatch
```

This launcher sweeps predefined formulas, composition type, resampling, and gamma values.

## Required checkpoints

`sample_cnf_compositions.py` expects these base-model checkpoints in `experiments/color_mnist/checkpoints/`:

- `gen_MN1_ckpt_195.pth`
- `gen_MN2_ckpt_195.pth`
- `gen_MN3_ckpt_195.pth`

If checkpoint names/locations differ, update the paths in the script.

## Outputs

Each run creates an output directory containing metadata, samples, likelihood logs, and image grids.
