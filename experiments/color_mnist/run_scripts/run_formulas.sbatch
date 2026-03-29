#!/bin/bash
#SBATCH --job-name=cmnist_gen
#SBATCH --array=0-31
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1         # <- uncomment if you need a GPU on your cluster
#SBATCH -p gpu-v100-32g,gpu-a100-80g                # <- or your GPU partition name

set -euo pipefail
export PYTHONUNBUFFERED=1
PYTHON_CMD=${PYTHON_CMD:-python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SLURM_SUBMIT_DIR"

FORMULAS=(
'[[1]]'
'[[2]]'
'[[3]]'

'[[1,2]]'
'[[1,3]]'
'[[2,3]]'
'[[1,2,3]]'

'[[1],[2]]'
'[[1],[3]]'
'[[2],[3]]'
'[[1],[2],[3]]'

'[[1],[-2],[-3]]'
'[[2],[-1],[-3]]'
'[[3],[-1],[-2]]'

'[[1,2],[-3]]'
'[[1,3],[-2]]'
'[[2,3],[-1]]'

'[[1],[2],[-3]]'
'[[1],[3],[-2]]'
'[[2],[3],[-1]]'

'[[1,2],[-1,-2]]'
'[[1,3],[-1,-3]]'
'[[2,3],[-2,-3]]'

'[[1,2],[-1,-2],[3]]'
'[[1,3],[-1,-3],[2]]'
'[[2,3],[-2,-3],[1]]'

'[[1,2],[-1,-2],[-3]]'
'[[1,3],[-1,-3],[-2]]'
'[[2,3],[-2,-3],[-1]]'

'[[1,2,3],[-1,-2],[-2,-3],[-1,-3]]'
'[[1,2],[-1,-2],[2,3],[-2,-3],[1,3],[-1,-3]]'
# '[[1,2,3],[-1,-2,-3],[1,-2,-3],[-1,2,3],[-1,2,-3],[1,-2,3]]'
'[[-1,2,3],[1,-2,3],[1,2,-3],[-1,-2,-3]]'
)

GAMMAS=(0.1 1 10)
RESAMPLES=(0 1)
COMPOSITIONS=("dombi" "poe")

IDX=${SLURM_ARRAY_TASK_ID}
CNF_STR=${FORMULAS[$IDX]}

echo "Formula index $IDX: $CNF_STR"

for COMP in "${COMPOSITIONS[@]}"; do
  for RESAMPLE in "${RESAMPLES[@]}"; do
    for G in "${GAMMAS[@]}"; do
      TAG="${IDX}-g${G//./p}"   # e.g., 7-g0p1  (kept same tag format)
      OUTDIR="/scratch/work/blohmp1/comp_dif/color_mnist_${COMP}_gamma${G}_resample${RESAMPLE}"

      echo "  -> comp = ${COMP}, resample = ${RESAMPLE}, gamma = ${G} (tag: ${TAG})"
      "$PYTHON_CMD" "$SCRIPT_DIR/sample_cnf_compositions.py" \
        --cnf "$CNF_STR" \
        --gamma "$G" \
        --resample "$RESAMPLE" \
        --composition "$COMP" \
        --tag "$TAG" \
        --outdir "$OUTDIR"
    done
  done
done
