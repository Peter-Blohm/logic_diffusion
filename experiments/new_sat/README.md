# new_sat SAT experiments

This folder is a self-contained SAT experiment package for running composition experiments and producing the paper-style table.

## Contents

- `scripts/run_sat_single.py` — run one SAT configuration and append one row to a CSV.
- `scripts/slurm_run_sat_grid.sh` — Slurm array launcher for the fixed experiment grid.
- `scripts/merge_sat_csv_rows.py` — render LaTeX table from a combined CSV.
- `sat/` — Python SAT library modules used by the scripts.
- `notebooks/dombi_operator_sat.ipynb` — notebook for reproducing run + table flow, plus plotting cell.

## Typical usage

1. Run experiment jobs (Slurm):

```bash
sbatch experiments/new_sat/scripts/slurm_run_sat_grid.sh
```

2. Render table from CSV (after runs finish):

```bash
python experiments/new_sat/scripts/merge_sat_csv_rows.py \
  --input-csv output.csv \
  --table-tex output_table.tex
```

## Notes

- The script `run_sat_single.py` enforces `method=prob` only with `lamb=1`.
- CSV writes are append-based with a file lock (`fcntl`) to reduce interleaving risk under concurrent jobs.
