#!/usr/bin/env bash
#SBATCH --array=0-4%1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
windows=(10 20 30 40 50)
exec "$1" eusipco prepare --run "$2" --data "$3" "${windows[$SLURM_ARRAY_TASK_ID]}"
