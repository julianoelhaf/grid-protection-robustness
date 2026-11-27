#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$1" eusipco case --run "$2" "$SLURM_ARRAY_TASK_ID" --jobs "${SLURM_CPUS_PER_TASK:-1}"
