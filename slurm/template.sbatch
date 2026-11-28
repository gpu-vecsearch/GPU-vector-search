#!/bin/bash
#SBATCH --job-name=gpu-vector-search
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --exclude=euler01,euler04   # these have RTX 4000 SFF Ada GPUs; see slurm/require_gpu.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

# Starting point for new Euler jobs; these settings work for slurm/bench.sbatch,
# slurm/ncu.sbatch and slurm/baselines.sbatch. Copy this file, rename the job,
# and replace the commands at the bottom.
#
# - Submit from the repo root: sbatch slurm/<job>.sbatch  (create results/ first)
# - CPU-only jobs (v1/v2): drop --gres and raise --cpus-per-task (up to 20 for
#   OpenMP scaling), and set OMP_NUM_THREADS per run.
# - SIFT1M must be downloaded on a login node first: bash scripts/get_sift1m.sh
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

module load nvidia/cuda   # pin a version if the default changes
source slurm/require_gpu.sh   # stop if this is not the RTX 4000 Ada our results use

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true

# --- experiment-specific commands go here ---
# make v3
# ./build/bench_v3 --batch 1000 --k 10
