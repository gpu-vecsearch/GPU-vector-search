#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

# Benchmark one version on SIFT1M across query batch sizes; appends one row per
# run to results/benchmark.csv.
#
# Submit from the repo root, after `bash scripts/get_sift1m.sh` on a login node:
#   sbatch --export=ALL,VERSION=v3 slurm/bench.sbatch
#   sbatch --export=ALL,VERSION=v3,BATCHES="1 100 1000",KS="10 100" slurm/bench.sbatch
#
# CPU versions (v1/v2) do not need the GPU; for v2 thread scaling, use a job
# with more --cpus-per-task and set OMP_NUM_THREADS per run.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

VERSION="${VERSION:?set VERSION, e.g. --export=ALL,VERSION=v3}"
BATCHES="${BATCHES:-1 10 100 1000 10000}"
KS="${KS:-10}"
REPEAT="${REPEAT:-5}"

module load nvidia/cuda   # pin a version if the default changes
source slurm/require_gpu.sh   # stop if this is not the RTX 4000 Ada our results use

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true
nvcc --version | tail -1

make "$VERSION"
VERIFY_ONLY=1 bash scripts/get_sift1m.sh   # fail fast if the dataset is missing

for k in $KS; do
    for b in $BATCHES; do
        ./build/bench_"$VERSION" --batch "$b" --k "$k" --repeat "$REPEAT"
    done
done
