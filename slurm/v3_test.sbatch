#!/bin/bash
#SBATCH --job-name=v3-naive-test
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

# Build and run the v3 (naive CUDA) correctness test on one Euler GPU.
# Submit from the repo root: sbatch slurm/v3_test.sbatch
set -euo pipefail

module load nvidia/cuda   # pin a version if the default changes

nvidia-smi --query-gpu=name,compute_cap --format=csv
make v3
./build/test_v3 200000 128 128 10
