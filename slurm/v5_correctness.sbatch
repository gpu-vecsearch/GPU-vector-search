#!/bin/bash
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=v5-correctness
#SBATCH --output=v5-correctness-%j.out
#SBATCH --error=v5-correctness-%j.err

# Submit from repository root: sbatch slurm/v5_correctness.sbatch
# Never execute this script with bash on an Euler login node.
# Load a currently available CUDA toolkit before submission. We intentionally
# do not hard-code the older profiling toolkit from the course examples.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script using sbatch}"
: "${SLURM_SUBMIT_DIR:?Missing Slurm submit directory}"
cd "$SLURM_SUBMIT_DIR"
command -v nvcc
mkdir -p build

# C++14 suffices for V5 and avoids requiring C++17 from older CUDA toolkits.
nvcc -std=c++14 -O2 -lineinfo -Xcompiler=-Wall,-Wextra -Iinclude \
    src/cuda/v5_fused.cu tests/v5_correctness.cpp -o build/v5_correctness
./build/v5_correctness

# Optional: submit with sbatch --export=ALL,V5_SANITIZE=1 ... to check memory.
if [[ "${V5_SANITIZE:-0}" == 1 ]]; then
    compute-sanitizer --error-exitcode 1 ./build/v5_correctness
fi
