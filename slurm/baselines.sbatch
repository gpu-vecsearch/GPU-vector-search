#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

# Reference measurements (comparison only; none of these are our versions):
#   micro   achievable DRAM bandwidth and FP32 FMA ceilings  -> results/micro.csv
#   cublas  cuBLAS FP32/FP16 GEMM at the distance-GEMM shapes -> results/cublas_gemm.csv
#   faiss   FAISS GPU exact search on SIFT1M                 -> results/benchmark.csv
#
# Submit from the repo root:
#   sbatch slurm/baselines.sbatch                               # all three
#   sbatch --export=ALL,STEPS="faiss" slurm/baselines.sbatch    # just one
# FAISS needs a one-time `bash scripts/setup_faiss.sh` on a login node, and
# SIFT1M in data/sift/ (`bash scripts/get_sift1m.sh`).
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

STEPS="${STEPS:-micro cublas faiss}"
FAISS_VENV="${FAISS_VENV:-$HOME/.venvs/vecsearch-faiss}"
BATCHES="${BATCHES:-1,10,100,1000,10000}"
GEMM_BATCHES="${GEMM_BATCHES:-1,10,100,1000,2048}"

module load nvidia/cuda   # pin a version if the default changes
source slurm/require_gpu.sh   # stop if this is not the RTX 4000 Ada our results use

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true

for step in $STEPS; do
    case "$step" in
        micro)
            echo "== micro: roofline ceilings"
            make micro
            ./build/bandwidth
            ./build/fma
            ;;
        cublas)
            echo "== cublas: GEMM baseline"
            make cublas
            ./build/cublas_gemm --batches "$GEMM_BATCHES"
            ;;
        faiss)
            echo "== faiss: GPU flat index on SIFT1M"
            if [[ ! -x "$FAISS_VENV/bin/python" ]]; then
                echo "FAISS env missing at $FAISS_VENV; run 'bash scripts/setup_faiss.sh' on a login node" >&2
                exit 1
            fi
            VERIFY_ONLY=1 bash scripts/get_sift1m.sh
            "$FAISS_VENV/bin/python" -I baselines/faiss_flat.py --batches "$BATCHES"
            ;;
        *)
            echo "unknown step '$step' (use micro, cublas, faiss)" >&2
            exit 1
            ;;
    esac
done
