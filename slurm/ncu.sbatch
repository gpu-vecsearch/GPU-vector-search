#!/bin/bash
#SBATCH --job-name=ncu
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --exclude=euler01,euler04   # these have RTX 4000 SFF Ada GPUs; see slurm/require_gpu.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

# Profile one version's kernels with Nsight Compute for the roofline.
#
# Submit from the repo root (SIFT1M must already be in data/sift/):
#   sbatch slurm/ncu.sbatch                                   # v3, batch 100, k 10
#   sbatch --export=ALL,VERSION=v4,BATCH=1000 slurm/ncu.sbatch
#
# Outputs in results/ncu/ (TAG = <version>_b<batch>_k<k>):
#   TAG_raw.csv    roofline counters per kernel launch   -> commit this
#   TAG.ncu-rep    full report; open in the Nsight Compute GUI (has a roofline
#                  chart)                                 -> git-ignored, scp it
# then scripts/roofline.py turns every *_raw.csv into results/roofline.csv.
#
# Profiling runs search() once (no warm-up, no repeats). ncu replays each kernel
# many times and locks clocks to base, so kernel times here are longer than in
# benchmark.csv; the peaks are measured at the same clocks, so the roofline is
# consistent. For v3 the arithmetic intensity does not depend on the batch, so a
# small batch keeps the replays fast.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "$SLURM_SUBMIT_DIR"

VERSION="${VERSION:-v3}"
BATCH="${BATCH:-100}"
K="${K:-10}"
LAUNCHES="${LAUNCHES:-20}"   # max kernel launches to profile

module load nvidia/cuda   # ncu ships with the CUDA toolkit
source slurm/require_gpu.sh   # stop if this is not the RTX 4000 Ada our results use

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true
command -v ncu >/dev/null || { echo "ncu not found after 'module load nvidia/cuda'" >&2; exit 1; }
ncu --version | tail -1

make "$VERSION"
VERIFY_ONLY=1 bash scripts/get_sift1m.sh

OUT=results/ncu
TAG="${VERSION}_b${BATCH}_k${K}"
mkdir -p "$OUT"
BENCH=(./build/bench_"$VERSION" --batch "$BATCH" --k "$K" --warmup 0 --repeat 1 --no-csv)

# Counters for scripts/roofline.py: time, DRAM/L2 traffic, FP32 instruction
# counts (FLOPs = 2*FFMA + FADD + FMUL), and the device peaks at the profiling clocks.
METRICS="gpu__time_duration.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained,\
sm__cycles_elapsed.avg.per_second,\
dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed"

ncu_hint() {
    echo "ncu failed. If the log mentions ERR_NVGPUCTRPERM, GPU performance counters" >&2
    echo "are restricted on this node; ask the Euler admins / course staff to enable them." >&2
}

echo "== pass 1: roofline counters -> $OUT/${TAG}_raw.csv"
ncu --metrics "$METRICS" --launch-count "$LAUNCHES" \
    --csv --page raw --print-units base \
    --log-file "$OUT/${TAG}_raw.csv" "${BENCH[@]}" || { ncu_hint; cat "$OUT/${TAG}_raw.csv" >&2 || true; exit 1; }

echo "== pass 2: full report -> $OUT/${TAG}.ncu-rep"
ncu --set full --import-source yes --launch-count "$LAUNCHES" \
    -f -o "$OUT/$TAG" "${BENCH[@]}" || { ncu_hint; exit 1; }

echo "== roofline summary"
python3 scripts/roofline.py
