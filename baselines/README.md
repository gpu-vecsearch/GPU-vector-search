# Baselines

Reference measurements we compare our versions against. **None of this code is one of our six versions**; FAISS and cuBLAS are used only as yardsticks, never as building blocks.

| Baseline | What it measures | Compared against | Output |
| --- | --- | --- | --- |
| [`faiss_flat.py`](faiss_flat.py) | FAISS `GpuIndexFlatL2` exact search on SIFT1M | every version (end-to-end throughput, recall) | rows in `results/benchmark.csv` |
| [`cublas_gemm.cu`](cublas_gemm.cu) | cuBLAS `Q·Xᵀ` at SIFT1M shapes, FP32 and FP16 (Tensor Cores) | v4's tiled GEMM (FP32), v6's WMMA (FP16) | `results/cublas_gemm.csv` |
| [`../micro/`](../micro) | achievable DRAM bandwidth and FP32 FMA throughput | roofline ceilings for all CUDA versions | `results/micro.csv` |

## Running on Euler

```bash
bash scripts/setup_faiss.sh        # once, on a login node: Python venv with faiss-gpu
bash scripts/get_sift1m.sh         # once, if data/sift/ is not there yet
sbatch slurm/baselines.sbatch      # micro + cublas + faiss on one GPU
# just one part: sbatch --export=ALL,STEPS="faiss" slurm/baselines.sbatch
```

Commit the CSVs the job writes; Slurm logs are git-ignored.

## Reading the numbers fairly

- **FAISS appears twice in `benchmark.csv`.**
  - `faiss-gpu-resident`: the database is loaded onto the GPU once, and each timed call only searches. This is how FAISS is normally used.
  - `faiss-gpu-e2e`: each timed call builds the index, copies the 1M-vector database to the GPU and searches. This is the same work our `search()` interface does on every call, so it is the like-for-like comparison.
- **FAISS flat search is exact in FP32,** but it computes distances as ‖q‖² − 2q·x + ‖x‖², so rounding can reorder near-ties. The log line prints tie-aware recall to separate ties from real errors, as `bench` does.
- **cuBLAS times only the GEMM** (CUDA events, no transfers, no top-k), so it bounds v4's distance step, not the whole search. FP32 uses default math (no TF32). FP16 uses FP16 inputs with FP32 accumulation, the same arithmetic as v6. Batches stop at 2048 because B × N must stay below 2³¹ elements for the 32-bit cuBLAS API.
- **Microbenchmarks run at normal (boost) clocks.** Nsight Compute locks base clocks while profiling, so `results/roofline.csv` peaks are lower than `results/micro.csv` ceilings. Compare kernels to the ceilings measured under the same conditions.
- **Same GPU model for everything.** Euler's instruction nodes all advertise `gpu:rtx4000ada`, but some GPUs are RTX 4000 **SFF** Ada (lower bandwidth and clocks). Every benchmark job sources `slurm/require_gpu.sh`, which stops the job unless the GPU is `NVIDIA RTX 4000 Ada Generation`; euler01 and euler04 have SFF cards, so the job scripts exclude them; resubmit with `--exclude=<node>` if a job lands on another SFF card.
