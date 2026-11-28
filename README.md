# GPU Vector Search

## Build and run on Euler

**1. Get SIFT1M** (once, on a login node — compute nodes may not have internet):

```bash
bash scripts/get_sift1m.sh        # downloads to data/sift/ and checks file sizes
```

**2. Benchmark a version** (submit from the repo root):

```bash
mkdir -p results
sbatch --export=ALL,VERSION=v3 slurm/bench.sbatch
# options: BATCHES="1 100 10000"  KS="10 100"  REPEAT=5
```

The job builds `build/bench_v3`, runs it for each batch size and k, and appends one row per run to `results/benchmark.csv`:

```text
version,batch,k,latency_ms,throughput_qps,recall_at_k
```

Commit the CSV rows; the Slurm `.out`/`.err` logs are git-ignored.

**Running the driver by hand** (inside a GPU job, e.g. `srun -p instruction --gres=gpu:1 --pty bash`, then `module load nvidia/cuda`):

```bash
make v3                                   # build/bench_v3 and build/test_v3
./build/bench_v3 --batch 1000 --k 10      # --repeat 5 --warmup 1 --no-csv --version LABEL
```

The log also prints a **tie-aware recall** (not written to the CSV): a returned neighbor counts as correct if it is exactly as close as the k-th ground-truth neighbor. SIFT1M has duplicate vectors, so an exact search can score slightly below 1.0 on plain recall@k purely from ties; the log line says how many misses are ties versus real errors.

Each version links the same driver (`bench/bench.cpp`) against its own `search()`; `make vN` builds `build/bench_vN`. Latency is the median of `--repeat` timed calls after `--warmup` untimed ones. Because `search()` takes host pointers, GPU timings include copying the database to the device.

**Correctness test** for v3 (random data, exact match against CPU brute force): `sbatch slurm/v3_test.sbatch`.

**Roofline profiling** (Nsight Compute): `sbatch slurm/ncu.sbatch` profiles v3 (`--export=ALL,VERSION=v4,BATCH=1000` for others). It writes `results/ncu/<tag>_raw.csv` (commit it) and a `.ncu-rep` report for the Nsight Compute GUI (git-ignored), then runs `python3 scripts/roofline.py`, which turns every raw CSV into `results/roofline.csv`: arithmetic intensity, achieved GFLOP/s and GB/s, and % of the roof per kernel.

**Baselines and ceilings** (FAISS GPU, cuBLAS GEMM, bandwidth/FMA microbenchmarks): `bash scripts/setup_faiss.sh` once on a login node, then `sbatch slurm/baselines.sbatch`. See [baselines/README.md](baselines/README.md) for what each measures and how to compare fairly.

Set `CUDA_ARCH` to target one GPU, e.g. `make v3 CUDA_ARCH=-arch=sm_89`.

## Versions

| Version | Implementation | Status |
| --- | --- | --- |
| v1 | Serial CPU | TODO |
| v2 | OpenMP CPU | TODO |
| v3 | Naive CUDA | Done |
| v4 | Tiled CUDA | In Progress |
| v5 | Fused CUDA | In Progress |
| v6 | WMMA CUDA | TODO |

## Results

TODO: add the benchmark results and final plots.



## Playground

The `zplayground/` directory is intended for individual learning, experimentation, and testing without affecting the main project implementations. The `z` prefix is simply to keep the directory at the bottom of the repository listing.

**Commits containing only playground changes should begin with `Learning:`**

I hope this directory encourages us to experiment freely, explore ideas, and use it as a scratchpad without worrying about writing polished code.

Since the ideas and experiments developed along the way are valuable to the project, and our evaluation also considers commit history, I believe maintaining this directory will help document our individual learning and contributions.

### Directory Structure

Each team member should create a personal subdirectory.

Experiments can be organized into daily folders using the `YYYY-MM-DD` format, as I have done in my directory. However, **feel free to organize your playground in whatever way feels most natural to you.**

```text
zplayground/
├── member1/
│   ├── 2026-10-21/
│   │   └── experiment.ipynb
│   └── 2026-10-22/
│       └── test.cpp
├── member2/
└── member3/
```

### Guidelines

1. **Personal workspace:** Work within your own directory and avoid modifying other members' experiments.

2. **Commit convention:** Commits containing only playground changes should begin with `Learning:`.

   * Example: `Learning: Exploring SIFT1M dataset structure`

3. **Experiment freely:** Code in the playground does not need to be optimized, polished, or production-ready. The purpose is to encourage exploration and learning.

4. **Integration:** Successful experiments can be moved into the appropriate main project directories after team review.

5. **Data management:** Avoid committing large datasets, compiled binaries, or unnecessary temporary files.

6. **Jupyter notebooks:** Use `nbstripout` to prevent notebook outputs from being committed.

**Note:** The playground is a shared learning space and is not part of the official benchmarking pipeline.
