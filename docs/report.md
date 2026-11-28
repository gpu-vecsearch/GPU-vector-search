# Report

TODO: add the final project report.

## v3: Naive CUDA

v3 is the most direct GPU translation of exact top-k search: one thread per (query, database vector) pair computes a squared L2 distance, and a second kernel selects the k smallest distances per query. It is the baseline for the CUDA versions. It is exact, and the roofline below shows it is entirely memory-bound, which is the problem v4 is designed to fix.

Code: [`src/cuda/v3_naive.cu`](../src/cuda/v3_naive.cu), [`src/cuda/topk.cu`](../src/cuda/topk.cu) (shared with v4), test [`tests/test_search.cpp`](../tests/test_search.cpp).

### Design

**Distance kernel.** The grid is ⌈N/256⌉ × B blocks of 256 threads. Thread *i* of block row *q* computes

  D[q][i] = Σⱼ (q_j − x_ij)²

with one subtraction and one fused multiply-add per dimension, and writes the result to a B × N distance matrix in global memory. All threads of a block read the same query element at each step, so query reads are broadcasts. Adjacent threads read rows of the database that are d × 4 = 512 bytes apart, so database reads are not coalesced. Nothing is staged in shared memory, and no loaded database vector is reused for a second query.

**Top-k kernel.** One block of 256 threads handles one query row. Each thread scans a strided slice of the row (so global reads are coalesced) and keeps a private sorted list of its k best candidates using insertion. The block then merges the 256 lists in k rounds. In each round, a warp-shuffle argmin within each warp and a second argmin across the 8 warp results find the smallest head. The thread that owns it advances its list. Ties are broken by the smaller database index, so results are deterministic. The kernel is templated on a maximum k (16, 32, 64, 128 or 256), and the per-thread lists live in local memory.

**Memory limit.** A full distance matrix for SIFT1M (10,000 × 1,000,000 floats) is 40 GB, twice the 20 GB on the test GPU. `search()` therefore processes queries in chunks: the chunk size is the number of distance rows that fit in 80% of the free device memory (capped at the 65,535 grid-y limit). With the 512 MB database resident, that is roughly 3,700 queries per chunk on a 20 GB card (estimated from the sizing rule, not logged).

**Interface.** Like every version, v3 implements the shared `search()` function. That function takes host pointers, so each call allocates device memory, copies the database and queries to the GPU, and copies the results back. Benchmark latencies therefore include these transfers.

### Correctness

**Unit test.** `tests/test_search.cpp` compares v3 against a CPU brute-force search on random integer-valued vectors (values 0–255, as in SIFT). With integer inputs every squared distance is exactly representable in FP32, so indices and distances must match exactly, including tie order. The cases cover an odd dimension (d = 7), k values across the template buckets (10, 37, 64, 100), k larger than N (padding with index −1), and N = 200,000 with B = 128. All cases pass on the RTX 4000 Ada.

**SIFT1M.** Against the dataset's ground truth, recall@10 is 1.0000 for batches of 1 and 10 queries, and 0.9990–0.9994 for larger batches. Plain recall@k cannot distinguish ties from errors, so the benchmark driver also reports a tie-aware recall: it recomputes each returned neighbor's distance on the CPU in double precision and counts it correct if it is no farther than the k-th ground-truth neighbor. For the full 10,000-query set:

| Metric | Value |
| --- | --- |
| recall@10 | 0.9994 |
| tie-aware recall@10 | 1.0000 |
| misses | 56 |
| misses at exactly the 10th true distance | 56 |
| misses farther than the 10th true distance | 0 |

Every miss returns a different vector at exactly the same distance as the 10th ground-truth neighbor, so v3 is exact up to distance ties.

### Performance

All runs used SIFT1M (N = 1,000,000, d = 128), k = 10 and an NVIDIA RTX 4000 Ada (sm_89, 20 GB) on Euler. Each latency is the median of 5 timed `search()` calls after one untimed warm-up call, and includes host–device transfers.

| Batch | Latency | Throughput | recall@10 |
| ---: | ---: | ---: | ---: |
| 1 | 35.1 ms | 28.5 queries/s | 1.0000 |
| 10 | 57.7 ms | 173.4 queries/s | 1.0000 |
| 100 | 285.1 ms | 350.7 queries/s | 0.9990 |
| 1,000 | 2.57 s | 389.6 queries/s | 0.9994 |
| 10,000 | 24.9 s | 401.3 queries/s | 0.9994 |

Throughput levels off at about 400 queries/s, or 2.5 ms per query, because every query costs the same fixed amount of work: one full pass over the database. At small batches a fixed per-call cost dominates instead. A single query takes 35 ms, of which the distance kernel accounts for about 2.5 ms (from the roofline profile below). We attribute most of the remainder to copying the 512 MB database to the GPU on every call (at a typical ~25 GB/s PCIe rate that alone is about 20 ms), but we did not measure the transfer separately.

### Roofline analysis

We profiled one `search()` call on 100 queries with Nsight Compute (`slurm/ncu.sbatch`; summary from `scripts/roofline.py`, raw counters in `results/ncu/`). Nsight Compute locks the GPU to base clocks while profiling, so the peaks below are measured at those clocks (1.5 GHz SM clock), and kernel times are slightly longer than in the benchmark.

| | Distance kernel | Top-k kernel |
| --- | ---: | ---: |
| Time | 244.8 ms | 2.85 ms |
| FP32 operations | 38.4 GFLOP | ~0 (comparisons only) |
| DRAM traffic | 51.6 GB | 0.40 GB |
| L2 traffic | 127.0 GB | 0.88 GB |
| Arithmetic intensity | 0.74 FLOP/byte | — |
| Achieved compute | 156.8 GFLOP/s (0.85% of 18.4 TFLOP/s) | — |
| Achieved DRAM bandwidth | 210.8 GB/s (62% of 341.7 GB/s) | 141.1 GB/s (41%) |

The ridge point of the device is 18,432 / 341.7 ≈ 54 FLOP/byte. The distance kernel takes 99% of the GPU time and sits at 0.74 FLOP/byte, about 70× below the ridge, so it is firmly memory-bound. It reaches 62% of the memory-bandwidth roof and less than 1% of FP32 peak.

The counters match the kernel's structure exactly:

- **FLOPs.** 12.8 billion FADD and 12.8 billion FFMA instructions for 100 × 10⁶ × 128 elements: 3 FLOPs per element (subtract, then multiply-add).
- **DRAM reads.** 51.2 GB is exactly 100 × 512 MB. Every query streams the entire database from DRAM; there is no reuse across queries.
- **Arithmetic intensity.** Per (query, vector) pair the kernel does 3d = 384 FLOPs and moves 4d = 512 bytes of database plus a 4-byte distance write: 384 / 516 = 0.744 FLOP/byte, matching the measured 0.74. Because nothing is reused, this intensity does not depend on the batch size, which is why larger batches do not help.
- **L2 traffic.** 127 GB passes through L2 against 51.6 GB from DRAM, about 2.5× more. We attribute this to the 512-byte stride between adjacent threads, which defeats L1 caching of database rows, but the counters alone do not prove the cause.

These numbers also bound what v3 could ever achieve. At the measured 342 GB/s, streaming 516 MB per query allows at most about 660 queries/s, and v3 already reaches about 400. Faster versions must therefore reduce the bytes moved per query, not just access memory more efficiently.

### What v3 tells the later versions

- **Reuse is the main lever.** Tiling the distance computation as a matrix multiply (v4) lets each database vector loaded on-chip serve many queries, raising arithmetic intensity roughly in proportion to the tile size.
- **The distance matrix is cheap now but will not stay cheap.** In v3 it is about 1.5% of DRAM traffic (4 MB written and 4 MB read back per query, against 512 MB of database reads). Once v4 removes the database traffic, writing and re-reading the B × N matrix becomes the dominant cost and caps intensity. That is what the fused top-k in v5 removes.
- **Exactness needs a tie-aware check.** SIFT1M's ground truth breaks ties differently from our kernels, so plain recall@k sits slightly below 1.0 for an exact search. The benchmark's tie-aware recall separates ties from real errors for every version.

### Reproducing

```bash
bash scripts/get_sift1m.sh                          # once, on a login node
sbatch slurm/v3_test.sbatch                         # correctness test
sbatch --export=ALL,VERSION=v3 slurm/bench.sbatch   # benchmark sweep -> results/benchmark.csv
sbatch slurm/ncu.sbatch                             # Nsight Compute -> results/ncu/, results/roofline.csv
```
