# v4 playground: tiled GEMM distances + separate top-k

## `cpu_reference.cpp`

CPU model of the v4 algorithm, checked against direct brute force.

v4 computes distances as `||q||^2 - 2 q·x + ||x||^2` with a tiled GEMM, then
selects the top-k per query in a separate pass. It processes the database in
chunks and merges the per-chunk candidates. This file runs each step on the
CPU with the same structure the CUDA kernels will use:

| Function | Mirrors |
| --- | --- |
| `blocked_distances` | the shared-memory tiled GEMM: tiles over queries, database and dimension |
| `block_topk_model` | the GPU top-k: strided per-thread sorted lists, then k rounds of argmin over list heads |
| `chunked_search` | the pipeline: database chunks → per-chunk top-k with global ids → merge pass |

Results are checked against direct `sum((q - x)^2)` distances and a full sort.
The data are small integers (as in SIFT1M), so FP32 sums are exact and results
are compared with `==`. Ties are broken by the smaller index.

```
g++ -std=c++17 -O2 -Wall -Wextra zPlayground/Nitesh/2026-11-24-tiled-v4/cpu_reference.cpp -o build/cpu_reference
./build/cpu_reference
```
