# Proposal

Home department: Computer Science

Project Title: GPU vector search, from naive to Tensor Core

GitHub link: https://github.com/gpu-vecsearch/GPU-vector-search

Problem statement: Given a database of N embedding vectors and a batch of B query vectors of dimension d, return the k nearest database vectors for each query by L2 distance (exact top-k search). This is the core operation of vector databases and retrieval systems. 
We will implement it as a sequence of increasingly optimized versions:
Baseline: naive C++
Multicore: OpenMP
Naive CUDA: one thread per (query, vector) pair
Tiled CUDA: distances computed as a shared-memory tiled GEMM, followed by a separate per-query top-k kernel
Fused top-k: selection done inside the distance kernel with warp-level primitives, so the B×N distance matrix is never written to global memory
Tensor Cores: distance computation with WMMA in FP16
Every version is then placed on a roofline model and compared against FAISS (GPU flat index).
Stretch goals (only once version 6 is done): SIMD/AVX single-core optimization; MPI sharding across Euler nodes; INT8 Tensor Cores; the higher-dimensional GIST1M dataset; approximate IVF search with deadline-aware ("anytime") retrieval that returns the best top-k found when a time budget expires.
Motivation/Rationale: Vector search underlies retrieval-augmented generation, semantic search, and recommendation, and production vector databases increasingly run it on GPUs. Exact search is also a clean testbed for ECE 759 ideas. 

Expanding ‖q − x‖² = ‖q‖² − 2 q·x + ‖x‖² turns the expensive part into a matrix multiply (Q·Xᵀ), which builds directly on the matmul work in HW02 and HW03, while top-k selection adds an irregular parallel problem on top. 

The arithmetic intensity grows with the query batch (about B/2 FLOP/byte in FP32), so the same problem moves from memory-bound to compute-bound as B grows. Writing out the full distance matrix caps intensity near d/2, which is exactly what the fused top-k removes. This gives a concrete roofline story. 

Our team is interested in data systems and AI workloads, and this project shows how the kernels inside them are built.
 
Explain how you contemplate going about it:
CPU and GPU parallelism: OpenMP on a 20-core Euler node; CUDA on Euler GPUs.
Data: SIFT1M (1M × 128-d vectors, 10K queries, with ground truth).
Shared interface: every version implements the same search() function, and a single benchmark driver times any version and checks recall against the ground truth. This lets five people work in parallel.
Algorithms: shared-memory tiled GEMM; warp-shuffle top-k selection; WMMA 16×16×16 fragments.
Libraries: every version we compare is our own code. FAISS (GPU flat index) and cuBLAS serve only as reference baselines, not as building blocks.
Timeline:
10/31 - 11/6: data loader, recall checker, benchmark driver; serial baseline; OpenMP
11/7 - 11/13: naive CUDA; tiled GEMM with separate top-k
11/14 - 11/20: fused top-k; WMMA FP16
11/21 - 11/27: roofline measurements, FAISS comparison, plots; code freeze 11/27
11/28 - 12/7: presentation prep and final report
 ECE 759 aspects the proposed work draws on:
Memory hierarchy and loop ordering (HW02)
OpenMP parallel loops and scaling analysis (HW03)
CUDA thread hierarchy, memory coalescing, shared-memory tiling
Warp-level primitives and parallel selection
Tensor Cores (WMMA)
Timing methodology, Slurm on Euler, roofline analysis

Deliverables: 
6 implementations in the team repo
Results table: time and queries/second for every version on SIFT1M across batch sizes
Scaling plots: OpenMP threads (1-20) and query batch size
Roofline plot placing each CUDA kernel
Comparison against FAISS GPU, and against cuBLAS for the GEMM step
Presentation and final report
How you will demonstrate what you accomplished: 
Correctness: every exact version must reach recall@k = 1.0 (up to distance ties) against the SIFT1M ground truth; the FP16 version reports any recall loss explicitly.
Performance: speedup of each version over the previous one and over the baseline, timed per the course timing methodology on Euler compute nodes through Slurm.
Explanation: each speedup, or lack of one, tied to the roofline: where the kernel sits and what bounds it.
Progress: steady commits from all members, with a tagged milestone for each version.
