#include "search.h"

#include <algorithm>
#include <cstddef>

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "topk.cuh"

// v3: naive CUDA.
//
// Distance kernel: one thread per (query, database vector) pair computes
// ||q - x||^2 directly and writes it to a B x N distance matrix in global
// memory. Then a separate per-query top-k kernel (topk.cu) selects the k
// smallest entries from each row.
//
// Deliberately naive: no shared memory, no reuse of loaded vectors across
// threads, and reads of the database are strided by `dimension` across a warp
// (uncoalesced). The query element is the same for the whole block, so those
// reads are broadcasts. This is the baseline that v4 (tiled GEMM) improves on.
//
// A full B x N matrix does not fit for SIFT1M (10K x 1M floats = 40 GB), so
// queries are processed in chunks sized to the free device memory.

namespace {

constexpr int kDistThreads = 256;
constexpr std::size_t kMaxGridY = 65535;  // gridDim.y limit

__global__ void l2_naive_kernel(const float* __restrict__ database,
                                const float* __restrict__ queries,
                                std::size_t database_count,
                                std::size_t dimension,
                                float* __restrict__ distances) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t q = blockIdx.y;
    if (i >= database_count) {
        return;
    }

    const float* x = database + i * dimension;
    const float* qv = queries + q * dimension;

    float acc = 0.0f;
    for (std::size_t j = 0; j < dimension; ++j) {
        const float diff = qv[j] - x[j];
        acc = fmaf(diff, diff, acc);
    }
    distances[q * database_count + i] = acc;
}

// Number of queries whose distance rows fit in the remaining device memory.
std::size_t pick_chunk(std::size_t query_count, std::size_t database_count) {
    if (database_count == 0) {
        return std::min(query_count, kMaxGridY);
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const std::size_t budget = free_bytes / 10 * 8;  // leave ~20% headroom
    const std::size_t row_bytes = database_count * sizeof(float);
    std::size_t chunk = budget / row_bytes;
    chunk = std::min({chunk, query_count, kMaxGridY});
    return std::max<std::size_t>(chunk, 1);
}

}  // namespace

void search(const float* database,
            const float* queries,
            std::size_t database_count,
            std::size_t query_count,
            std::size_t dimension,
            std::size_t k,
            int* neighbor_indices,
            float* squared_l2_distances) {
    if (query_count == 0 || k == 0) {
        return;
    }

    float* d_database = nullptr;
    float* d_queries = nullptr;
    int* d_indices = nullptr;
    float* d_topk_dist = nullptr;
    float* d_distances = nullptr;

    const std::size_t db_bytes = database_count * dimension * sizeof(float);
    const std::size_t q_bytes = query_count * dimension * sizeof(float);
    const std::size_t out_count = query_count * k;

    CUDA_CHECK(cudaMalloc(&d_database, db_bytes));
    CUDA_CHECK(cudaMalloc(&d_queries, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_indices, out_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_topk_dist, out_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_database, database, db_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries, q_bytes, cudaMemcpyHostToDevice));

    const std::size_t chunk = pick_chunk(query_count, database_count);
    CUDA_CHECK(cudaMalloc(&d_distances, chunk * database_count * sizeof(float)));

    const unsigned grid_x =
        static_cast<unsigned>((database_count + kDistThreads - 1) / kDistThreads);

    for (std::size_t q0 = 0; q0 < query_count; q0 += chunk) {
        const std::size_t rows = std::min(chunk, query_count - q0);

        if (database_count > 0) {
            const dim3 grid(grid_x, static_cast<unsigned>(rows));
            l2_naive_kernel<<<grid, kDistThreads>>>(d_database,
                                                    d_queries + q0 * dimension,
                                                    database_count,
                                                    dimension,
                                                    d_distances);
            CUDA_CHECK_LAUNCH();
        }

        topk_rows_device(d_distances, rows, database_count, k,
                         d_indices + q0 * k, d_topk_dist + q0 * k);
    }

    CUDA_CHECK(cudaMemcpy(neighbor_indices, d_indices, out_count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(squared_l2_distances, d_topk_dist, out_count * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_topk_dist));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_database));
}
