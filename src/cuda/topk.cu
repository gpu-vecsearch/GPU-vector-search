#include "topk.cuh"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_check.cuh"

// One block per row. Each thread scans a strided slice of the row (so global
// reads are coalesced) and keeps a sorted private top-k list. The block then
// merges those lists with k rounds of a block-wide argmin over the list heads.

namespace {

constexpr int kThreads = 256;  // must be a multiple of 32
constexpr unsigned kFullMask = 0xffffffffu;

// Order by distance, then by index. Indices are unique, so this is a total order.
__device__ __forceinline__ bool better(float da, int ia, float db, int ib) {
    return da < db || (da == db && ia < ib);
}

__device__ __forceinline__ void warp_argmin(float& d, int& i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float od = __shfl_down_sync(kFullMask, d, offset);
        const int oi = __shfl_down_sync(kFullMask, i, offset);
        if (better(od, oi, d, i)) {
            d = od;
            i = oi;
        }
    }
}

template <int MAX_K>
__global__ void topk_rows_kernel(const float* __restrict__ dist,
                                 std::size_t cols,
                                 int k,
                                 int* __restrict__ out_idx,
                                 float* __restrict__ out_dist) {
    const std::size_t row = blockIdx.x;
    const float* row_ptr = dist + row * cols;

    // Phase 1: private sorted top-k per thread (lives in local memory).
    float vals[MAX_K];
    int ids[MAX_K];
    int count = 0;

    for (std::size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = row_ptr[c];
        const int id = static_cast<int>(c);
        int p;
        if (count < k) {
            p = count++;
        } else if (better(v, id, vals[k - 1], ids[k - 1])) {
            p = k - 1;
        } else {
            continue;
        }
        while (p > 0 && better(v, id, vals[p - 1], ids[p - 1])) {
            vals[p] = vals[p - 1];
            ids[p] = ids[p - 1];
            --p;
        }
        vals[p] = v;
        ids[p] = id;
    }

    // Phase 2: k rounds of block argmin over each thread's current head.
    __shared__ float warp_d[kThreads / 32];
    __shared__ int warp_i[kThreads / 32];
    __shared__ float win_d;
    __shared__ int win_i;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    int head = 0;

    for (int r = 0; r < k; ++r) {
        float d = (head < count) ? vals[head] : INFINITY;
        int i = (head < count) ? ids[head] : INT_MAX;

        warp_argmin(d, i);
        if (lane == 0) {
            warp_d[warp] = d;
            warp_i[warp] = i;
        }
        __syncthreads();

        if (warp == 0) {
            d = (lane < num_warps) ? warp_d[lane] : INFINITY;
            i = (lane < num_warps) ? warp_i[lane] : INT_MAX;
            warp_argmin(d, i);
            if (lane == 0) {
                win_d = d;
                win_i = i;
            }
        }
        __syncthreads();

        const int wi = win_i;
        if (threadIdx.x == 0) {
            const std::size_t o = row * static_cast<std::size_t>(k) + r;
            out_idx[o] = (wi == INT_MAX) ? -1 : wi;
            out_dist[o] = (wi == INT_MAX) ? INFINITY : win_d;
        }
        // Exactly one thread owns the winning index; it pops its head.
        if (head < count && ids[head] == wi) {
            ++head;
        }
        // No trailing barrier needed: warp_d/warp_i are next written after
        // every thread has passed the second barrier above, and win_* is next
        // written only after the next round's first barrier.
    }
}

template <int MAX_K>
void launch(const float* dist, std::size_t rows, std::size_t cols, int k,
            int* out_idx, float* out_dist, cudaStream_t stream) {
    topk_rows_kernel<MAX_K><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        dist, cols, k, out_idx, out_dist);
    CUDA_CHECK_LAUNCH();
}

}  // namespace

void topk_rows_device(const float* dist,
                      std::size_t rows,
                      std::size_t cols,
                      std::size_t k,
                      int* out_idx,
                      float* out_dist,
                      cudaStream_t stream) {
    if (rows == 0 || k == 0) {
        return;
    }
    if (k > kTopkMaxK) {
        std::fprintf(stderr, "topk_rows_device: k=%zu exceeds max %zu\n", k, kTopkMaxK);
        std::exit(EXIT_FAILURE);
    }
    if (cols > static_cast<std::size_t>(INT_MAX)) {
        std::fprintf(stderr, "topk_rows_device: cols=%zu exceeds INT_MAX\n", cols);
        std::exit(EXIT_FAILURE);
    }
    if (rows > static_cast<std::size_t>(INT_MAX)) {
        std::fprintf(stderr, "topk_rows_device: rows=%zu exceeds grid limit\n", rows);
        std::exit(EXIT_FAILURE);
    }

    const int ki = static_cast<int>(k);
    if (k <= 16) {
        launch<16>(dist, rows, cols, ki, out_idx, out_dist, stream);
    } else if (k <= 32) {
        launch<32>(dist, rows, cols, ki, out_idx, out_dist, stream);
    } else if (k <= 64) {
        launch<64>(dist, rows, cols, ki, out_idx, out_dist, stream);
    } else if (k <= 128) {
        launch<128>(dist, rows, cols, ki, out_idx, out_dist, stream);
    } else {
        launch<256>(dist, rows, cols, ki, out_idx, out_dist, stream);
    }
}
