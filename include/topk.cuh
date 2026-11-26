#pragma once

#include <cstddef>

#include <cuda_runtime.h>

// Largest k supported by topk_rows_device().
constexpr std::size_t kTopkMaxK = 256;

// Per-row top-k selection on a row-major rows x cols device matrix.
//
// For each row r, writes the k smallest values (ascending) and their column
// indices to out_dist[r * k .. r * k + k) and out_idx[r * k .. r * k + k).
// Ties are broken by the smaller column index, so results are deterministic.
// If k > cols, the remaining slots are filled with index -1 and +inf.
//
// All pointers are device pointers. Requires k <= kTopkMaxK and cols <= INT_MAX.
void topk_rows_device(const float* dist,
                      std::size_t rows,
                      std::size_t cols,
                      std::size_t k,
                      int* out_idx,
                      float* out_dist,
                      cudaStream_t stream = 0);
