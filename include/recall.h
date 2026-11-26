#pragma once

#include <cstddef>
#include <cstdint>

// Mean recall@k over query_count queries: for each query, the fraction of the
// true top-k (first k entries of its ground-truth row) that appear anywhere in
// the k returned neighbors.
//
// found:        query_count x k, row-major (output of search()).
// ground_truth: query_count x gt_stride, row-major (SIFT1M: gt_stride = 100).
// Requires k <= gt_stride.
double recall_at_k(const int* found,
                   const std::int32_t* ground_truth,
                   std::size_t query_count,
                   std::size_t k,
                   std::size_t gt_stride);
