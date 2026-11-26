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

// Tie-aware check of the same results. A returned neighbor counts as correct
// if its true squared L2 distance (recomputed here in double precision, not
// taken from search()) is <= the distance of the k-th ground-truth neighbor.
// This accepts a different but equally distant neighbor when distances tie
// (SIFT1M contains duplicate vectors), so an exact search scores 1.0.
struct TieAwareRecall {
    double recall = 0.0;           // fraction of returned neighbors within the k-th true distance
    std::size_t misses = 0;        // returned neighbors not in the ground-truth top-k
    std::size_t tied_misses = 0;   // ...of which are exactly as close as the k-th true neighbor
    std::size_t wrong = 0;         // ...of which are farther than the k-th true neighbor (real errors)
    std::size_t invalid = 0;       // out-of-range or duplicate indices
};

TieAwareRecall tie_aware_recall_at_k(const int* found,
                                     const std::int32_t* ground_truth,
                                     const float* database,
                                     std::size_t database_count,
                                     const float* queries,
                                     std::size_t dimension,
                                     std::size_t query_count,
                                     std::size_t k,
                                     std::size_t gt_stride);
