#include "recall.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

double recall_at_k(const int* found,
                   const std::int32_t* ground_truth,
                   std::size_t query_count,
                   std::size_t k,
                   std::size_t gt_stride) {
    if (k > gt_stride) {
        throw std::invalid_argument("recall_at_k: k exceeds ground-truth depth");
    }
    if (query_count == 0 || k == 0) {
        return 0.0;
    }

    std::size_t hits = 0;
    std::vector<int> truth(k);
    for (std::size_t q = 0; q < query_count; ++q) {
        const std::int32_t* gt = ground_truth + q * gt_stride;
        std::copy(gt, gt + k, truth.begin());
        std::sort(truth.begin(), truth.end());

        const int* got = found + q * k;
        for (std::size_t r = 0; r < k; ++r) {
            if (std::binary_search(truth.begin(), truth.end(), got[r])) {
                ++hits;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(query_count * k);
}

namespace {

double squared_l2(const float* a, const float* b, std::size_t d) {
    double acc = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
        const double diff = static_cast<double>(a[j]) - static_cast<double>(b[j]);
        acc += diff * diff;
    }
    return acc;
}

}  // namespace

TieAwareRecall tie_aware_recall_at_k(const int* found,
                                     const std::int32_t* ground_truth,
                                     const float* database,
                                     std::size_t database_count,
                                     const float* queries,
                                     std::size_t dimension,
                                     std::size_t query_count,
                                     std::size_t k,
                                     std::size_t gt_stride) {
    if (k > gt_stride) {
        throw std::invalid_argument("tie_aware_recall_at_k: k exceeds ground-truth depth");
    }
    TieAwareRecall out;
    if (query_count == 0 || k == 0) {
        return out;
    }

    std::size_t hits = 0;
    std::vector<int> truth(k);
    std::vector<int> got(k);
    for (std::size_t q = 0; q < query_count; ++q) {
        const float* qv = queries + q * dimension;
        const std::int32_t* gt = ground_truth + q * gt_stride;
        const double kth = squared_l2(qv, database + static_cast<std::size_t>(gt[k - 1]) * dimension,
                                      dimension);

        std::copy(gt, gt + k, truth.begin());
        std::sort(truth.begin(), truth.end());
        std::copy(found + q * k, found + q * k + k, got.begin());
        std::sort(got.begin(), got.end());

        for (std::size_t r = 0; r < k; ++r) {
            const int id = got[r];
            if (id < 0 || static_cast<std::size_t>(id) >= database_count ||
                (r > 0 && got[r - 1] == id)) {
                ++out.invalid;
                continue;
            }
            const bool in_truth = std::binary_search(truth.begin(), truth.end(), id);
            const double dist = squared_l2(qv, database + static_cast<std::size_t>(id) * dimension,
                                           dimension);
            if (!in_truth) {
                ++out.misses;
                if (dist == kth) ++out.tied_misses;
                if (dist > kth) ++out.wrong;
            }
            if (in_truth || dist <= kth) {
                ++hits;
            }
        }
    }
    out.recall = static_cast<double>(hits) / static_cast<double>(query_count * k);
    return out;
}
