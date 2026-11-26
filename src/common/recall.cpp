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
