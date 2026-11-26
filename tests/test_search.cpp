// Correctness test for any search() implementation.
//
// Generates random integer-valued vectors (like SIFT, values 0..255), so every
// squared L2 distance is exactly representable in float and the GPU result
// must match the CPU brute force exactly, including tie order (smaller index
// first).
//
// Usage: test_search [N] [B] [d] [k] [seed]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "search.h"

namespace {

void reference(const std::vector<float>& db, const std::vector<float>& qs,
               std::size_t n, std::size_t b, std::size_t d, std::size_t k,
               std::vector<int>& idx, std::vector<float>& dist) {
    std::vector<std::pair<float, int>> row(n);
    for (std::size_t q = 0; q < b; ++q) {
        for (std::size_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < d; ++j) {
                const float diff = qs[q * d + j] - db[i * d + j];
                acc += diff * diff;
            }
            row[i] = {acc, static_cast<int>(i)};
        }
        const std::size_t kk = std::min(k, n);
        std::partial_sort(row.begin(), row.begin() + kk, row.end());
        for (std::size_t r = 0; r < k; ++r) {
            idx[q * k + r] = r < kk ? row[r].second : -1;
            dist[q * k + r] = r < kk ? row[r].first : INFINITY;
        }
    }
}

bool run_case(std::size_t n, std::size_t b, std::size_t d, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> val(0, 255);
    std::vector<float> db(n * d), qs(b * d);
    for (auto& v : db) v = static_cast<float>(val(rng));
    for (auto& v : qs) v = static_cast<float>(val(rng));

    std::vector<int> got_idx(b * k), ref_idx(b * k);
    std::vector<float> got_dist(b * k), ref_dist(b * k);

    const auto t0 = std::chrono::steady_clock::now();
    search(db.data(), qs.data(), n, b, d, k, got_idx.data(), got_dist.data());
    const auto t1 = std::chrono::steady_clock::now();
    reference(db, qs, n, b, d, k, ref_idx, ref_dist);

    std::size_t idx_mismatch = 0, dist_mismatch = 0;
    for (std::size_t o = 0; o < b * k; ++o) {
        if (got_idx[o] != ref_idx[o]) ++idx_mismatch;
        if (got_dist[o] != ref_dist[o]) ++dist_mismatch;
        if ((got_idx[o] != ref_idx[o] || got_dist[o] != ref_dist[o]) &&
            idx_mismatch + dist_mismatch <= 5) {
            std::printf("  mismatch q=%zu r=%zu: got (%d, %.1f) want (%d, %.1f)\n",
                        o / k, o % k, got_idx[o], got_dist[o], ref_idx[o], ref_dist[o]);
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const bool ok = idx_mismatch == 0 && dist_mismatch == 0;
    std::printf("%s N=%zu B=%zu d=%zu k=%zu  search %.2f ms  idx mismatches %zu  dist mismatches %zu\n",
                ok ? "PASS" : "FAIL", n, b, d, k, ms, idx_mismatch, dist_mismatch);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100000;
    const std::size_t b = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 64;
    const std::size_t d = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 128;
    const std::size_t k = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 10;
    const unsigned seed = argc > 5 ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 759;

    bool ok = true;
    // Edge cases: odd sizes, k across template buckets, k > N padding.
    ok &= run_case(1000, 3, 7, 37, seed);
    ok &= run_case(4097, 5, 128, 100, seed + 1);
    ok &= run_case(50, 2, 16, 64, seed + 2);
    // Main case (from the command line).
    ok &= run_case(n, b, d, k, seed + 3);

    std::printf(ok ? "ALL PASS\n" : "SOME FAILED\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
