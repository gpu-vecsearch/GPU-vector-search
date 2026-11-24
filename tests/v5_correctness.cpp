// Standalone V5 correctness driver; does not depend on unfinished shared bench.
// Build/run via: sbatch slurm/v5_correctness.sbatch (on Euler, from repo root).
// Finite small integers keep distances exact despite FP32 reduction ordering.
// This is not a SIFT1M recall test or a benchmark.
#include "search.h"

#include <algorithm>
#include <climits>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
struct Candidate {
    float distance;
    int index;
};

void check_case(std::size_t n, std::size_t b, std::size_t d, std::size_t k, bool all_ties = false) {
    std::vector<float> database(n * d);
    std::vector<float> queries(b * d);
    for (std::size_t i = 0; i < database.size(); ++i) {
        database[i] = all_ties ? 0.0f : static_cast<float>(static_cast<int>((i * 13 + i / d * 7) % 17) - 8);
    }
    for (std::size_t i = 0; i < queries.size(); ++i) {
        queries[i] = all_ties ? 1.0f : static_cast<float>(static_cast<int>((i * 3 + i / d * 5) % 19) - 9);
    }
    // Ensure a unique late winner for the first query on a chunk boundary tail.
    if (!all_ties) {
        for (std::size_t dimension = 0; dimension < d; ++dimension) {
            queries[dimension] = 30.0f;
            database[(n - 1) * d + dimension] = 30.0f;
        }
    }
    std::vector<int> indices(b * k, -1);
    std::vector<float> distances(b * k, -1.0f);
    search(database.data(), queries.data(), n, b, d, k, indices.data(), distances.data());
    for (std::size_t query = 0; query < b; ++query) {
        std::vector<Candidate> expected;
        expected.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            float distance = 0.0f;
            for (std::size_t dimension = 0; dimension < d; ++dimension) {
                const float delta = queries[query * d + dimension] - database[i * d + dimension];
                distance += delta * delta;
            }
            expected.push_back({distance, static_cast<int>(i)});
        }
        std::sort(expected.begin(), expected.end(), [](Candidate a, Candidate c) {
            return a.distance < c.distance || (a.distance == c.distance && a.index < c.index);
        });
        for (std::size_t rank = 0; rank < k; ++rank) {
            const std::size_t output = query * k + rank;
            if (indices[output] != expected[rank].index || distances[output] != expected[rank].distance) {
                std::cerr << "Mismatch: N=" << n << " B=" << b << " d=" << d << " k=" << k
                          << " query=" << query << " rank=" << rank
                          << " expected=(" << expected[rank].index << ',' << expected[rank].distance
                          << ") actual=(" << indices[output] << ',' << distances[output] << ")\n";
                throw std::runtime_error("V5 disagrees with full-sort reference");
            }
        }
    }
    std::cout << "PASS: N=" << n << " B=" << b << " d=" << d << " k=" << k
              << (all_ties ? " all ties" : "") << '\n';
}

template <typename Exception, typename Function>
void expect_exception(Function operation) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("Expected argument validation exception");
}

void check_arguments() {
    float input = 0.0f;
    float output_distance = -1.0f;
    int output_index = -1;
    search(nullptr, nullptr, 0, 0, 0, 1, nullptr, nullptr);
    search(nullptr, nullptr, 0, 1, 0, 0, nullptr, nullptr);
    expect_exception<std::invalid_argument>([&] {
        search(&input, &input, 1, 1, 0, 1, &output_index, &output_distance);
    });
    expect_exception<std::invalid_argument>([&] {
        search(&input, &input, 1, 1, 1, 2, &output_index, &output_distance);
    });
    expect_exception<std::invalid_argument>([&] {
        search(&input, &input, 65, 1, 1, 65, &output_index, &output_distance);
    });
    expect_exception<std::invalid_argument>([&] {
        search(nullptr, &input, 1, 1, 1, 1, &output_index, &output_distance);
    });
    expect_exception<std::invalid_argument>([&] {
        search(&input, &input, static_cast<std::size_t>(INT_MAX) + 1, 1, 1, 1,
               &output_index, &output_distance);
    });
    expect_exception<std::overflow_error>([&] {
        search(&input, &input, 1, 1, std::numeric_limits<std::size_t>::max(), 1,
               &output_index, &output_distance);
    });
    std::cout << "PASS: no-op and invalid argument handling\n";
}
}  // namespace

int main() {
    try {
        check_arguments();
        check_case(1, 1, 1, 1);
        for (const std::size_t d : {1, 31, 32, 33, 128, 129}) {
            check_case(7, 3, d, 1);
            check_case(7, 3, d, 7);
        }
        check_case(1024, 2, 128, 10);   // Exactly one full chunk.
        check_case(1025, 3, 128, 64);   // One-element tail: padding and late winner.
        check_case(2051, 33, 33, 10);   // Three chunks and two query batches.
        check_case(2051, 3, 128, 64, true);  // Equal distances across chunks.
        check_case(64, 2, 128, 64);     // k=N at the supported maximum.
        std::cout << "PASS: all V5 correctness cases\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
