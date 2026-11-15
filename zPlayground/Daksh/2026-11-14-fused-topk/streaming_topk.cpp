// CPU learning model for V5. This is not a CUDA implementation or benchmark.
// Question: can we select the same neighbors without storing every distance?
// Build from the repository root (binary stays outside the repository):
// clang++ -std=c++17 -O2 -Wall -Wextra -Wpedantic zPlayground/Daksh/2026-11-14-fused-topk/streaming_topk.cpp -o /tmp/daksh_streaming_topk
// Run: /tmp/daksh_streaming_topk
// On Euler, run only in an authorized Slurm compute-node allocation.
//
// Read: squared_l2 -> full_sort -> retain -> streaming -> chunked.
// Assumptions: finite, equal-dimensional vectors, 1 <= k <= N, chunk_size > 0.
// Checks: a hand-calculated fixture and 1,030 comparisons with full sorting,
// including ties, k=1, k=N, chunks smaller than k, and partial final chunks.
// Small integer inputs make squared sums exact in FP32; GPU rounding, SIFT1M
// recall, and GPU performance still need separate validation.
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Vector = std::vector<float>;
using Database = std::vector<Vector>;

struct Candidate {
    float distance;
    std::size_t index;
};

// A total order for our finite inputs: smaller distance, then smaller index.
bool better(const Candidate& a, const Candidate& b) {
    return a.distance < b.distance ||
           (a.distance == b.distance && a.index < b.index);
}

float squared_l2(const Vector& a, const Vector& b) {
    float sum = 0.0f;
    for (std::size_t d = 0; d < a.size(); ++d) {
        const float difference = a[d] - b[d];
        sum += difference * difference;
    }
    return sum;
}

// Independent selection reference: retain every distance and sort everything.
std::vector<Candidate> full_sort(const Database& database,
                                 const Vector& query, std::size_t k) {
    std::vector<Candidate> all;
    for (std::size_t i = 0; i < database.size(); ++i) {
        all.push_back({squared_l2(query, database[i]), i});
    }
    std::sort(all.begin(), all.end(), better);
    all.resize(k);
    return all;
}

// Invariant: best is sorted and holds the best min(k, seen) candidates.
// Insertion costs O(k); the intent here is clarity, not fast CPU selection.
void retain(std::vector<Candidate>& best, Candidate candidate, std::size_t k) {
    if (best.size() == k && !better(candidate, best.back())) {
        return;
    }
    // Make room before inserting so the list never holds more than k entries.
    if (best.size() == k) {
        best.pop_back();
    }
    const auto position = std::lower_bound(best.begin(), best.end(), candidate, better);
    best.insert(position, candidate);
}

// Compute a distance and immediately consider it for selection: no N-entry array.
std::vector<Candidate> streaming(const Database& database, const Vector& query,
                                 std::size_t k, std::size_t begin, std::size_t end) {
    std::vector<Candidate> best;
    best.reserve(k);
    for (std::size_t i = begin; i < end; ++i) {
        retain(best, {squared_l2(query, database[i]), i}, k);
    }
    return best;
}

// A discarded chunk candidate has at least k candidates ahead of it in that
// chunk, so it cannot be in the global top-k under the same distance/index order.
// Thus merging only chunk top-k lists preserves the exact answer.
//
// Intermediate storage, excluding inputs and final outputs:
// full matrix: B*N distances; streaming: k candidates; this CPU chunk model: 2*k.
// Parallel GPU chunks: up to B*C*k partial candidates plus per-block working
// state, where C is the chunk count. Candidates contain both distance and index.
// Savings require C*k << N; chunking alone does not guarantee lower memory use.
std::vector<Candidate> chunked(const Database& database, const Vector& query,
                               std::size_t k, std::size_t chunk_size) {
    std::vector<Candidate> merged;
    merged.reserve(k);
    for (std::size_t begin = 0; begin < database.size(); begin += chunk_size) {
        const std::size_t end = std::min(begin + chunk_size, database.size());
        const auto local = streaming(database, query, k, begin, end);
        // CPU model merges each chunk immediately. Parallel GPU chunks would
        // write partial lists to a compact buffer for a later merge kernel.
        for (const auto& candidate : local) {
            retain(merged, candidate, k);
        }
    }
    return merged;
}

void require_equal(const std::vector<Candidate>& actual,
                   const std::vector<Candidate>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Result length differs");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].index != expected[i].index ||
            actual[i].distance != expected[i].distance) {
            throw std::runtime_error("Neighbor index or squared distance differs");
        }
    }
}

std::size_t check_dataset(const Database& database, const Database& queries) {
    std::size_t checks = 0;
    for (const auto& query : queries) {
        for (std::size_t k = 1; k <= database.size(); ++k) {
            const auto expected = full_sort(database, query, k);
            require_equal(streaming(database, query, k, 0, database.size()), expected);
            ++checks;
            // Includes single-vector chunks, a partial last chunk, and C=1.
            for (const std::size_t chunk_size :
                 {std::size_t{1}, std::size_t{3}, database.size(), database.size() + 2}) {
                require_equal(chunked(database, query, k, chunk_size), expected);
                ++checks;
            }
        }
    }
    return checks;
}

int main() {
    try {
        // Tie order and distances are hand checked, independent of full_sort.
        const Database database{{0, 0}, {1, 0}, {-1, 0}, {0, 2},
                                {2, 0}, {1, 1}, {3, 0}};
        const Database queries{{0, 0}, {1, 0}, {0, 2}};
        const std::vector<Candidate> known{{0, 0}, {1, 1}, {1, 2}, {2, 5},
                                           {4, 3}, {4, 4}, {9, 6}};
        require_equal(full_sort(database, queries[0], database.size()), known);
        std::size_t checks = check_dataset(database, queries);

        // Integer-valued floats keep these small squared sums exact in FP32.
        std::mt19937 generator(759);
        auto make_vectors = [&generator](std::size_t count) {
            Database vectors(count, Vector(5));
            for (auto& vector : vectors) {
                for (auto& value : vector) {
                    value = static_cast<float>(static_cast<int>(generator() % 17) - 8);
                }
            }
            return vectors;
        };
        const auto generated_database = make_vectors(37);
        const auto generated_queries = make_vectors(5);
        checks += check_dataset(generated_database, generated_queries);

        std::cout << "PASS: hand-checked distance/tie fixture\n"
                  << "PASS: " << checks << " selection comparisons\n"
                  << "Includes k=1, k=N, ties, and partial final chunks\n"
                  << "Example: query (0,0), k=3 ->";
        for (const auto& candidate : streaming(database, queries[0], 3, 0, database.size())) {
            std::cout << " (index=" << candidate.index
                      << ", distance=" << candidate.distance << ')';
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
