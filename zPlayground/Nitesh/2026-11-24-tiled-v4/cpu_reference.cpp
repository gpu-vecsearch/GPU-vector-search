#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <random>
#include <vector>

struct Candidate {
    float dist;
    int idx;
};

// Total order: smaller distance first, then smaller index.
static bool better(float da, int ia, float db, int ib) {
    return da < db || (da == db && ia < ib);
}
static bool better(const Candidate& a, const Candidate& b) {
    return better(a.dist, a.idx, b.dist, b.idx);
}

static std::vector<float> random_integer_matrix(int count, int dim, int max_value,
                                                unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> value(0, max_value);
    std::vector<float> m(static_cast<size_t>(count) * dim);
    for (float& x : m) x = static_cast<float>(value(rng));
    return m;
}

// Reference: direct distances and full sort.

static float direct_l2(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int j = 0; j < d; ++j) {
        const float diff = a[j] - b[j];
        s += diff * diff;
    }
    return s;
}

static std::vector<Candidate> full_sort_topk(const float* q, const std::vector<float>& X,
                                             int N, int d, int k) {
    std::vector<Candidate> all(N);
    for (int n = 0; n < N; ++n) all[n] = {direct_l2(q, &X[static_cast<size_t>(n) * d], d), n};
    std::sort(all.begin(), all.end(), [](const Candidate& a, const Candidate& b) {
        return better(a, b);
    });
    all.resize(k);
    return all;
}

// v4 step 1: norms and blocked expanded-form distances.

static std::vector<float> row_norms(const float* M, int count, int d) {
    std::vector<float> out(count);
    for (int r = 0; r < count; ++r) {
        float s = 0.0f;
        for (int j = 0; j < d; ++j) s += M[static_cast<size_t>(r) * d + j] * M[static_cast<size_t>(r) * d + j];
        out[r] = s;
    }
    return out;
}

// Fills D (B x N, leading dimension ldD) for queries Q (B x d) against
// database X (N x d). The (bt, nt) loops pick an output tile, the kt loop walks
// d in TILE steps, and the inner loops accumulate that slice into the tile.
// Edge tiles are clipped, so B, N and d need not be multiples of TILE.
template <int TILE>
static void blocked_distances(const float* Q, const float* X, const float* qnorm,
                              const float* xnorm, int B, int N, int d, float* D, int ldD) {
    std::vector<float> acc(TILE * TILE);
    for (int bt = 0; bt < B; bt += TILE) {
        const int bh = std::min(TILE, B - bt);
        for (int nt = 0; nt < N; nt += TILE) {
            const int nw = std::min(TILE, N - nt);
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int kt = 0; kt < d; kt += TILE) {
                const int kw = std::min(TILE, d - kt);
                for (int i = 0; i < bh; ++i) {
                    const float* q = Q + static_cast<size_t>(bt + i) * d + kt;
                    for (int j = 0; j < nw; ++j) {
                        const float* x = X + static_cast<size_t>(nt + j) * d + kt;
                        float s = 0.0f;
                        for (int kk = 0; kk < kw; ++kk) s += q[kk] * x[kk];
                        acc[i * TILE + j] += s;
                    }
                }
            }
            for (int i = 0; i < bh; ++i)
                for (int j = 0; j < nw; ++j)
                    D[static_cast<size_t>(bt + i) * ldD + nt + j] =
                        qnorm[bt + i] + xnorm[nt + j] - 2.0f * acc[i * TILE + j];
        }
    }
}

// v4 step 2: the GPU top-k algorithm, run sequentially.
//
// Element c of the row goes to "thread" c % THREADS, as in the strided GPU
// scan. Each thread keeps a sorted list of its k best; then k rounds take the
// best list head across all threads, and its owner advances.
// ids == nullptr means index = id_offset + column.
static void block_topk_model(const float* row, const int* ids, int row_len, int id_offset,
                             int k, int threads, Candidate* out) {
    std::vector<std::vector<Candidate>> lists(
        threads, std::vector<Candidate>(k, Candidate{INFINITY, INT_MAX}));
    for (int c = 0; c < row_len; ++c) {
        std::vector<Candidate>& best = lists[c % threads];
        const Candidate cand{row[c], ids ? ids[c] : id_offset + c};
        if (!better(cand, best[k - 1])) continue;
        int p = k - 1;
        while (p > 0 && better(cand, best[p - 1])) {
            best[p] = best[p - 1];
            --p;
        }
        best[p] = cand;
    }
    std::vector<int> head(threads, 0);
    for (int round = 0; round < k; ++round) {
        int winner = -1;
        Candidate w{INFINITY, INT_MAX};
        for (int t = 0; t < threads; ++t) {
            if (head[t] < k && (winner < 0 || better(lists[t][head[t]], w))) {
                w = lists[t][head[t]];
                winner = t;
            }
        }
        out[round] = w;
        if (winner >= 0) ++head[winner];
    }
}

// v4 pipeline: database chunks, per-chunk top-k, merge.
static std::vector<Candidate> chunked_search(const std::vector<float>& Q,
                                             const std::vector<float>& X, int B, int N, int d,
                                             int k, int chunk, int threads) {
    const std::vector<float> qnorm = row_norms(Q.data(), B, d);
    const std::vector<float> xnorm = row_norms(X.data(), N, d);
    const int num_chunks = (N + chunk - 1) / chunk;
    const int cand_ld = num_chunks * k;

    std::vector<float> tile(static_cast<size_t>(B) * chunk);
    std::vector<Candidate> cand(static_cast<size_t>(B) * cand_ld);
    std::vector<Candidate> local(k);
    for (int c = 0; c < num_chunks; ++c) {
        const int n0 = c * chunk;
        const int nc = std::min(chunk, N - n0);
        blocked_distances<16>(Q.data(), X.data() + static_cast<size_t>(n0) * d, qnorm.data(),
                              xnorm.data() + n0, B, nc, d, tile.data(), chunk);
        for (int b = 0; b < B; ++b) {
            block_topk_model(&tile[static_cast<size_t>(b) * chunk], nullptr, nc, n0, k, threads,
                             local.data());
            std::copy(local.begin(), local.end(), &cand[static_cast<size_t>(b) * cand_ld + c * k]);
        }
    }

    std::vector<Candidate> result(static_cast<size_t>(B) * k);
    std::vector<float> cd(cand_ld);
    std::vector<int> ci(cand_ld);
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < cand_ld; ++j) {
            cd[j] = cand[static_cast<size_t>(b) * cand_ld + j].dist;
            ci[j] = cand[static_cast<size_t>(b) * cand_ld + j].idx;
        }
        block_topk_model(cd.data(), ci.data(), cand_ld, 0, k, threads,
                         &result[static_cast<size_t>(b) * k]);
    }
    return result;
}

// Checks.

// Blocked expanded-form distances equal direct distances exactly.
static bool check_distances(int B, int N, int d, int max_value) {
    const auto Q = random_integer_matrix(B, d, max_value, 1);
    const auto X = random_integer_matrix(N, d, max_value, 2);
    const auto qn = row_norms(Q.data(), B, d);
    const auto xn = row_norms(X.data(), N, d);
    std::vector<float> D(static_cast<size_t>(B) * N);
    blocked_distances<16>(Q.data(), X.data(), qn.data(), xn.data(), B, N, d, D.data(), N);
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            if (D[static_cast<size_t>(b) * N + n] !=
                direct_l2(&Q[static_cast<size_t>(b) * d], &X[static_cast<size_t>(n) * d], d))
                return false;
    return true;
}

// Chunked search equals full-sort brute force exactly (distances and ids).
static bool check_search(int B, int N, int d, int k, int chunk, int threads, int max_value) {
    const auto Q = random_integer_matrix(B, d, max_value, 3 + N);
    const auto X = random_integer_matrix(N, d, max_value, 4 + k);
    const auto got = chunked_search(Q, X, B, N, d, k, chunk, threads);
    for (int b = 0; b < B; ++b) {
        const auto ref = full_sort_topk(&Q[static_cast<size_t>(b) * d], X, N, d, k);
        for (int j = 0; j < k; ++j) {
            const Candidate& g = got[static_cast<size_t>(b) * k + j];
            if (g.dist != ref[j].dist || g.idx != ref[j].idx) {
                std::printf("    query %d pos %d: got (%g, %d) want (%g, %d)\n", b, j, g.dist,
                            g.idx, ref[j].dist, ref[j].idx);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int failed = 0;
    auto report = [&](bool ok, const char* what) {
        failed += !ok;
        std::printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    };

    std::printf("blocked distances == direct distances\n");
    report(check_distances(64, 512, 128, 127), "B=64 N=512 d=128 (SIFT-like)");
    report(check_distances(37, 301, 100, 127), "B=37 N=301 d=100 (edge tiles in every dim)");
    report(check_distances(5, 7, 3, 127), "B=5 N=7 d=3 (smaller than one tile)");

    std::printf("chunked top-k search == full sort\n");
    report(check_search(16, 2000, 128, 10, 512, 128, 127), "k=10, chunk=512, last chunk partial");
    report(check_search(16, 2000, 128, 1, 512, 128, 127), "k=1");
    report(check_search(8, 300, 16, 100, 64, 32, 127), "k=100 > chunk=64 (chunks pad with inf)");
    report(check_search(8, 50, 16, 50, 16, 128, 127), "k=N, more threads than elements");
    report(check_search(16, 3000, 8, 20, 700, 128, 2), "heavy ties: values in [0, 2], d=8");
    report(check_search(4, 1000, 128, 10, 1000, 1, 127), "one chunk, one thread (pure insertion)");

    std::printf("\n%s\n", failed ? "SOME CHECKS FAILED" : "all checks passed");
    return failed ? 1 : 0;
}
