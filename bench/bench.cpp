// Benchmark driver shared by every version.
//
// Each version defines the same search() symbol, so the Makefile links this
// driver once per version (build/bench_v1 ... build/bench_v6) and bakes the
// version name in with -DBENCH_VERSION. One run = one CSV row:
//   version,batch,k,latency_ms,throughput_qps,recall_at_k
//
// Timing: `warmup` untimed calls (CUDA context creation, first-touch), then
// `repeat` timed calls of search() on the first `batch` SIFT1M queries; the
// reported latency is the median. search() takes host pointers, so for GPU
// versions the time includes host<->device transfers of the database.
//
// Usage: bench_vN [--batch B] [--k K] [--data DIR] [--warmup W] [--repeat R]
//                 [--csv PATH | --no-csv] [--version LABEL]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fvecs.h"
#include "recall.h"
#include "search.h"
#include "timer.h"

#ifndef BENCH_VERSION
#define BENCH_VERSION "unknown"
#endif

namespace {

constexpr const char* kCsvHeader = "version,batch,k,latency_ms,throughput_qps,recall_at_k";

struct Options {
    std::size_t batch = 10000;
    std::size_t k = 10;
    std::string data_dir = "data/sift";
    int warmup = 1;
    int repeat = 5;
    std::string csv = "results/benchmark.csv";
    bool write_csv = true;
    std::string version = BENCH_VERSION;
};

[[noreturn]] void usage(const char* prog, const std::string& error = "") {
    if (!error.empty()) std::fprintf(stderr, "error: %s\n", error.c_str());
    std::fprintf(stderr,
                 "usage: %s [--batch B] [--k K] [--data DIR] [--warmup W] [--repeat R]\n"
                 "          [--csv PATH | --no-csv] [--version LABEL]\n"
                 "defaults: --batch 10000 --k 10 --data data/sift --warmup 1 --repeat 5\n"
                 "          --csv results/benchmark.csv --version %s\n",
                 prog, BENCH_VERSION);
    std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

std::size_t parse_count(const char* prog, const std::string& flag, const char* text, bool allow_zero) {
    char* end = nullptr;
    const long long v = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || (!allow_zero && v == 0)) {
        usage(prog, "invalid value for " + flag + ": " + text);
    }
    return static_cast<std::size_t>(v);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0], "missing value for " + a);
            return argv[++i];
        };
        if (a == "--batch") o.batch = parse_count(argv[0], a, next(), false);
        else if (a == "--k") o.k = parse_count(argv[0], a, next(), false);
        else if (a == "--data") o.data_dir = next();
        else if (a == "--warmup") o.warmup = static_cast<int>(parse_count(argv[0], a, next(), true));
        else if (a == "--repeat") o.repeat = static_cast<int>(parse_count(argv[0], a, next(), false));
        else if (a == "--csv") o.csv = next();
        else if (a == "--no-csv") o.write_csv = false;
        else if (a == "--version") o.version = next();
        else if (a == "-h" || a == "--help") usage(argv[0]);
        else usage(argv[0], "unknown argument " + a);
    }
    return o;
}

void append_csv(const std::string& path, const std::string& row) {
    // Inspect the existing file: write the header if it is empty, and make sure
    // the new row starts on its own line.
    bool empty = true;
    bool ends_with_newline = true;
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (in && in.tellg() > 0) {
            empty = false;
            in.seekg(-1, std::ios::end);
            char last = '\0';
            in.get(last);
            ends_with_newline = (last == '\n');
        }
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot open " + path + " for appending");
    }
    if (empty) out << kCsvHeader << '\n';
    else if (!ends_with_newline) out << '\n';
    out << row << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);

    try {
        const std::string dir = opt.data_dir + "/";
        Timer load_timer;
        load_timer.start();
        const FloatVecs base = read_fvecs(dir + "sift_base.fvecs");
        const FloatVecs queries = read_fvecs(dir + "sift_query.fvecs", opt.batch);
        const IntVecs truth = read_ivecs(dir + "sift_groundtruth.ivecs", opt.batch);
        const double load_ms = load_timer.elapsed_ms();

        if (queries.count < opt.batch) {
            usage(argv[0], "--batch " + std::to_string(opt.batch) + " exceeds the " +
                               std::to_string(queries.count) + " available queries");
        }
        if (queries.dim != base.dim) {
            throw std::runtime_error("query and base dimensions differ");
        }
        if (opt.k > truth.dim) {
            usage(argv[0], "--k " + std::to_string(opt.k) + " exceeds ground-truth depth " +
                               std::to_string(truth.dim));
        }

        std::printf("[%s] N=%zu d=%zu batch=%zu k=%zu  (loaded in %.0f ms)\n",
                    opt.version.c_str(), base.count, base.dim, opt.batch, opt.k, load_ms);

        std::vector<int> indices(opt.batch * opt.k);
        std::vector<float> distances(opt.batch * opt.k);
        auto run = [&]() {
            search(base.data.data(), queries.data.data(), base.count, opt.batch, base.dim,
                   opt.k, indices.data(), distances.data());
        };

        for (int i = 0; i < opt.warmup; ++i) {
            run();
        }

        std::vector<double> times;
        times.reserve(opt.repeat);
        for (int i = 0; i < opt.repeat; ++i) {
            Timer t;
            t.start();
            run();
            times.push_back(t.elapsed_ms());
        }
        std::sort(times.begin(), times.end());
        const double median_ms = times[times.size() / 2];
        const double qps = static_cast<double>(opt.batch) / (median_ms / 1000.0);
        const double recall =
            recall_at_k(indices.data(), truth.data.data(), opt.batch, opt.k, truth.dim);

        std::printf("  latency ms: median %.3f  min %.3f  max %.3f  (%d runs)\n",
                    median_ms, times.front(), times.back(), opt.repeat);
        std::printf("  throughput: %.1f queries/s   recall@%zu: %.4f\n", qps, opt.k, recall);

        if (opt.write_csv) {
            char row[256];
            std::snprintf(row, sizeof(row), "%s,%zu,%zu,%.3f,%.1f,%.4f", opt.version.c_str(),
                          opt.batch, opt.k, median_ms, qps, recall);
            append_csv(opt.csv, row);
            std::printf("  appended to %s\n", opt.csv.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
