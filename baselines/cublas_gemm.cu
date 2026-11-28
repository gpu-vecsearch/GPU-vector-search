// cuBLAS reference for the distance GEMM step (comparison only, not a version).
//
// Times C = Q * X^T, the dominant term of ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2,
// at SIFT1M shapes: Q is B x d (queries), X is N x d (database), C is B x N,
// all row-major. v4's tiled GEMM is compared against these numbers.
//
// Precisions:
//   fp32  cublasSgemm, default math (no TF32), the exact-search reference
//   fp16  cublasGemmEx, FP16 inputs, FP32 accumulate (Tensor Cores), the v6 reference
//
// Data are synthetic integers 0..255 (like SIFT). Every dot product is then an
// integer below 2^24, exact in FP32 for both precisions, so results are checked
// exactly against the CPU on sampled entries.
//
// Usage: cublas_gemm [--n N] [--d D] [--batches 1,10,...] [--repeat R] [--csv PATH]
// Appends rows to results/cublas_gemm.csv:
//   precision,batch,n,d,time_ms,gflops

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "csv_append.h"
#include "cuda_check.cuh"

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t s_ = (call);                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                        \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(s_), \
                         __FILE__, __LINE__);                                     \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace {

constexpr const char* kHeader = "precision,batch,n,d,time_ms,gflops";

// Deterministic integer value in 0..255 for element i of a matrix (same on host and device).
__host__ __device__ inline float synth(std::size_t i, std::uint32_t seed) {
    std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u ^ seed ^
                      static_cast<std::uint32_t>(i >> 32) * 40503u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return static_cast<float>(h & 255u);
}

__global__ void fill_kernel(float* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        p[i] = synth(i, seed);
    }
}

__global__ void to_half_kernel(const float* in, __half* out, std::size_t n) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        out[i] = __float2half(in[i]);
    }
}

std::vector<std::size_t> parse_list(const std::string& s) {
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::strtoull(item.c_str(), nullptr, 10));
    }
    return out;
}

struct Options {
    std::size_t n = 1000000;
    std::size_t d = 128;
    // B * N must stay below 2^31 elements for the 32-bit cuBLAS API; 2048 x 1M fits.
    std::vector<std::size_t> batches{1, 10, 100, 1000, 2048};
    int repeat = 5;
    std::string csv = "results/cublas_gemm.csv";
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--n") o.n = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--d") o.d = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--batches") o.batches = parse_list(next());
        else if (a == "--repeat") o.repeat = std::max(1, std::atoi(next().c_str()));
        else if (a == "--csv") o.csv = next();
        else {
            std::fprintf(stderr, "usage: %s [--n N] [--d D] [--batches 1,10,100] [--repeat R] [--csv PATH]\n", argv[0]);
            std::exit(a == "-h" || a == "--help" ? 0 : 1);
        }
    }
    return o;
}

void launch_fill(float* p, std::size_t n, std::uint32_t seed) {
    fill_kernel<<<1024, 256>>>(p, n, seed);
    CUDA_CHECK_LAUNCH();
}

void launch_to_half(const float* in, __half* out, std::size_t n) {
    to_half_kernel<<<1024, 256>>>(in, out, n);
    CUDA_CHECK_LAUNCH();
}

// Row-major C (B x N) = Q (B x d) * X^T is column-major C^T (N x B) = X * Q^T.
// Row-major X / Q are column-major X^T (d x N) / Q^T (d x B), so op(A) = T on X.
void gemm(cublasHandle_t h, bool fp16, std::size_t b, std::size_t n, std::size_t d,
          const float* X, const float* Q, const __half* Xh, const __half* Qh, float* C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m_ = static_cast<int>(n), n_ = static_cast<int>(b), k_ = static_cast<int>(d);
    if (!fp16) {
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m_, n_, k_, &alpha, X, k_, Q, k_,
                                 &beta, C, m_));
    } else {
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m_, n_, k_, &alpha, Xh, CUDA_R_16F,
                                  k_, Qh, CUDA_R_16F, k_, &beta, C, CUDA_R_32F, m_,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}

// Check a few entries of C exactly against a CPU dot product.
bool verify(const float* dC, std::size_t b, std::size_t n, std::size_t d) {
    const std::size_t rows[] = {0, b / 2, b - 1};
    const std::size_t cols[] = {0, n / 3, n - 1};
    for (std::size_t q : rows) {
        for (std::size_t x : cols) {
            float got = 0.0f;
            CUDA_CHECK(cudaMemcpy(&got, dC + q * n + x, sizeof(float), cudaMemcpyDeviceToHost));
            double want = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                want += static_cast<double>(synth(q * d + j, 2u)) * synth(x * d + j, 1u);
            }
            if (static_cast<double>(got) != want) {
                std::fprintf(stderr, "  mismatch C[%zu][%zu]: got %.1f want %.1f\n", q, x, got, want);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("cuBLAS GEMM baseline on %s: N=%zu d=%zu\n", prop.name, opt.n, opt.d);

    const std::size_t max_b = *std::max_element(opt.batches.begin(), opt.batches.end());
    float *X = nullptr, *Q = nullptr;
    __half *Xh = nullptr, *Qh = nullptr;
    CUDA_CHECK(cudaMalloc(&X, opt.n * opt.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Q, max_b * opt.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Xh, opt.n * opt.d * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&Qh, max_b * opt.d * sizeof(__half)));
    launch_fill(X, opt.n * opt.d, 1u);
    launch_fill(Q, max_b * opt.d, 2u);
    launch_to_half(X, Xh, opt.n * opt.d);
    launch_to_half(Q, Qh, max_b * opt.d);
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));  // FP32 stays FP32 (no TF32)

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    bool all_ok = true;
    for (std::size_t b : opt.batches) {
        if (b == 0 || b * opt.n >= (1ull << 31)) {
            std::printf("  skip batch %zu: B*N must be in (0, 2^31)\n", b);
            continue;
        }
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t c_bytes = b * opt.n * sizeof(float);
        if (c_bytes > free_bytes / 10 * 9) {
            std::printf("  skip batch %zu: output needs %.1f GB, %.1f GB free\n", b, c_bytes / 1e9,
                        free_bytes / 1e9);
            continue;
        }
        float* C = nullptr;
        CUDA_CHECK(cudaMalloc(&C, c_bytes));

        for (int p = 0; p < 2; ++p) {
            const bool fp16 = (p == 1);
            gemm(handle, fp16, b, opt.n, opt.d, X, Q, Xh, Qh, C);  // warm-up
            CUDA_CHECK(cudaDeviceSynchronize());
            const bool ok = verify(C, b, opt.n, opt.d);
            all_ok &= ok;

            std::vector<float> times;
            for (int r = 0; r < opt.repeat; ++r) {
                CUDA_CHECK(cudaEventRecord(start));
                gemm(handle, fp16, b, opt.n, opt.d, X, Q, Xh, Qh, C);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                times.push_back(ms);
            }
            std::sort(times.begin(), times.end());
            const double ms = times[times.size() / 2];
            const double gflops = 2.0 * b * opt.n * opt.d / (ms * 1e-3) / 1e9;
            const char* prec = fp16 ? "fp16" : "fp32";
            std::printf("  %s batch %5zu: %10.3f ms  %9.1f GFLOP/s  check %s\n", prec, b, ms, gflops,
                        ok ? "ok" : "FAILED");

            char row[160];
            std::snprintf(row, sizeof(row), "%s,%zu,%zu,%zu,%.4f,%.1f", prec, b, opt.n, opt.d, ms, gflops);
            append_csv_row(opt.csv, kHeader, row);
        }
        CUDA_CHECK(cudaFree(C));
    }
    std::printf("appended to %s\n", opt.csv.c_str());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(Qh));
    CUDA_CHECK(cudaFree(Xh));
    CUDA_CHECK(cudaFree(Q));
    CUDA_CHECK(cudaFree(X));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
