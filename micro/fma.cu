// FP32 FMA-throughput ceiling microbenchmark (roofline "compute roof" you can actually reach).
//
// Every thread runs 8 independent FMA chains entirely in registers, so the
// kernel does no memory traffic and enough independent work to fill every FP32
// pipe. FLOPs = 2 per FMA. Reports the best of --repeat timed runs (after one
// warm-up) in GFLOP/s and appends a row to results/micro.csv: device,test,value,unit
//
// Runs at normal (boost) clocks, unlike Nsight Compute, which locks base clocks.
//
// Usage: fma [--iters N] [--repeat R] [--csv PATH]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "csv_append.h"
#include "cuda_check.cuh"

namespace {

constexpr const char* kHeader = "device,test,value,unit";
constexpr int kChains = 8;

__global__ void fma_kernel(int iters, float mul, float add, float* __restrict__ out) {
    float acc[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
        acc[c] = static_cast<float>(threadIdx.x + c);
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            acc[c] = fmaf(acc[c], mul, add);  // mul < 1 keeps values bounded
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < kChains; ++c) sum += acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;  // keep the chains live
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 1 << 14;
    int repeat = 5;
    std::string csv = "results/micro.csv";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::max(1, std::atoi(argv[++i]));
        else if (a == "--repeat" && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--iters N] [--repeat R] [--csv PATH]\n", argv[0]);
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    const int threads = 256;
    const int blocks = sms * 8;  // 2048 threads (64 warps) per SM
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, static_cast<std::size_t>(blocks) * threads * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    fma_kernel<<<blocks, threads>>>(iters, 0.999f, 0.001f, out);  // warm-up
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        fma_kernel<<<blocks, threads>>>(iters, 0.999f, 0.001f, out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAUNCH();
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best = std::min(best, ms);
    }

    const double flops = 2.0 * kChains * static_cast<double>(iters) * blocks * threads;
    const double gflops = flops / (best * 1e-3) / 1e9;
    std::printf("%s (%d SMs), %d blocks x %d threads x %d chains x %d iters, best of %d\n",
                prop.name, sms, blocks, threads, kChains, iters, repeat);
    std::printf("  FP32 FMA: %10.1f GFLOP/s\n", gflops);

    char row[256];
    std::snprintf(row, sizeof(row), "%s,fp32_fma,%.1f,GFLOP/s", prop.name, gflops);
    append_csv_row(csv, kHeader, row);
    std::printf("appended to %s\n", csv.c_str());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
