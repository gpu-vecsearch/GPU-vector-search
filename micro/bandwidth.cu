// Memory-bandwidth ceiling microbenchmark (roofline "memory roof" you can actually reach).
//
// Streams large device arrays with coalesced float4 accesses:
//   read  - sum every element (bytes moved = array size)
//   copy  - b[i] = a[i]       (bytes moved = 2 x array size)
// Reports the best of --repeat timed runs (after one warm-up) in GB/s and appends
// rows to results/micro.csv: device,test,value,unit
//
// Runs at normal (boost) clocks, unlike Nsight Compute, which locks base clocks.
//
// Usage: bandwidth [--mib SIZE_PER_ARRAY] [--repeat R] [--csv PATH]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "csv_append.h"
#include "cuda_check.cuh"

namespace {

constexpr const char* kHeader = "device,test,value,unit";

__global__ void read_kernel(const float4* __restrict__ a, std::size_t n4, float* __restrict__ sink) {
    float acc = 0.0f;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n4;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Keep the loads live without a reduction: one write per thread, almost never taken.
    if (acc == -1.0f) sink[0] = acc;
}

__global__ void copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, std::size_t n4) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n4;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        b[i] = a[i];
    }
}

__global__ void init_kernel(float4* a, std::size_t n4) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n4;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        a[i] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t mib = 1024;
    int repeat = 10;
    std::string csv = "results/micro.csv";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mib" && i + 1 < argc) mib = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--repeat" && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--mib SIZE_PER_ARRAY] [--repeat R] [--csv PATH]\n", argv[0]);
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    const std::size_t bytes = mib << 20;
    const std::size_t n4 = bytes / sizeof(float4);
    float4 *a = nullptr, *b = nullptr;
    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    init_kernel<<<sms * 8, 256>>>(a, n4);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const dim3 grid(sms * 8), block(256);

    auto best_ms = [&](auto launch) {
        launch();  // warm-up
        CUDA_CHECK_LAUNCH();
        float best = 1e30f;
        for (int r = 0; r < repeat; ++r) {
            CUDA_CHECK(cudaEventRecord(start));
            launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK_LAUNCH();
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            best = std::min(best, ms);
        }
        return static_cast<double>(best);
    };

    const double read_ms = best_ms([&] { read_kernel<<<grid, block>>>(a, n4, sink); });
    const double copy_ms = best_ms([&] { copy_kernel<<<grid, block>>>(a, b, n4); });
    const double read_gbs = bytes / (read_ms * 1e-3) / 1e9;
    const double copy_gbs = 2.0 * bytes / (copy_ms * 1e-3) / 1e9;

    std::printf("%s (%d SMs), %zu MiB per array, best of %d\n", prop.name, sms, mib, repeat);
    std::printf("  read: %8.1f GB/s\n  copy: %8.1f GB/s\n", read_gbs, copy_gbs);

    char row[256];
    std::snprintf(row, sizeof(row), "%s,dram_read,%.1f,GB/s", prop.name, read_gbs);
    append_csv_row(csv, kHeader, row);
    std::snprintf(row, sizeof(row), "%s,dram_copy,%.1f,GB/s", prop.name, copy_gbs);
    append_csv_row(csv, kHeader, row);
    std::printf("appended to %s\n", csv.c_str());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    return 0;
}
