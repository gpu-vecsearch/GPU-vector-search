// Learning: compute one squared L2 distance cooperatively across a warp.
// This isolates the distance step; it is not fused top-k or a benchmark.
// Unlike V5, this tiny experiment deliberately writes every pair's distance.
//
// Build from the repository root on an authorized Euler GPU compute allocation
// after loading the cluster's CUDA toolkit (module name must be confirmed):
// nvcc -std=c++17 -O2 -lineinfo zPlayground/Daksh/2026-11-15-warp-distance/warp_distance.cu -o /tmp/daksh_warp_distance
// Run within that allocation: /tmp/daksh_warp_distance
// Optional memory check there: compute-sanitizer /tmp/daksh_warp_distance
// Do not execute on an Euler login node. CUDA execution is unavailable on Mac.
//
// Reference: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(result) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Launch contract: exactly 32 threads per block, grid = (N, B).
// Each block is one complete warp and handles one query/database pair.
__global__ void warp_distance(const float* database, const float* queries,
                              int dimension, int database_count, float* distances) {
    const int lane = threadIdx.x;
    const int vector_index = blockIdx.x;
    const int query_index = blockIdx.y;

    // For d=128: lane 0 reads dimensions 0,32,64,96;
    // lane 1 reads 1,33,65,97, and so on. Neighboring lanes read neighboring
    // floats. For d<32, unused lanes contribute zero but still join the shuffle.
    float partial = 0.0f;
    for (int d = lane; d < dimension; d += 32) {
        const float difference = queries[query_index * dimension + d] -
                                 database[vector_index * dimension + d];
        partial += difference * difference;
    }

    // Exchange register values to reduce 32 partial sums in five steps.
    // All 32 lanes must execute each shuffle with this full membership mask.
    // After the final step only lane 0 is used as the complete sum.
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) {
        distances[query_index * database_count + vector_index] = partial;
    }
}

int main() {
    constexpr int database_count = 7;
    constexpr int query_count = 3;
    int checked_pairs = 0;

    // Exercise a partial warp's worth of dimensions and loop tails as well
    // as SIFT's d=128. Every launched warp still has all 32 participating lanes.
    for (const int dimension : {1, 31, 32, 33, 128, 129}) {
        std::vector<float> database(database_count * dimension);
        std::vector<float> queries(query_count * dimension);
        std::vector<float> actual(query_count * database_count);
        for (int i = 0; i < database_count; ++i) {
            for (int d = 0; d < dimension; ++d) {
                database[i * dimension + d] = static_cast<float>((i * 7 + d * 3) % 17 - 8);
            }
        }
        for (int q = 0; q < query_count; ++q) {
            for (int d = 0; d < dimension; ++d) {
                queries[q * dimension + d] = static_cast<float>((q * 5 + d * 3) % 17 - 8);
            }
        }

        float* device_database = nullptr;
        float* device_queries = nullptr;
        float* device_distances = nullptr;
        check_cuda(cudaMalloc(&device_database, database.size() * sizeof(float)), "allocate database");
        check_cuda(cudaMalloc(&device_queries, queries.size() * sizeof(float)), "allocate queries");
        check_cuda(cudaMalloc(&device_distances, actual.size() * sizeof(float)), "allocate distances");
        check_cuda(cudaMemcpy(device_database, database.data(), database.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy database");
        check_cuda(cudaMemcpy(device_queries, queries.data(), queries.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy queries");

        warp_distance<<<dim3(database_count, query_count), 32>>>(
            device_database, device_queries, dimension, database_count, device_distances);
        check_cuda(cudaGetLastError(), "launch warp_distance");
        check_cuda(cudaDeviceSynchronize(), "execute warp_distance");
        check_cuda(cudaMemcpy(actual.data(), device_distances, actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy distances");
        check_cuda(cudaFree(device_database), "free database");
        check_cuda(cudaFree(device_queries), "free queries");
        check_cuda(cudaFree(device_distances), "free distances");

        for (int q = 0; q < query_count; ++q) {
            for (int i = 0; i < database_count; ++i) {
                float expected = 0.0f;
                for (int d = 0; d < dimension; ++d) {
                    const float difference = queries[q * dimension + d] - database[i * dimension + d];
                    expected += difference * difference;
                }
                // Small integer inputs keep these sums exact in FP32 despite
                // the different reduction order. Real inputs need a tolerance.
                if (actual[q * database_count + i] != expected) {
                    std::cerr << "FAIL: d=" << dimension << " query=" << q << " vector=" << i
                              << " expected=" << expected
                              << " actual=" << actual[q * database_count + i] << '\n';
                    return EXIT_FAILURE;
                }
                ++checked_pairs;
            }
        }
        std::cout << "PASS: dimension=" << dimension << '\n';
    }
    std::cout << "PASS: " << checked_pairs << " GPU distances match CPU reference\n";
}
