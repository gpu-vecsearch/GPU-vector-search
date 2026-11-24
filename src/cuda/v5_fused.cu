#include "search.h"

// V5 correctness-first implementation; GPU validation/performance are pending.
// V5 conventions (the shared header does not yet specify these):
// - Host pointers, row-major vectors, sorted query-major outputs.
// - Distance ties use the smaller database index. Inputs must be finite and
//   squared distances must fit in FP32; no approximate/pruned search is used.
// - Supports 1 <= k <= min(N, 64), N <= INT_MAX, and positive dimension.
// - k=0 or B=0 is a no-op. Invalid arguments throw std::invalid_argument;
//   size overflow throws std::overflow_error; CUDA errors throw runtime_error.
// - Allocates/copies each call and returns after copying results to the host.
//   Wall-clock search() timing includes transfers; it is not kernel-only timing.
//
// A warp computes distances for one (query, database chunk). Lane 0 updates a
// shared top-k immediately, without writing the full B*N distance matrix.
// A second kernel merges chunk top-k lists. Serial insertion is intentional
// for this first version; parallel selection and tuning are later work.
// At most 32 queries are processed at once to bound intermediate allocation.
// Partial lists occupy batch * ceil(N/1024) * k distance/index pairs. At SIFT1M
// N=1,000,000, batch=32, k=64 this is about 15.3 MiB, versus about 122 MiB
// for just that batch's dense FP32 distance matrix (excluding all inputs).

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr int max_k = 64;
constexpr std::size_t chunk_size = 1024;
constexpr std::size_t query_batch_size = 32;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("V5 ") + operation + ": " + cudaGetErrorString(error));
    }
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("V5 array size overflow");
    }
    return a * b;
}

// Free allocations on both normal return and exceptions. Destructors must not
// throw; the checked copies/launches report CUDA errors before normal return.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), product(count, sizeof(T))),
                   "allocate device buffer");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
private:
    T* pointer_ = nullptr;
};

__device__ bool precedes(float a_distance, int a_index, float b_distance, int b_index) {
    return a_distance < b_distance || (a_distance == b_distance && a_index < b_index);
}

// Single-writer helper. Arrays start filled with (infinity, INT_MAX).
__device__ void retain(float* distances, int* indices, int k, float distance, int index) {
    if (!precedes(distance, index, distances[k - 1], indices[k - 1])) {
        return;
    }
    int position = k - 1;
    while (position > 0 && precedes(distance, index, distances[position - 1], indices[position - 1])) {
        distances[position] = distances[position - 1];
        indices[position] = indices[position - 1];
        --position;
    }
    distances[position] = distance;
    indices[position] = index;
}

// Launch exactly 32 threads/block. Whole warps take the same loop iterations
// and participate in each full-mask shuffle, including dimension < 32.
__global__ void fused_chunks(const float* database, const float* queries,
                             std::size_t database_count, std::size_t dimension,
                             std::size_t chunks, int k,
                             float* partial_distances, int* partial_indices) {
    __shared__ float best_distances[max_k];
    __shared__ int best_indices[max_k];
    const int lane = threadIdx.x;
    const std::size_t query = blockIdx.y;
    const std::size_t chunk = blockIdx.x;
    for (int rank = lane; rank < k; rank += 32) {
        best_distances[rank] = CUDART_INF_F;
        best_indices[rank] = INT_MAX;
    }
    __syncwarp();

    const std::size_t begin = chunk * chunk_size;
    const std::size_t end = (begin + chunk_size < database_count) ? begin + chunk_size : database_count;
    for (std::size_t index = begin; index < end; ++index) {
        float distance = 0.0f;
        for (std::size_t d = lane; d < dimension; d += 32) {
            const float difference = queries[query * dimension + d] - database[index * dimension + d];
            distance += difference * difference;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            distance += __shfl_down_sync(0xffffffffu, distance, offset);
        }
        if (lane == 0) {
            retain(best_distances, best_indices, k, distance, static_cast<int>(index));
        }
    }
    __syncwarp();  // Publish lane 0's final updates before cooperative writes.
    const std::size_t output = (query * chunks + chunk) * k;
    for (int rank = lane; rank < k; rank += 32) {
        partial_distances[output + rank] = best_distances[rank];
        partial_indices[output + rank] = best_indices[rank];
    }
}

// One thread/block per query for an intentionally simple first merge.
// A candidate omitted from a chunk already has k candidates ahead of it there;
// therefore merging only chunk top-k lists preserves the global top-k.
__global__ void merge_chunks(const float* partial_distances, const int* partial_indices,
                             std::size_t chunks, int k, float* output_distances, int* output_indices) {
    __shared__ float best_distances[max_k];
    __shared__ int best_indices[max_k];
    const std::size_t query = blockIdx.x;
    for (int rank = 0; rank < k; ++rank) {
        best_distances[rank] = CUDART_INF_F;
        best_indices[rank] = INT_MAX;
    }
    const std::size_t begin = query * chunks * k;
    for (std::size_t candidate = 0; candidate < chunks * k; ++candidate) {
        const int index = partial_indices[begin + candidate];
        if (index != INT_MAX) {  // Chunks smaller than k contain padding.
            retain(best_distances, best_indices, k, partial_distances[begin + candidate], index);
        }
    }
    for (int rank = 0; rank < k; ++rank) {
        output_distances[query * k + rank] = best_distances[rank];
        output_indices[query * k + rank] = best_indices[rank];
    }
}
}  // namespace

void search(const float* database, const float* queries,
            std::size_t database_count, std::size_t query_count,
            std::size_t dimension, std::size_t k,
            int* neighbor_indices, float* squared_l2_distances) {
    if (query_count == 0 || k == 0) {
        return;
    }
    if (database_count == 0 || dimension == 0 || k > database_count || k > max_k ||
        database_count > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("V5 requires N <= INT_MAX, d > 0, and 1 <= k <= min(N, 64)");
    }
    if (!database || !queries || !neighbor_indices || !squared_l2_distances) {
        throw std::invalid_argument("V5 requires non-null host input/output pointers");
    }

    const std::size_t database_elements = product(database_count, dimension);
    const std::size_t database_bytes = product(database_elements, sizeof(float));
    product(product(query_count, dimension), sizeof(float));
    const std::size_t output_elements = product(query_count, k);
    product(output_elements, sizeof(float));
    product(output_elements, sizeof(int));
    const std::size_t chunks = (database_count + chunk_size - 1) / chunk_size;
    const std::size_t batch_capacity = std::min(query_count, query_batch_size);
    const std::size_t partial_elements = product(product(batch_capacity, chunks), k);

    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "get device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "get device properties");
    if (chunks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        batch_capacity > static_cast<std::size_t>(properties.maxGridSize[1])) {
        throw std::invalid_argument("V5 grid exceeds device limits");
    }

    DeviceBuffer<float> device_database(database_elements);
    DeviceBuffer<float> device_queries(product(batch_capacity, dimension));
    DeviceBuffer<float> partial_distances(partial_elements);
    DeviceBuffer<int> partial_indices(partial_elements);
    DeviceBuffer<float> output_distances(product(batch_capacity, k));
    DeviceBuffer<int> output_indices(product(batch_capacity, k));
    check_cuda(cudaMemcpy(device_database.get(), database, database_bytes, cudaMemcpyHostToDevice),
               "copy database");

    for (std::size_t first = 0; first < query_count;) {
        const std::size_t batch = std::min(batch_capacity, query_count - first);
        check_cuda(cudaMemcpy(device_queries.get(), queries + first * dimension,
                              batch * dimension * sizeof(float), cudaMemcpyHostToDevice), "copy queries");
        fused_chunks<<<dim3(static_cast<unsigned>(chunks), static_cast<unsigned>(batch)), 32>>>(
            device_database.get(), device_queries.get(), database_count, dimension, chunks,
            static_cast<int>(k), partial_distances.get(), partial_indices.get());
        check_cuda(cudaGetLastError(), "launch fused chunks");
        merge_chunks<<<static_cast<unsigned>(batch), 1>>>(
            partial_distances.get(), partial_indices.get(), chunks, static_cast<int>(k),
            output_distances.get(), output_indices.get());
        check_cuda(cudaGetLastError(), "launch merge");
        // Same-stream ordering connects the two kernels. Blocking D2H copies
        // complete the computation and surface asynchronous execution errors.
        check_cuda(cudaMemcpy(neighbor_indices + first * k, output_indices.get(),
                              batch * k * sizeof(int), cudaMemcpyDeviceToHost), "copy result indices");
        check_cuda(cudaMemcpy(squared_l2_distances + first * k, output_distances.get(),
                              batch * k * sizeof(float), cudaMemcpyDeviceToHost), "copy result distances");
        first += batch;
    }
}
