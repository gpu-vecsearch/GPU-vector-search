#pragma once

#include <cstddef>

// Return the k nearest database vectors for each query under squared L2 distance.
// Output arrays are row-major with query_count * k entries.
void search(const float* database,
            const float* queries,
            std::size_t database_count,
            std::size_t query_count,
            std::size_t dimension,
            std::size_t k,
            int* neighbor_indices,
            float* squared_l2_distances);