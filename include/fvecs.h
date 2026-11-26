#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Row-major vectors loaded from a TEXMEX .fvecs / .ivecs file.
// On disk every record is: int32 dim, then dim values (float32 or int32).
template <typename T>
struct VecSet {
    std::vector<T> data;  // count * dim values, row-major
    std::size_t count = 0;
    std::size_t dim = 0;

    const T* row(std::size_t i) const { return data.data() + i * dim; }
};

using FloatVecs = VecSet<float>;
using IntVecs = VecSet<std::int32_t>;

constexpr std::size_t kAllRows = std::numeric_limits<std::size_t>::max();

// Read up to max_rows records. Throws std::runtime_error on a missing or
// malformed file (inconsistent dims, truncated record).
FloatVecs read_fvecs(const std::string& path, std::size_t max_rows = kAllRows);
IntVecs read_ivecs(const std::string& path, std::size_t max_rows = kAllRows);
