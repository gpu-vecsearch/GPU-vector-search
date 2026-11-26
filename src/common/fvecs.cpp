#include "fvecs.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {

struct FileCloser {
    void operator()(std::FILE* f) const { std::fclose(f); }
};

template <typename T>
VecSet<T> read_vecs(const std::string& path, std::size_t max_rows) {
    static_assert(sizeof(T) == 4, "fvecs/ivecs values are 4 bytes");

    std::unique_ptr<std::FILE, FileCloser> file(std::fopen(path.c_str(), "rb"));
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }

    std::int32_t dim = 0;
    if (std::fread(&dim, sizeof(dim), 1, file.get()) != 1 || dim <= 0) {
        throw std::runtime_error("bad or empty vecs file: " + path);
    }

    // Size the output from the file length: each record is 4 + 4*dim bytes.
    std::fseek(file.get(), 0, SEEK_END);
    const long bytes = std::ftell(file.get());
    std::fseek(file.get(), 0, SEEK_SET);
    const std::size_t record_bytes = sizeof(std::int32_t) + sizeof(T) * static_cast<std::size_t>(dim);
    if (bytes < 0 || static_cast<std::size_t>(bytes) % record_bytes != 0) {
        throw std::runtime_error("file size is not a whole number of records: " + path);
    }

    VecSet<T> out;
    out.dim = static_cast<std::size_t>(dim);
    out.count = std::min(static_cast<std::size_t>(bytes) / record_bytes, max_rows);
    out.data.resize(out.count * out.dim);

    for (std::size_t i = 0; i < out.count; ++i) {
        std::int32_t d = 0;
        if (std::fread(&d, sizeof(d), 1, file.get()) != 1 || d != dim) {
            throw std::runtime_error("inconsistent dimension at record " + std::to_string(i) +
                                     " in " + path);
        }
        if (std::fread(out.data.data() + i * out.dim, sizeof(T), out.dim, file.get()) != out.dim) {
            throw std::runtime_error("truncated record " + std::to_string(i) + " in " + path);
        }
    }
    return out;
}

}  // namespace

FloatVecs read_fvecs(const std::string& path, std::size_t max_rows) {
    return read_vecs<float>(path, max_rows);
}

IntVecs read_ivecs(const std::string& path, std::size_t max_rows) {
    return read_vecs<std::int32_t>(path, max_rows);
}
