#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Abort with file/line context if a CUDA runtime call fails.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Check for launch errors right after a kernel launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
