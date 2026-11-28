.PHONY: v1 v2 v3 v4 v5 v6 clean test-v3 data micro cublas

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
CXX ?= g++
NVCC ?= nvcc
# Default covers Ampere (A100 sm_80, RTX 30xx sm_86) and Ada (RTX 4000 Ada sm_89),
# plus sm_89 PTX for JIT on newer GPUs. Volta (sm_70) is not supported by CUDA 13.
# Override for a specific GPU, e.g. `make v3 CUDA_ARCH=-arch=sm_89`.
CUDA_ARCH ?= -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=[sm_89,compute_89]
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
CPPFLAGS := -Iinclude

BUILD := build

# ---------------------------------------------------------------------------
# Shared sources
# ---------------------------------------------------------------------------
COMMON_SRC := src/common/fvecs.cpp src/common/recall.cpp src/common/timer.cpp
BENCH_SRC := bench/bench.cpp $(COMMON_SRC)
BENCH_DEPS := $(BENCH_SRC) include/search.h include/fvecs.h include/recall.h include/timer.h
TOPK_SRC := src/cuda/topk.cu
CUDA_HEADERS := include/topk.cuh include/cuda_check.cuh

# `make vN` builds the benchmark binary build/bench_vN (plus tests, if any).
# Each binary links bench.cpp with exactly one version's search().
v1: $(BUILD)/bench_v1
v2: $(BUILD)/bench_v2
v3: $(BUILD)/bench_v3 $(BUILD)/test_v3
v4: $(BUILD)/bench_v4
v5: $(BUILD)/bench_v5
v6: $(BUILD)/bench_v6

$(BUILD):
	mkdir -p $(BUILD)

# --- CPU versions ----------------------------------------------------------
$(BUILD)/bench_v1: $(BENCH_DEPS) src/cpu/v1_serial.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DBENCH_VERSION='"v1"' -o $@ $(BENCH_SRC) src/cpu/v1_serial.cpp

$(BUILD)/bench_v2: $(BENCH_DEPS) src/cpu/v2_omp.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp $(CPPFLAGS) -DBENCH_VERSION='"v2"' -o $@ $(BENCH_SRC) src/cpu/v2_omp.cpp

# --- CUDA versions ---------------------------------------------------------
$(BUILD)/bench_v3: $(BENCH_DEPS) src/cuda/v3_naive.cu $(TOPK_SRC) $(CUDA_HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -DBENCH_VERSION='"v3"' -o $@ \
		$(BENCH_SRC) src/cuda/v3_naive.cu $(TOPK_SRC)

$(BUILD)/bench_v4: $(BENCH_DEPS) src/cuda/v4_tiled.cu $(TOPK_SRC) $(CUDA_HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -DBENCH_VERSION='"v4"' -o $@ \
		$(BENCH_SRC) src/cuda/v4_tiled.cu $(TOPK_SRC)

$(BUILD)/bench_v5: $(BENCH_DEPS) src/cuda/v5_fused.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -DBENCH_VERSION='"v5"' -o $@ \
		$(BENCH_SRC) src/cuda/v5_fused.cu

$(BUILD)/bench_v6: $(BENCH_DEPS) src/cuda/v6_wmma.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -DBENCH_VERSION='"v6"' -o $@ \
		$(BENCH_SRC) src/cuda/v6_wmma.cu

# --- Tests -----------------------------------------------------------------
$(BUILD)/test_v3: tests/test_search.cpp src/cuda/v3_naive.cu $(TOPK_SRC) $(CUDA_HEADERS) include/search.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -o $@ \
		tests/test_search.cpp src/cuda/v3_naive.cu $(TOPK_SRC)

test-v3: $(BUILD)/test_v3
	./$(BUILD)/test_v3

# --- Reference measurements (not versions) ---------------------------------
# Roofline ceilings you can actually reach, and the cuBLAS GEMM baseline.
micro: $(BUILD)/bandwidth $(BUILD)/fma
cublas: $(BUILD)/cublas_gemm

$(BUILD)/bandwidth: micro/bandwidth.cu include/csv_append.h include/cuda_check.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -o $@ micro/bandwidth.cu

$(BUILD)/fma: micro/fma.cu include/csv_append.h include/cuda_check.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -o $@ micro/fma.cu

$(BUILD)/cublas_gemm: baselines/cublas_gemm.cu include/csv_append.h include/cuda_check.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -o $@ baselines/cublas_gemm.cu -lcublas

# --- Data ------------------------------------------------------------------
data:
	bash scripts/get_sift1m.sh

clean:
	rm -rf $(BUILD)
