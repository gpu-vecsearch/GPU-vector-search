.PHONY: v1 v2 v3 v4 v5 v6 bench clean test-v3

NVCC ?= nvcc
# Override for a specific GPU, e.g. `make v3 CUDA_ARCH=-arch=sm_80`.
CUDA_ARCH ?= -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=[sm_86,compute_86]
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
CPPFLAGS := -Iinclude

BUILD := build
CUDA_COMMON := src/cuda/topk.cu
CUDA_HEADERS := include/search.h include/topk.cuh include/cuda_check.cuh

$(BUILD):
	mkdir -p $(BUILD)

# --- v3: naive CUDA --------------------------------------------------------
v3: $(BUILD)/test_v3

$(BUILD)/test_v3: tests/test_search.cpp src/cuda/v3_naive.cu $(CUDA_COMMON) $(CUDA_HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -o $@ \
		tests/test_search.cpp src/cuda/v3_naive.cu $(CUDA_COMMON)

test-v3: $(BUILD)/test_v3
	./$(BUILD)/test_v3

# --- not implemented yet ---------------------------------------------------
v1 v2 v4 v5 v6 bench:
	@echo "$@ is not implemented yet; add its build or benchmark recipe to the Makefile."
	@exit 1

clean:
	rm -rf $(BUILD)
