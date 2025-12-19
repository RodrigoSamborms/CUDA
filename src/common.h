#pragma once
#include <cuda_runtime.h>
#include <cstdio>

inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}
