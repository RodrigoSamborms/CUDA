#include <cstdio>
#include <cuda_runtime.h>
#include "common.h"

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    float *d_x, *d_y;
    cuda_check(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
    cuda_check(cudaMalloc(&d_y, bytes), "cudaMalloc d_y");

    cuda_check(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice), "Memcpy x");
    cuda_check(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice), "Memcpy y");

    float a = 3.0f;
    int block = 256;
    int grid = (n + block - 1) / block;
    saxpy<<<grid, block>>>(n, a, d_x, d_y);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost), "Memcpy back");

    bool ok = true;
    for (int i = 0; i < n; i += n / 10) {
        float expected = a * 1.0f + 2.0f; // 5.0f
        if (h_y[i] != expected) { ok = false; break; }
    }
    std::printf("saxpy %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return ok ? 0 : 1;
}
