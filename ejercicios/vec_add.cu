#include <cstdio>
#include <cuda_runtime.h>
#include "common.h"

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20; // 1M elementos
    const size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cuda_check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    cuda_check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    cuda_check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    cuda_check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "Memcpy a");
    cuda_check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "Memcpy b");

    int block = 256;
    int grid = (n + block - 1) / block;
    vec_add<<<grid, block>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "Memcpy c");

    bool ok = true;
    for (int i = 0; i < n; i += n / 10) {
        if (h_c[i] != 3.0f) { ok = false; break; }
    }
    std::printf("vec_add %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return ok ? 0 : 1;
}
