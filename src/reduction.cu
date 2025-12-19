#include <cstdio>
#include <cuda_runtime.h>
#include "common.h"

__global__ void reduce_sum(const float* __restrict__ x, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? x[i] : 0.0f;
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h_x[i] = 1.0f;

    float *d_x, *d_out;
    cuda_check(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
    cuda_check(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc d_out");
    float zero = 0.0f;
    cuda_check(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice), "Memcpy x");
    cuda_check(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice), "Init out");

    int block = 256;
    int grid = (n + block - 1) / block;
    size_t smem = block * sizeof(float);
    reduce_sum<<<grid, block, smem>>>(d_x, d_out, n);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    float h_out = -1.0f;
    cuda_check(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost), "Memcpy out");

    bool ok = (h_out == static_cast<float>(n));
    std::printf("reduction sum=%f (%s)\n", h_out, ok ? "OK" : "FAIL");

    cudaFree(d_x); cudaFree(d_out);
    free(h_x);
    return ok ? 0 : 1;
}
