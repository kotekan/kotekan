#include "cudaMemsetInt.hpp"

__global__ void cuda_memset_int(int* data, int value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = value;
}

void cudaMemsetInt(int* data, int value, size_t n) {
    std::size_t threads = 256;
    std::size_t blocks = (n + threads - 1) / threads;
    cuda_memset_int<<<blocks, threads>>>(data, value, n);
}
