#ifndef CUDA_QUANTIZE_KERNEL_8_CHIME_HPP
#define CUDA_QUANTIZE_KERNEL_8_CHIME_HPP

#include "DataType.hpp"

#include <cstdint>
#include <cuda_runtime.h>

void cpu_quantize8chime(const float* input, float16_t* outputf, std::uint8_t* outputi);

void gpu_quantize8chime(const float* input, float16_t* outputf, std::uint8_t* outputi,
                        cudaStream_t stream);

#endif // #ifndef CUDA_QUANTIZE_KERNEL_8_CHIME_HPP
