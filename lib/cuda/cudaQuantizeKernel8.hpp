#ifndef CUDA_QUANTIZE_KERNEL_8_HPP
#define CUDA_QUANTIZE_KERNEL_8_HPP

#include "DataType.hpp"

#include <cstdint>
#include <cuda_runtime.h>

void cpu_quantize8(const float16_t* input, float16_t* outputf, std::uint8_t* outputi,
                   int input_size1, int input_size2, int input_size3, int input_stride2,
                   int input_stride3, int outputf_size1, int outputf_size2, int outputf_size3,
                   int outputf_stride2, int outputf_stride3, int outputi_size1, int outputi_size2,
                   int outputi_size3, int outputi_stride2, int outputi_stride3);

void gpu_quantize8(const float16_t* input, float16_t* outputf, std::uint8_t* outputi,
                   int input_size1, int input_size2, int input_size3, int input_stride2,
                   int input_stride3, int outputf_size1, int outputf_size2, int outputf_size3,
                   int outputf_stride2, int outputf_stride3, int outputi_size1, int outputi_size2,
                   int outputi_size3, int outputi_stride2, int outputi_stride3,
                   cudaStream_t stream);

#endif // #ifndef CUDA_QUANTIZE_KERNEL_8_HPP
