#ifndef CUDA_QUANTIZE_KERNEL_4_HPP
#define CUDA_QUANTIZE_KERNEL_4_HPP

#include <DataType.hpp>

void gpu_quantize4(const float16_t* input, float16_t* outputf, kotekan::uint4x2_t* outputi,
                   const int input_size1, int input_size2, int input_size3, int input_stride2,
                   int input_stride3, int outputf_size1, int outputf_size2, int outputf_size3,
                   int outputf_stride2, int outputf_stride3, int outputi_size1, int outputi_size2,
                   int outputi_size3, int outputi_stride2, int outputi_stride3,
                   cudaStream_t stream);

#endif // #ifndef CUDA_QUANTIZE_KERNEL_4_HPP
