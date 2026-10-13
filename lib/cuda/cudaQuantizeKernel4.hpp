// 4-bit FRB beam quantizer for CHORD

#ifndef CUDA_QUANTIZE_KERNEL_4_HPP
#define CUDA_QUANTIZE_KERNEL_4_HPP

#include "DataType.hpp"

#include <cstdint>
#include <cuda_runtime.h>

// `pl_mask[time * pl_mask_stride + freq]` is the packet loss mask of the input (1 = valid). Masked
// samples are quantized as usual, but are skipped when choosing each chunk's offset and scale.
void cpu_quantize4(const __half* input, const std::uint8_t* pl_mask, int pl_mask_stride,
                   __half* outputf, kotekan::int4x2_t* outputi, int input_size1, int input_size2,
                   int input_size3, int input_stride2, int input_stride3, int outputf_size1,
                   int outputf_size2, int outputf_size3, int outputf_stride2, int outputf_stride3,
                   int outputi_size1, int outputi_size2, int outputi_size3, int outputi_stride2,
                   int outputi_stride3, float stddev_cutoff);

void gpu_quantize4(const float16_t* input, const std::uint8_t* pl_mask, int pl_mask_stride,
                   float16_t* outputf, kotekan::int4x2_t* outputi, int input_size1, int input_size2,
                   int input_size3, int input_stride2, int input_stride3, int outputf_size1,
                   int outputf_size2, int outputf_size3, int outputf_stride2, int outputf_stride3,
                   int outputi_size1, int outputi_size2, int outputi_size3, int outputi_stride2,
                   int outputi_stride3, float stddev_cutoff, cudaStream_t stream);

#endif // #ifndef CUDA_QUANTIZE_KERNEL_4_HPP
