// 8-bit FRB beam quantizer for CHIME, using the float16 FRB beamformer and CHIME's
// sending-to-Bonsai network format

#ifndef CUDA_QUANTIZE_KERNEL_8_HPP
#define CUDA_QUANTIZE_KERNEL_8_HPP

#include "DataType.hpp"

#include <cstdint>
#include <cuda_runtime.h>

void cpu_quantize8(const float16_t* input, float16_t* outputf, std::uint8_t* outputi);

void gpu_quantize8(const float16_t* input, float16_t* outputf, std::uint8_t* outputi,
                   cudaStream_t stream);

#endif // #ifndef CUDA_QUANTIZE_KERNEL_8_HPP
