// 8-bit FRB beam quantizer for CHIME, using the chromatic CHIME float32 FRB beamformer and CHIME's
// sending-to-Bonsai network format

#ifndef CUDA_QUANTIZE_KERNEL_8_CHIME_HPP
#define CUDA_QUANTIZE_KERNEL_8_CHIME_HPP

#include <cstdint>
#include <cuda_runtime.h>

void cpu_quantize8chime(const float* input, float* outputf, std::uint8_t* outputi);

void gpu_quantize8chime(const float* input, float* outputf, std::uint8_t* outputi,
                        cudaStream_t stream);

#endif // #ifndef CUDA_QUANTIZE_KERNEL_8_CHIME_HPP
