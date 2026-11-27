/**
 * @file
 * @brief CPU version of CUDA upchannelization kernel
 */

#ifndef SIMULATE_CUDA_UPCHANNELIZE_HPP
#define SIMULATE_CUDA_UPCHANNELIZE_HPP

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for KOTEKAN_FLOAT16, float16_t
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class gpuSimulateCudaUpchannelize
 * @brief Stage for faking CUDA upchannelization.
 *
 * These tests use float32 on purpose, although the respective GPU kernels use float16.
 * This catches an important possible issue: there could be catastrophic cancellation
 * when using float16, and this would then happen on both CPU and GPU, so we would never know.
 * That is very unlikely in float32. We thus know that the float32 results don’t suffer from
 * catastrophic floating-point round-off problems. Since the bots agree with me that comparing
 * float16 to float32 is fine, we’re doing that, allowing the results to differ by at most 1 ulp.
 *
 * @par Buffers
 * @buffer voltage_in_buf   Input voltages
 *         @buffer_format Int4+4
 *         @buffer_metadata chimeMetadata or oneHotMetadata
 * @buffer voltage_out_buf  Output buffer for formed beams
 *         @buffer_format Int4+4
 *         @buffer_metadata Copied from voltage_in_buf
 *
 * @conf  num_dishes            Int.  Number of dishes.
 * @conf  num_local_freq        Int.  Number of frequencies in each frame.
 * @conf  samples_per_data_set  Int.  Number of time samples per frame.
 * @conf  upchan_factor         Int.  Upchannelization factor.
 * @conf  zero_output           Bool. Zero out the array before filling it?  Useful when doing
 * one-hot sparse outputs.
 */
class gpuSimulateCudaUpchannelize : public kotekan::Stage {
public:
    gpuSimulateCudaUpchannelize(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& buffer_container);
    ~gpuSimulateCudaUpchannelize();
    void main_thread() override;

private:
    using int4x2_t = uint8_t;
#if KOTEKAN_FLOAT16
    void upchan_simple(std::string tag, const void* __restrict__ const E,
                       void* __restrict__ const Ebar);
    void upchan_simple_sub(std::string tag, const void* __restrict__ const E,
                           void* __restrict__ const Ebar, int t, int p, int f, int d);

    std::vector<float16_t> gains16;
#endif

    Buffer* voltage_in_buf;
    Buffer* voltage_out_buf;

    /// Number of dishes in the telescope
    int32_t _num_dishes;
    /// Number of input frequencies
    int32_t _num_local_freq;
    /// Number of input time samples.
    int32_t _samples_per_data_set;
    /// Upchannelization factor
    int32_t _upchan_factor;
};

#endif // SIMULATE_CUDA_UPCHANNELIZE_HPP
