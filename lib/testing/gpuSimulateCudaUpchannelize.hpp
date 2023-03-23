/**
 * @file
 * @brief CPU version of CUDA upchannelization kernel
 */

#ifndef SIMULATE_CUDA_UPCHANNELIZE_HPP
#define SIMULATE_CUDA_UPCHANNELIZE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "visUtil.hpp"

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @class gpuSimulateCudaUpchannelize
 * @brief Stage for faking CUDA upchannelization.
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
    void upchan_simple(std::string tag,
                       const void* __restrict__ const E, void* __restrict__ const Ebar);
    /*
      const float16_t* __restrict__ const W,
      const float16_t* __restrict__ const G,
      // const storage_t *__restrict__ const E,
      // storage_t *__restrict__ const Ebar,
      const int T, // 32768; // number of times
      const int D, // = 512;   // number of dishes
      const int F, // = 16;    // input frequency channels per GPU
      const int U  // = 16;    // upchannelization factor
    */
    void upchan_simple_sub(std::string tag,
                           const void* __restrict__ const E, void* __restrict__ const Ebar,
                           int t, int p, int f, int d);

    std::vector<float16_t> gains16;
#endif

    struct Buffer* voltage_in_buf;
    struct Buffer* voltage_out_buf;

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
