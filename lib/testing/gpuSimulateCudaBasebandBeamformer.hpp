/**
 * @file
 * @brief CPU version of CUDA baseband beamforming kernel
 */

#ifndef SIMULATE_CUDA_BASEBAND_BEAMFORMER_HPP
#define SIMULATE_CUDA_BASEBAND_BEAMFORMER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @class gpuSimulateCudaBasebandBeamformer
 * @brief Stage for faking CUDA baseband beamforming.
 */
class gpuSimulateCudaBasebandBeamformer : public kotekan::Stage {
public:
    gpuSimulateCudaBasebandBeamformer(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container);
    ~gpuSimulateCudaBasebandBeamformer();
    void main_thread() override;

private:
    using int4x2_t = uint8_t;
    void bb_simple(std::string id_tag,
                   const int8_t *__restrict__ const A,
                   const int4x2_t *__restrict__ const E,
                   const int32_t *__restrict__ const s,
                   int4x2_t *__restrict__ const J,
                   const int T,   // 32768; // number of times
                   const int B,   // = 96;    // number of beams
                   const int D,   // = 512;   // number of dishes
                   const int F    // = 16;    // frequency channels per GPU
                   );
    void bb_simple_sub(std::string id_tag,
                       const int8_t *__restrict__ const A,
                       const int4x2_t *__restrict__ const E,
                       const int32_t *__restrict__ const s,
                       int4x2_t *__restrict__ const J,
                       const int T,   // 32768; // number of times
                       const int B,   // = 96;    // number of beams
                       const int D,   // = 512;   // number of dishes
                       const int F,    // = 16;    // frequency channels per GPU
                       const int t,
                       const int b,
                       const int d,
                       const int f,
                       const int p
                   );

    struct Buffer* voltage_buf;
    struct Buffer* phase_buf;
    struct Buffer* shift_buf;
    struct Buffer* output_buf;

    /// Number of elements on the telescope
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    // Number of beams to form.
    int32_t _num_beams;
};

#endif // SIMULATE_CUDA_BASEBAND_BEAMFORMER_HPP
