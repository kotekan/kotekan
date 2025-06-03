/**
 * @file
 * @brief CPU version of CUDA FRB beamforming kernel
 */

#ifndef SIMULATE_CUDA_FRB_BEAMFORMER_HPP
#define SIMULATE_CUDA_FRB_BEAMFORMER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string
#include <vector>

/**
 * @class gpuSimulateCudaFRBBeamformer
 * @brief Stage for faking CUDA FRB beamforming.
 *
 * @par Buffers
 * @buffer voltage_in_buf   Voltages
 *         @buffer_format Int4+4
 *         @buffer_metadata chimeMetadata or oneHotMetadata
 * @buffer phase_in_buf   Beamformer phases
 *         @buffer_format Int8
 *         @buffer_metadata chimeMetadata or oneHotMetadata
 * @buffer shift_in_buf   Number of bits to shift result
 *         @buffer_format Int32
 * @buffer beams_out_buf  Output buffer for formed beams
 *         @buffer_format Int4+4
 *         @buffer_metadata Copied from voltage_in_buf
 *
 * @conf  num_elements          Int.  Number of dishes x polarizations.
 * @conf  num_local_freq        Int.  Number of frequencies in each frame.
 * @conf  samples_per_data_set  Int.  Number of time samples per frame.
 * @conf  num_beams             Int.  Number of beams being formed.
 * @conf  zero_output           Bool. Zero out the array before filling it?  Useful when doing
 * one-hot sparse outputs.
 */
class gpuSimulateCudaFRBBeamformer : public kotekan::Stage {
public:
    gpuSimulateCudaFRBBeamformer(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& buffer_container);
    ~gpuSimulateCudaFRBBeamformer();
    void main_thread() override;

private:
    using int4x2_t = uint8_t;

    Buffer* voltage_buf;
    // Buffer* dashlayout_buf;
    Buffer* phase_buf;
    Buffer* beamgrid_buf;

    /// Number of dishes in the telescope
    int32_t _num_dishes;
    /// Dish grid side length
    int32_t _dish_grid_size;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    /// Time downsampling factor
    int32_t _time_downsampling;
    /// Dish layout array
    std::vector<int> _dishlayout;
};

#endif // SIMULATE_CUDA_FRB_BEAMFORMER_HPP
