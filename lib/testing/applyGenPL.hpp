#ifndef APPLY_GEN_PL_HPP
#define APPLY_GEN_PL_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint> // for int32_t
#include <string>  // for string

/**
 * @brief Apply a generated packet loss mask to a generated voltage stream
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_n2k.yaml`.
 *
 * @par Buffers
 *
 * @buffer voltage_in_buf The input voltages.
 *      @buffer_format 4+4-bit complex int, encoded with CHIME convention:
 *          offset encoded by +8, imaginary in low bits, real in high. In
 *          kotekan this is a int4x2_swapped_withoffset.
 *      @buffer_shape [samples_per_data_set, num_local_freq, num_elements] or
 *          equivalently [samples_per_data_set, num_local_freq,
 *          num_polarizations, num_dishes]
 *      @buffer_metadata chordMetadata
 *
 * @buffer plmask_in_buf The input packet loss (PL) mask.
 *      @buffer_format bitmask in uint64_t
 *      @buffer_shape [samples_per_data_set / 128, num_local_freq / 4,
 *          num_elements / 8]. Downsampled by 2 in time, 4 in freq, 8 in
 *          element.
 *          The time axis is split into a 128 sample (in 64 bits, downsampled
 *          by 2 in time) fast index and a slow index.
 *      @buffer_metadata chordMetadata time_downsample_fpga[] = 128
 *
 * @buffer voltage_out_buf The output voltages, zeroed (8'd) by the PL mask.
 *      @buffer_format 4+4-bit complex int, encoded with CHIME convention:
 *          offset encoded by +8, imaginary in low bits, real in high. In
 *          kotekan this is a int4x2_swapped_withoffset.
 *      @buffer_shape [samples_per_data_set, num_local_freq, num_elements] or
 *          equivalently [samples_per_data_set, num_local_freq,
 *          num_polarizations, num_dishes]
 *      @buffer_metadata chordMetadata
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x
 *      polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 */

class applyGenPL : public kotekan::Stage {
public:
    applyGenPL(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~applyGenPL();
    void main_thread() override;

private:
    Buffer* input_buf;
    Buffer* plmask_buf;
    Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif
