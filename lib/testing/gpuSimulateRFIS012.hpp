#ifndef GPU_SIMULATE_RFI_S012_HPP
#define GPU_SIMULATE_RFI_S012_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint> // for int32_t
#include <string>  // for string

/**
 * @brief Perform on CPU the equivalent of the cudaRFIS012 stage.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_rfi_s012.yaml`.
 *
 * This stage computes sums over 1, |E|^2, and |E^4|, weighted by the PL mask,
 * for rfi_downsampling_factor time samples, and places the results in the
 * RFI S012 buffer.
 *
 * The raw packet loss (PL) mask is downsampled by 2 in time, 4 in frequency,
 * and 8 in element (polarization x dish).
 *
 * In memory the PL mask has its time axis split into fast (length 64 bits)
 * and coarse (length samples_per_data_set / 2 / 64 bits) axes.  Each bit in
 * the mask corresponds to 2 times (and 4 frequencies), so the length 64 bit
 * fast time axis represents 128 time samples.
 *
 * @par Buffers
 * @buffer in_plmask_buf The input packet loss mask (not expanded).
 *      @buffer_format bitmask: uint64_t, equivalently uint8_t or uint1x8_t
 *      @buffer_shape [samples_per_data_set / 128, num_local_freq / 4,
 *          num_elements / 8] or equivalently [samples_per_data_set / 128,
 *          num_local_freq / 4, num_elements / 8, 8] if the datatype is uint8_t
 *          or uint1x8_t. If the elements axis is constructed as polarization,
 *          dish pairs, its shape is taken to be [num_polarizations,
 *          num_dishes / 8]. Size of a frame is samples_per_data_set
 *          * num_local_freq * num_elements / 512 bytes.
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = 128
 *
 * @buffer in_voltage_buf The input voltages.
 *      @buffer_format 4+4-bit complex int, encoded with CHIME convention:
 *          offset encoded by +8, imaginary in low bits, real in high. In
 *          kotekan this is a int4x2_swapped_withoffset.
 *      @buffer_shape [samples_per_data_set, num_local_freq, num_elements] or
 *          equivalently [samples_per_data_set, num_local_freq,
 *          num_polarizations, num_dishes]
 *      @buffer_metadata chordMetadata
 *
 * @buffer out_rfis012_buf The output RFI S012 buffer
 *      @buffer_format uint64_t
 *      @buffer_shape [samples_per_data_set / rfi_downsampling_factor, num_local_freq,
 *          3, num_polarizations, num_dishes] or equivalently
 *          [samples_per_data_set / rfi_downsampling_factor, num_local_freq * 3 * num_elements]
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = rfi_downsampling_factor
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x
 *      polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 * @conf rfi_downsampling_factor  Int.  Factor to downsample the voltage by
 */
class gpuSimulateRFIS012 : public kotekan::Stage {
public:
    gpuSimulateRFIS012(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& buffer_container);
    ~gpuSimulateRFIS012();
    void main_thread() override;

private:
    Buffer* in_plmask_buf;
    Buffer* in_voltage_buf;
    Buffer* out_rfis012_buf;

    // Config options
    const int64_t _num_polarizations;
    const int64_t _num_dishes;
    const int64_t _num_elements;
    const int64_t _num_local_freq;
    const int64_t _samples_per_data_set;
    const int64_t _rfi_downsampling_factor;
};

#endif
