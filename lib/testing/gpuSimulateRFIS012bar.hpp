#ifndef GPU_SIMULATE_RFI_S012_BAR_HPP
#define GPU_SIMULATE_RFI_S012_BAR_HPP

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
 * The raw packet loss (PL) mask is downsampled by 2 in time, 4 in frequency,
 * and 8 in element (polarization x dish). To be combined with the RFI mask in
 * the 1-bit correlator, the PL mask first has to be expanded (entries cloned)
 * so that it is no longer downsampled in time or frequency.  The downsampling
 * in element will remain.
 *
 * In memory the PL mask has its time axis split into fast (length 64 bits)
 * and coarse (length samples_per_data_set / 2 / 64 bits) axes.  Each bit in
 * the mask corresponds to 2 times (and 4 frequencies), so the length 64 bit
 * fast time axis represents 128 time samples.
 *
 * This stage performs the expansion in time and frequency by cloning bits
 * by 2 in time and 4 in frequency.
 *
 * @par Buffers
 * @buffer in_buf The input packet loss mask.
 *      @buffer_format bitmask: uint64_t, equivalently uint8_t or uint1x8_t
 *      @buffer_shape [samples_per_data_set / 128, num_local_freq / 4,
 *          num_elements / 8] or equivalently [samples_per_data_set / 128,
 *          num_local_freq / 4, num_elements / 8, 8] if the datatype is uint8_t
 *          or uint1x8_t. If the elements axis is constructed as polarization,
 *          dish pairs, its shape is taken to be [num_polarizations,
 *          num_dishes / 8]. Size of a frame is samples_per_data_set
 *          * num_local_freq * num_elements / 512 bytes.
 *      @buffer_metadata chordMetadata. time_downsample_fpga[] = 128
 *
 * @buffer out_buf The output, RFI_S012 buffer.
 *      @buffer_format bitmask: uint64_t, equivalently uint8_t or uint1x8_t
 *      @buffer_shape [samples_per_data_set / 64, num_local_freq,
 *          num_elements / 8] or equivalently [samples_per_data_set / 64,
 *          num_local_freq, num_elements / 8, 8] if the datatype is uint8_t
 *          or uint1x8_t. If the elements axis is constructed as polarization,
 *          dish pairs, its shape is taken to be [num_polarizations,
 *          num_dishes / 8]. Size of a frame is samples_per_data_set
 *          * num_local_freq * num_elements / 64 bytes.
 *      @buffer_metadata chordMetadata. time_downsample_fpga[] = 64
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x
 *      polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 */
class gpuSimulateRFIS012bar : public kotekan::Stage {
public:
    gpuSimulateRFIS012bar(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~gpuSimulateRFIS012bar();
    void main_thread() override;

private:
    Buffer* in_plmask_buf;
    Buffer* in_voltage_buf;
    Buffer* out_rfis012_buf;

    // Config options
    int64_t _num_local_freq;
    int64_t _num_elements;
    int64_t _samples_per_data_set;
    int64_t _rfi_downsampling_factor;
};

#endif
