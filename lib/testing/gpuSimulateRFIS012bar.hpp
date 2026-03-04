#ifndef GPU_SIMULATE_RFI_S012_BAR_HPP
#define GPU_SIMULATE_RFI_S012_BAR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint> // for int32_t
#include <string>  // for string

/**
 * @brief Perform on CPU the equivalent of the cudaRFIS012bar stage.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_rfi_s012bar.yaml`.
 *
 * This stage downsamples the RFI S012 buffer in time by a factor "rfi_second_downsampling_factor"
 * via accumulation.
 *
 * @par Buffers
 * @buffer in_buf The input RFI S012 buffer
 *      @buffer_format uint64_t
 *      @buffer_shape [samples_per_data_set / rfi_downsampling_factor, num_local_freq,
 *          3, num_polarizations, num_dishes] or equivalently
 *          [samples_per_data_set / rfi_downsampling_factor, num_local_freq * 3 * num_elements]
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = rfi_downsampling_factor
 *
 * @buffer out_buf The output RFI S012 buffer
 *      @buffer_format uint64_t
 *      @buffer_shape [samples_per_data_set / rfi_downsampling_factor /
 * rfi_second_downsampling_factor, num_local_freq, 3, num_polarizations, num_dishes] or equivalently
 *          [samples_per_data_set / rfi_downsampling_factor / rfi_second_downsampling_factor,
 *          num_local_freq * 3 * num_elements]
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = rfi_downsampling_factor *
 * rfi_second_downsampling_factor
 *
 * @conf num_polarizations     Int.  Number of polarizations
 * @conf num_dishes            Int.  Number of feeds per polarization.
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 * @conf rfi_downsampling_factor  Int.  Downsampling factor of input buffer
 * @conf rfi_second_downsampling_factor  Int.   Additional downsampling to perform
 */
class gpuSimulateRFIS012bar : public kotekan::Stage {
public:
    gpuSimulateRFIS012bar(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~gpuSimulateRFIS012bar();
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    // Config options
    const int64_t _num_polarizations;
    const int64_t _num_dishes;
    const int64_t _num_elements;
    const int64_t _num_local_freq;
    const int64_t _samples_per_data_set;
    const int64_t _rfi_downsampling_factor;
    const int64_t _rfi_second_downsampling_factor;
};

#endif
