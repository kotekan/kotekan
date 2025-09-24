#ifndef GPU_SIMULATE_N2K_CORR_HPP
#define GPU_SIMULATE_N2K_CORR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @brief Perform on CPU the equivalent of the CudaCorrelator stage:
 * N2 correlation.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_n2k.yaml`.
 *
 * @par Buffers
 * @buffer voltage_in_buf The input voltages.  Size per frame: samples_per_data_set * num_element *
 * num_local_freq
 * @buffer_format 4+4-bit complex offset encoded (Im lo, Re hi)
 * @buffer rfimask_in_buf The input RFI mask.  Size per frame: num_local_freq * samples_per_data_set / 8
 * @buffer_format bitmask in uint8_t. Shape is [samples_per_data_set / 1024, num_local_freq, 128]
 * @buffer corr_out_buf  The output correlation matrix.  Size per frame: num_local_freq *
 * (samples_per_data_set / sub_integration_ntime) * num_elements^2 * 2 * sizeof_int32
 * @buffer_format int32 complex
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 * @conf sub_integration_ntime Int.  Number of samples to sum over for each N^2 correlation matrix.
 *
 * The output matrix's upper triangle is filled (the lower triangle is
 * zeroed out).
 */
class gpuSimulateN2kCorr : public kotekan::Stage {
public:
    gpuSimulateN2kCorr(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~gpuSimulateN2kCorr();
    void main_thread() override;

private:
    Buffer* input_buf;
    Buffer* rfimask_buf;
    Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
};

#endif
