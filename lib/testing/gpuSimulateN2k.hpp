#ifndef GPU_SIMULATE_N2K_HPP
#define GPU_SIMULATE_N2K_HPP

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
 * Reads config values:
 * - num_elements -- number of feeds or (antennas x polarizations)
 * - num_local_freq
 * - samples_per_data_set
 * - sub_integration_ntime -- number of samples to sum over for each
 *     N^2 correlation matrix.
 *
 * Input:
 * - network_in_buf: samples_per_data_set * num_element * num_local_freq
 *     (per frame)
 *     4+4-bit, complex voltages
 *
 * Output:
 * - corr_out_buf: num_local_freq * (samples_per_data_set / sub_integration_ntime) * num_elements * num_elements * 2 * sizeof_int32
 *     (per frame)
 *     int32, complex correlation values
 *
 * The output matrix's upper triangle is filled (the lower triangle is
 * zeroed out).
 */
class gpuSimulateN2k : public kotekan::Stage {
public:
    gpuSimulateN2k(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~gpuSimulateN2k();
    void main_thread() override;

private:
    struct Buffer* input_buf;
    struct Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
};

#endif
