#ifndef GPU_SIMULATE_N2K_PLEXP_HPP
#define GPU_SIMULATE_N2K_PLEXP_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @brief Perform on CPU the equivalent of the CudaCorrelator stage:
 * N2 PL 1 bit Correlator.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_n2k.yaml`.
 *
 * @par Buffers
 * @buffer in_buf The input packet loss mask.  Size per frame: samples_per_data_set / 2 * num_local_freq / 4 * num_element / 8 / 8
 * @buffer_format uint64 bitmask. shape: [samples_per_data_set / 128, num_local_freq / 4, num_elements / 8]
 * @buffer out_buf  The output packet loss mask.  Size per frame: samples_per_dataset * num_local_freq * num_elements/8 / 8
 * @buffer_format uint64 bitmask. shape: [samples_per_data_set / 64, num_local_freq, num_elements / 8]
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 */
class gpuSimulateN2kPL1bitCorr : public kotekan::Stage {
public:
    gpuSimulateN2kPL1bitCorr(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~gpuSimulateN2kPL1bitCorr();
    void main_thread() override;

private:
    Buffer* input_plmask_buf;
    Buffer* input_rfimask_buf;
    Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
};

#endif
