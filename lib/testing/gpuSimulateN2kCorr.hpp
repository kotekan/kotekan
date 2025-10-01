#ifndef GPU_SIMULATE_N2K_CORR_HPP
#define GPU_SIMULATE_N2K_CORR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint> // for int32_t
#include <string>  // for string

/**
 * @brief Perform on CPU the equivalent of the CudaCorrelator stage:
 * N2 correlation.
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
 * @buffer rfimask_in_buf The input fast cadence RFI mask.
 *      @buffer_format bitmask in (eqivalently) uint8_t or uint1x8_t.
 *      @buffer_shape [samples_per_data_set / 1024, num_local_freq, 128].
 *          The time axis is split into a 1024 sample (128 byte) fast index
 *          and a coarse samples_per_data_set / 1024 index.  The frame size is
 *          samples_per_data_set * num_local_freq / 8 bytes.
 *      @buffer_metadata chordMetadata
 *
 * @buffer corr_out_buf  The output correlation matrix, tiled into 16x16
 *      element blocks, storing the lower triangular blocks (e.g. B00, B10,
 *      B11, B20, ...).  There are num_blocks = num_elements / 16 blocks, and
 *      num_integrations = samples_per_data_set / sub_integration_ntime times
 *      represented in the output.
 *      @buffer_format int32+32 complex.  Each 2 int entry has the real part
 *          in the first int and the imaginary part in the second int.
 *      @buffer_shape [num_integrations, num_local_freq, num_blocks, 16, 16,
 *           2].  The final axis is over real/imaginary components. The size
 *           is num_integrations * num_local_freq * num_blocks * 16^2 * 2
 *           int32s.
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x
 *      polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 * @conf sub_integration_ntime Int.  Number of samples to sum over for each N^2
 *      correlation matrix.
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
