#ifndef ZERO_LOWER_TRIANGLE_HPP
#define ZERO_LOWER_TRIANGLE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

/**
 * @brief Zeroes out the lower-triangle of a correlation matrix
 * stream.  Created to test the CudaCorrelator stage, which writes
 * some junk in the lower triangle.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_n2k.yaml`.
 *
 * @par Buffers
 * @buffer  corr_in_buf  The input correlation matrix whose lower-triangle is to be zeroed out.  Size per frame: num_local_freq * (samples_per_data_set / sub_integration_ntime) *
 * num_elements^2 * 2 * sizeof(int32)
 * @buffer_format int32 complex
 * @buffer  curr_out_buf  The output correlation matrix with zeroed-out lower triangle.  Size per frame: num_local_freq * (samples_per_data_set / sub_integration_ntime) *
 * num_elements^2 * 2 * sizeof(int32)
 * @buffer_format int32 complex
 *
 * @conf num_elements          Int.  Number of radio input feeds.
 * @conf num_local_freq        Int.  Number of frequencies in the correlation matrix.
 * @conf samples_per_data_set  Int.  Time samples per frame.
 * @conf sub_integration_ntime Int.  Number of time samples that are summed into each correlation matrix.
 *
 * There are (samples_per_data_set / nt_inner) matrices per kotekan
 * block, each (num_local_freq * num_elements * num_elements) in size.
 * This stage copies only the upper-triangle part of the N*N matrix
 * from the input buffer to the output, zeroing out the
 * lower-triangle.
 *
 */

class zeroLowerTriangle : public kotekan::Stage {
public:
    zeroLowerTriangle(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~zeroLowerTriangle();
    void main_thread() override;

private:
    struct Buffer* input_buf;
    struct Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
    std::string _data_format;
};

#endif
