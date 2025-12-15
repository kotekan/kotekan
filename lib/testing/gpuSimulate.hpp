#ifndef GPU_SIMULATE_HPP
#define GPU_SIMULATE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h>    // for int32_t, uint32_t
#include <string>      // for string, basic_string
#include <sys/types.h> // for uint

/**
 * @brief CPU simulation of the GPU correlator.
 *
 * @par Buffers
 * @buffer input_buf  Input voltage buffer (consumer).
 * @buffer output_buf Output correlation buffer (producer).
 *
 * @conf num_elements       Int. Number of elements.
 * @conf num_local_freq     Int. Number of local frequencies.
 * @conf samples_per_data_set Int. Samples per frame.
 * @conf num_blocks         Int. Number of correlation blocks.
 * @conf block_size         Int. Block size.
 * @conf data_format        String. Data format (e.g., "4+4b", "dot4b", "cuda_wmma").
 *
 * @par Example
 * @code
 * gpu_simulate:
 *   kotekan_stage: gpuSimulate
 *   network_in_buf: volt_in
 *   corr_out_buf: corr_out
 *   num_elements: 2048
 *   num_local_freq: 16
 *   samples_per_data_set: 49152
 *   num_blocks: 32
 *   block_size: 8
 *   data_format: "4+4b"
 * @endcode
 */
class gpuSimulate : public kotekan::Stage {
public:
    gpuSimulate(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~gpuSimulate();
    void main_thread() override;

private:
    int dot4b(uint a, uint b);

    Buffer* input_buf;
    Buffer* output_buf;

    uint32_t* host_block_map;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _num_blocks;
    int32_t _block_size;
    std::string _data_format;
};

#endif
