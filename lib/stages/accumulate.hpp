#ifndef ACCUMULATE_HPP
#define ACCUMULATE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string


/**
 * @brief Accumulates a fixed number of GPU frames into a summed output frame.
 *
 * @par Buffers
 * @buffer in_buf  Input GPU buffer (consumer), int32 CHORD frames with chordMetadata.
 * @buffer out_buf Output buffer (producer), same format/metadata.
 *
 * @conf in_buf               String. Input buffer.
 * @conf out_buf              String. Output buffer.
 * @conf samples_per_data_set Int. Samples per GPU frame.
 * @conf num_gpu_frames       Int. Number of frames to accumulate per output.
 *
 * @par Example
 * @code
 * accumulate:
 *   in_buf: gpu_sum_in
 *   out_buf: gpu_sum_out
 *   samples_per_data_set: 49152
 *   num_gpu_frames: 4
 * @endcode
 */
class accumulate : public kotekan::Stage {
public:
    accumulate(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~accumulate();
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    int32_t _samples_per_data_set;
    int32_t _num_gpu_frames;
};

#endif
