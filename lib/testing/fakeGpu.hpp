/*****************************************
@file
@brief Generate fake data in gpu buffer format.
- fakeGpu : public KotekanProcess
*****************************************/
#ifndef FAKE_GPU_HPP
#define FAKE_GPU_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "fakeGpuPattern.hpp"

#include <memory>   // for unique_ptr
#include <stdint.h> // for int32_t
#include <string>   // for string


/**
 * @class fakeGpu
 * @brief Generate fake data in gpu buffer format.
 *
 * Produce data formatted with the gpu buffer blocked packing.
 *
 * @par Buffers
 * @buffer out_buf The buffer of fake data.
 *         @buffer_format N2 GPU buffer format
 *         @buffer_metadata chimeMetadata
 *
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in
 *                              the correlator data
 * @conf  block_size            Int. The block size of the packed data
 * @conf  num_ev                Int. The number of eigenvectors to be stored
 * @conf  freq_id               Int. The single frequency ID to generate frames
 *                              for.
 * @conf  cadence               Float. The interval of time (in seconds)
 *                              between frames.
 * @conf  pre_accumulate        Bool. Simulate the GPU data before any
 *                              accumulation. This ignores any cadence setting.
 * @conf  samples_per_data_set  Int. FPGA seq ticks per frame. Only use for
 *                              `pre_accumulate`.
 * @conf  wait                  Bool. Sleep to try and output data at roughly
 *                              the correct cadence.
 * @conf  num_frames            Int. Exit after num_frames have been produced. If
 *                              less than zero, no limit is applied. Default
 *                              is `-1`.
 * @conf  pattern               String. Name of the pattern to fill with. These
 *                              patterns are registerd subclasses of
 *                              fakeGpuPattern.
 * @conf  drop_probability      Float. Probability that any individual frame gets
 *                              dropped. Default is zero, i.e. no frames are dropped.
 *
 * @note Look at the documentation for the test patterns to see any addtional
 *       configuration they require.
 *
 * @warning The `stream_id_t` in the metadata is likely to be invalid as it is
 *          generated only such that it is decoded back to the input frequency
 *          id.
 * @author Richard Shaw
 */
class FakeGpu : public kotekan::Stage {
public:
    FakeGpu(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container);
    ~FakeGpu();
    void main_thread() override;


private:
    Buffer* out_buf;

    // Parameters read from the config
    int freq;
    float cadence;
    int32_t block_size;
    int32_t num_elements;
    int32_t samples_per_data_set;
    int32_t num_freq_in_frame;
    bool pre_accumulate;
    bool wait;
    int32_t num_frames;
    float drop_probability;

    // Pattern to use for filling
    std::unique_ptr<FakeGpuPattern> pattern;
};

#endif // FAKE_GPU
