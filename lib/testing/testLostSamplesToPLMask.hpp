#ifndef TEST_LOST_SAMPLES_TO_PL_MASK_HPP
#define TEST_LOST_SAMPLES_TO_PL_MASK_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Produces test data for lostSampleToPLMask.
 *
 * @par Buffers
 * @buffer pl_mask_buf Kotekan buffer for package loss mask
 *     @buffer_format [time / 2 % 64][dish / 8][polr][freq / 4][time / 2 / 64]
 *     @buffer_metadata chordMetadata
 * @buffer lost_samples_buf Array of flags which indicate if a sample in a given location is lost
 *     @buffer_format Array of flags uint8_t flags which are either 0 (unset) or 1 (set)
 *     @buffer_metadata chordMetadata
 *
 * @author Roland Haas
 */
class testLostSamplesToPLMask : public kotekan::Stage {
public:
    /// Standard constructor
    testLostSamplesToPLMask(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~testLostSamplesToPLMask();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread() override;

private:
    /// The buffer with the package loss data
    Buffer* pl_mask_buf;

    /// The buffer with the array of flags indicating lost data.
    std::vector<Buffer*> lost_samples_bufs;
};

#endif /* TEST_LOST_SAMPLES_TO_PL_MASK_HPP */
