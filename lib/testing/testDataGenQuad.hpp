/**
 * @file
 * @brief Contains a 4-buffer data generation producer for kotekan.
 *  - testDataGenQuad : public Stage
 */

#ifndef TEST_DATA_GEN_QUAD_H
#define TEST_DATA_GEN_QUAD_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for int32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class testDataGenQuad
 * @brief Producer which feeds fake data into 4 ``Buffers`` and simultaneously releases them.
 * The input buffers are fully seeded prior to beginning, then released at timed intervals.
 * Useful for testing CHIME things when we want operations synchronized between GPUs.
 *
 * @par Buffers
 * @buffer out_buf0 A kotekan buffer which will be fed, can be any size.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 * @buffer out_buf0 A kotekan buffer which will be fed, can be any size.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 * @buffer out_buf0 A kotekan buffer which will be fed, can be any size.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 * @buffer out_buf0 A kotekan buffer which will be fed, can be any size.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 *
 * @conf   type        String. Must be one of @c "random" oe @c "const"
 * @conf   value       Int Array. Values to seed if using constant data genration.
 *                     These are seeded into sequential frames, and looped through until
 *                     all frames in the target buffers are filled.
 *
 * @bug     Doesn't actually support random data, always generates constant.
 * @todo    Make the frame-to-frame sleep depend on the size of buffers.
 *
 * @author Keith Vanderlinde
 *
 */
class testDataGenQuad : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    testDataGenQuad(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);

    /// Destructor, cleans up local allocs.
    ~testDataGenQuad();

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

private:
    Buffer* buf[4];
    std::string type;
    std::vector<int32_t> value;
};

#endif
