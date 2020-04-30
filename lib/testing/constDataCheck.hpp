/**
 * @file
 * @brief Contains a consumer to verify that buffers match a constant value.
 *  - constDataCheck : public Stage
 */

#ifndef CONST_DATA_CHECK_H
#define CONST_DATA_CHECK_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // IWYU pragma: keep
#include "bufferContainer.hpp" // for bufferContainer

#include <atomic>     // for atomic_bool   // IWYU pragma: keep
#include <cstdint>    // for int32_t
#include <functional> // for _Bind_helper<>::type, bind, function   // IWYU pragma: keep
#include <string>     // for string
#include <vector>     // for vector


/**
 * @class constDataCheck
 * @brief Consumer which verifies constant-value ``Buffers``, useful for verification.
 *        Will throw ERROR messages for failed verifications.
 *
 * @par Buffers
 * @buffer in_buf A kotekan buffer which will be verified, can be any size.
 *     @buffer_format Array of @c ints.
 *     @buffer_metadata none
 *
 * @conf   real        Int Array. Expected real component, will loop through the array on subsequent
 * frames.
 * @conf   imag        Int Array. Expected imag component, will loop through the array on subsequent
 * frames.
 *
 * @author Andre Renard
 *
 */
class constDataCheck : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    constDataCheck(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Destructor, cleans up local allocs.
    ~constDataCheck();

    /// Primary loop to wait for buffers, verify, lather, rinse and repeat.
    void main_thread() override;

private:
    struct Buffer* buf;
    std::vector<int32_t> ref_real;
    std::vector<int32_t> ref_imag;
    int num_frames_to_test;
};

#endif
