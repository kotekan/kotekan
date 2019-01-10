/**
 * @file
 * @brief Contains a consumer to verify that buffers match a constant value.
 *  - constDataCheck : public KotekanProcess
 */

#ifndef CONST_DATA_CHECK_H
#define CONST_DATA_CHECK_H

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <unistd.h>

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
class constDataCheck : public kotekan::KotekanProcess {
public:
    /// Constructor, also initializes internal variables from config.
    constDataCheck(kotekan::Config& config, const string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Destructor, cleans up local allocs.
    ~constDataCheck();

    /// Primary loop to wait for buffers, verify, lather, rinse and repeat.
    void main_thread() override;

private:
    struct Buffer* buf;
    vector<int32_t> ref_real;
    vector<int32_t> ref_imag;
};

#endif
