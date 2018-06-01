/*****************************************
@file
@brief Processe for checking that FPGA counts are not older than 1h.
- countCheck : public KotekanProcess

*****************************************/
#ifndef COUNT_CHECK_HPP
#define COUNT_CHECK_HPP

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"


/**
 * @class countCheck
 * @brief Processe for checking that FPGA counts are not older than 1h.
 *
 * This task checks that the current FPGA count is not more than 1 hour
 * older than the previous one.
 * It raises SIGINT in case it is.
 *
 * @par Buffers
 * @buffer in_buf The buffer whose fpga count will be checked
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @author Mateus Fandino
 */
class countCheck : public KotekanProcess {

public:

    // Default constructor
    countCheck(Config &config,
              const string& unique_name,
              bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    // Main loop for the process
    void main_thread();

private:
    // Store the value of previous FPGA sequence count:
    uint64_t prev_fpga_seq;
    Buffer * in_buf;
};

#endif
