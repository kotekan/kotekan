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
 * @brief Processe that checks for acquisition re-start.
 *
 * This task finds the unix time at the start of the acquisition from 
 * the FPGA counts and the current unix time, assuming 390625 FPGA 
 * counts per second. 
 * It stores this value and checks each frame to look for changes.
 * If the initial time changes by more than 3 seconds, the process
 * raises SIGINT.
 *
 * @par Buffers
 * @buffer in_buf The buffer whose fpga count will be checked
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf  test_nframes  int. The number of frames to run before raising SIGINT.
+*                              Useful for testing.
 *                              Default is -1 which doesn't raise SIGINT if no count
 *                              failure is found
 * @author Mateus A Fandino
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
    // Store the unix time at start of correlation:
    uint64_t start_time;
    Buffer * in_buf;

    /// Parameters saved from the config files
    int test_nframes;
};

#endif
