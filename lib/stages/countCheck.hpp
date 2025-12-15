/*****************************************
@file
@brief Stage for checking that FPGA counts are not older than 1h.
- countCheck : public kotekan::Stage

*****************************************/
#ifndef COUNT_CHECK_HPP
#define COUNT_CHECK_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <stdint.h> // for int64_t
#include <string>   // for string


/**
 * @class countCheck
 * @brief Stage that checks for acquisition re-start.
 *
 * This stage finds the unix time at the start of the acquisition from
 * the FPGA counts and the current unix time, assuming 390625 FPGA
 * counts per second.
 * It stores this value and checks each frame to look for changes.
 * If the initial time changes by more than 'start_time_tolerance' (default=3)
 * seconds, the stage raises SIGTERM.
 *
 * @par Buffers
 * @buffer in_buf The buffer whose fpga count will be checked
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf  start_time_tolerance  Int (default 3). Allowed change in derived start time (seconds)
 *                              before exiting.
 *
 * @par Example
 * @code
 * count_check:
 *   kotekan_stage: countCheck
 *   in_buf: vis_in
 *   start_time_tolerance: 3
 * @endcode
 *
 * @author Mateus A Fandino
 */
class countCheck : public kotekan::Stage {

public:
    // Default constructor
    countCheck(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    // Main loop for the stage
    void main_thread() override;

private:
    // Store the unix time at start of correlation:
    int64_t start_time;
    Buffer* in_buf;
    // Tolerance for start time variability
    int start_time_tolerance;
};

#endif
