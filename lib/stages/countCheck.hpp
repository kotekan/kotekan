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
#include <utility>  // for pair


/**
 * @class countCheck
 * @brief Stage that checks for acquisition re-start.
 *
 * This stage finds the unix time at the start of the acquisition from
 * the FPGA counts and the current unix time, using the telescope tick
 * length (seq_length_nsec). It stores this value and checks each frame
 * to look for changes.
 * If the initial time changes by more than 'start_time_tolerance' (default=3)
 * seconds, the stage raises SIGTERM.
 *
 * @par Buffers
 * @buffer in_buf The buffer whose fpga count will be checked.
 *         Supports metadata types that provide timing information:
 *         @c VisMetadata, @c N2Metadata, @c HFBMetadata, @c BeamMetadata,
 *         and @c chordMetadata.
 *
 * @conf  start_time_tolerance  int. Tolerance for the start time error in
 *                                   seconds. Default is 3.
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
    /**
     * @brief Extract timing information from metadata.
     *
     * Determines the metadata type and extracts FPGA sequence number and
     * frame time in nanoseconds.
     *
     * @param frame_id The frame index to extract timing from.
     * @return A pair of (fpga_seq, time_ns). Logs FATAL_ERROR if metadata
     *         doesn't support timing.
     */
    std::pair<int64_t, int64_t> extract_timing_info(int frame_id);

    // Store the unix time at start of correlation (nanoseconds):
    int64_t start_time_ns;
    // Length of a single FPGA tick in nanoseconds (from Telescope)
    uint64_t tick_len_ns;
    // Tolerance converted to nanoseconds
    int64_t tolerance_ns;
    Buffer* in_buf;
    // Tolerance for start time variability
    int start_time_tolerance;
};

#endif
