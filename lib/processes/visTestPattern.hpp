/*****************************************
@file
@brief Process for comparing against an expected test pattern in the visBuffers.
- visTestPattern : public KotekanProcess
*****************************************/
#ifndef VISTESTPATTERN_HPP
#define VISTESTPATTERN_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "bufferContainer.hpp"
#include "buffer.h"

#include <stddef.h>
#include <fstream>
#include <string>
#include <vector>


/**
 * @class visTestPattern
 * @brief Checks if the visibility data matches a given expected pattern.
 *
 * Errors are calculated as the norm of the difference between the expected and
 * the actual (complex) visibility value.
 * For bad frames, the following data is written to a csv file specified in the
 * config:
 * fpga_count:  FPGA counter for the frame
 * time:        the frames timestamp
 * freq_id:     the frames frequency ID
 * num_bad:     number of values that have an error higher then the threshold
 * avg_err:     average error of bad values
 * min_err:     minimum error of bad values
 * max_err:     maximum error of bad balues
 * expected:    the visibility value that was expected according to the mode
 *
 * Additionally a report is printed in a configured interval.
 *
 * The modes are defined as follows:
 * `test_pattern_simple`: All visibility values are `1 + 0j`.
 * `test_pattern_freq`: The value `frequencies` defines frequency bins, the
 * visibilities in frames for those defined frequencies will have the values
 * defined in `freq_values` (in the same order). The visibilities in all other
 * frames will have the value, set by `default_val`.
 *
 * @par Buffers
 * @buffer in_buf               The buffer to debug
 *         @buffer_format       visBuffer structured
 *         @buffer_metadata     visMetadata
 * @buffer out_buf              All frames found to contain errors
 *         @buffer_format       visBuffer structured
 *         @buffer_metadata     visMetadata
 *
 * @conf  out_file              String. Path to the file to dump all output in.
 * @conf  report_freq           Int. Number of frames to print a summary for.
 * @conf  default_val           CFloat. Default expected visibility value.
 * @conf  freq_values           Array of CFloat. Expected visibility value for
 * each frequency bin (used in mode `test_pattern_freq`).
 * @conf  frequencies           Array of Float. Frequency bins (used in mode
 * `test_pattern_freq`).
 * @conf  num_freq              Float. Total number of frequencies in the frames
 * (used in mode `test_pattern_freq`).
 * @conf  tolerance             Float. Defines what difference to the expected
 * value is an error.
 * @conf mode                   String. One of `test_pattern_simple` and
 * `test_pattern_freq`.
 *
 * @author Rick Nitsche
 */
class visTestPattern : public KotekanProcess {

public:
    visTestPattern(Config &config,
             const std::string& unique_name,
             bufferContainer &buffer_container);

    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    // Config parameters
    float tolerance;
    size_t report_freq;
    std::string mode;
    cfloat exp_val;
    std::vector<cfloat> exp_val_freq;
    size_t num_freq;

    // file to dump all info in
    std::ofstream outfile;
    std::string outfile_name;
};

#endif // VISTESTPATTERN_HPP
