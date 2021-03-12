/*****************************************
@file
@brief Stage for comparing against an expected test pattern in the VisBuffers.
       CHIME specific.
- visTestPattern : public kotekan::Stage
*****************************************/
#ifndef VISTESTPATTERN_HPP
#define VISTESTPATTERN_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "dataset.hpp"         // for dset_id_t
#include "restServer.hpp"      // for connectionInstance
#include "visUtil.hpp"         // for cfloat, input_ctype, prod_ctype, freq_ctype

#include <fstream>  // for ofstream
#include <json.hpp> // for json
#include <map>      // for map
#include <mutex>    // for mutex
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, int8_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector


/**
 * @class visTestPattern
 * @brief Checks if the visibility data matches a given expected pattern (CHIME specific).
 *
 * This stage is just dropping incoming frames until it receives an update on what data to expect.
 * An update would be sent to the endpoint `/run-test`.  *
 * This stage will run the test for the given number of frames, then send a request to a
 * reply-endpoint to signal that the test is done. The reply endpoint is expected in the update.
 * In case there was an error during the test, it replies
 * with the error message in the result field.
 * Afterwards it will stay idle until the next test is started.
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
 *
 * Additionally a report is printed to the log in a configured interval.
 *
 * @par REST Endpoints
 * @endpoint    /run-test ``POST``  Updates the FPGA test pattern to expect and how long to run a
 *                                  test on that pattern. Required json values:
 *              name            String. Name of the test.
 *              num_frames      Int. Number of frames to check for the given test pattern. If this
 *                              is `0`, this stage will stay idle until a new update is received.
 *              test_pattern    Dictionary of String -> (List of uint8). Mapping of correlator
 *                              input serial number to a list of values for each frequency bin.
 *                              For example: `{'FCCXXYYZZ': [2048 list of uint8], ...}`, where the
 *                              2048 uint8 are the real and imaginary component for each frequency
 *                              bin and FCCXXYYZZ is the serial number of the correlator input.
 *              reply_host      String. Hostname to send test-done message to.
 *              reply_path      String. Endpoint path to send test-done message to.
 *              reply_port      Int. Port to send test-done message to.
 *
 * @par Buffers
 * @buffer in_buf               The buffer to debug
 *         @buffer_format       VisBuffer structured
 *         @buffer_metadata     VisMetadata
 * @buffer out_buf              All frames found to contain errors
 *         @buffer_format       VisBuffer structured
 *         @buffer_metadata     VisMetadata
 *
 * @conf  write_dir             String. Path to the place to dump all output in.
 * @conf  report_freq           Int. Number of frames to print a summary for (default: 1000).
 * @conf  tolerance             Float. Defines what difference to the expected value is an error
 *                              (default: 1e-6).
 * @conf  endpoint_name         String. Name for the update endpoint (default: "run_test").
 *
 * @author Rick Nitsche
 */
class visTestPattern : public kotekan::Stage {

public:
    visTestPattern(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    ~visTestPattern();

    void main_thread() override;

    /// Callback function to receive updates from configUpdater.
    void receive_update(kotekan::connectionInstance& conn, nlohmann::json& data);

private:
    /// Gets the frequency, input and product information from the datasetManager.
    void get_dataset_state(dset_id_t ds_id);

    /// Compute the expected vis data from the received FPGA buffer content.
    void compute_expected_data();

    /// Report error in error_msg to orchestrator and log. Exit kotekan.
    void exit_failed_test(std::string error_msg);

    /// Print a warning with the given message and send it as a reply.
    void reply_failure(kotekan::connectionInstance& conn, std::string& msg);

    Buffer* in_buf;
    Buffer* out_buf;

    // Config parameters
    double _tolerance;
    size_t _report_freq;

    // Report file floating point precision (function of configured tolerance).
    int8_t precision;

    // Data in the fpga buffers. Used to compute expected data.
    std::map<std::string, std::vector<cfloat>> fpga_buf_pattern;

    // Expected vis values for each input feed per frequency bin.
    std::vector<std::vector<cfloat>> expected;
    bool expected_data_ready;

    // file to dump all info in
    std::ofstream outfile;
    std::string write_dir;

    // Number of frames left to test.
    uint32_t num_frames;

    // Frequency indices.
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;

    // Input indices.
    std::vector<input_ctype> inputs;

    // Prod Ctypes
    std::vector<prod_ctype> prods;

    // Endpoint name (config value)
    std::string endpoint_name;

    // Stop at the next frame and lock the thread when an update comes in.
    std::mutex mtx_update;

    // Details on where to report back on finished test (received with update).
    std::string test_done_path;
    std::string test_done_host;
    unsigned short test_done_port;

    // Name of the test
    std::string test_name;
};

#endif // VISTESTPATTERN_HPP
