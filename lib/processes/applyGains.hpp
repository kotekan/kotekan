#ifndef APPLY_GAINS_HPP
#define APPLY_GAINS_HPP

#include <unistd.h>
#include "fpga_header_functions.h"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visFile.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"
#include "updateQueue.hpp"


class applyGains : public KotekanProcess {

public:

    /// Default constructor
    applyGains(Config &config,
              const string& unique_name,
              bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    /// Blend gains old and new
    cfloat blend_gains(int idx);

    /// Main loop for the process
    void main_thread();

    /// Callback function to receive updates on timestamps from configUpdater
    bool receive_update(nlohmann::json &json);

private:

    // Parameters saved from the config files

    /// Path to gains file
    // TODO: delete? Whait to see how I get this from config updater...
    std::string gains_path;

    /// Number of gains updates to maintain
    uint64_t num_kept_updates;

    /// Name of updatable config block for gain timestamps
    std::string updatable_config;

    /// Time over which to blend old and new gains in seconds. Default is 5 minutes.
    double tblend;

    /// Coefficients for blending gains old and new
    float coef_new = 1;
    float coef_old = 0;

    /// Vectors for storing old and new gains
    std::vector<cfloat> gain_new;
    std::vector<cfloat> gain_old;

    /// The gains and when to start applying them in a FIFO (len set by config)
    updateQueue<std::vector<std::vector<cfloat>>> gains2;

    /// Output buffer with gains applied
    Buffer * out_buf;
    /// Input buffer to read from
    Buffer * in_buf;

};


#endif

