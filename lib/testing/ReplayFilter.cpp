#include "ReplayFilter.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for allocate_new_metadata_object, mark_frame_full, register_p...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, dset_id_t, datasetManager
#include "datasetState.hpp"    // for eigenvalueState, freqState, inputState, metadataState
#include "errors.h"            // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "factory.hpp"         // for FACTORY
#include "kotekanLogging.hpp"  // for INFO, DEBUG
#include "version.h"           // for get_git_commit_hash
#include "visBuffer.hpp"       // for visFrameView
#include "visUtil.hpp"         // for prod_ctype, input_ctype, double_to_ts, current_time, freq...

#include <chrono>

using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;


REGISTER_KOTEKAN_STAGE(ReplayFilter);


ReplayFilter::ReplayFilter(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ReplayFilter::main_thread, this)),
    _start_time(config.get_default<double>(unique_name, "start_time", -1)),
    _fpga_length(config.get_default<double>(unique_name, "fpga_length", -1)),
    _wait(config.get_default<bool>(unique_name, "num_elements", true)),
    _modify_times(config.get_default<bool>(unique_name, "modify_times", true)),
    _drop_empty(config.get_default<bool>(unique_name, "drop_empty", true))
{
    // Fetch the output buffer and register it
    std::string in_name = config.get<std::string>(unique_name, "in_buf");
    in_buf = buffer_container.get_buffer(in_name);
    register_consumer(in_buf, unique_name.c_str());

    // Fetch the output buffer and register it
    std::string out_name = config.get<std::string>(unique_name, "out_buf");
    out_buf = buffer_container.get_buffer(out_name);
    register_producer(out_buf, unique_name.c_str());
}

void ReplayFilter::main_thread() {

    frameID input_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    /*
    timespec ts = double_to_ts(start_time);

    // Sleep before starting up
    timespec ts_sleep = double_to_ts(sleep_before);
    nanosleep(&ts_sleep, nullptr);

    double start = _start_time < 0 ? _start_time : current_time();
    */

    auto start_tp = std::chrono::steady_clock::now();
    auto clock_tick = std::chrono::duration<double>(_fpga_length);

    //INFO("Start time: {}, clock tick {}", start_tp, clock_tick);
    INFO("clock tick {}", clock_tick);

    std::optional<uint64_t> fpga0;


    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // Drop empty frames if requested
        // This identifies empty frames by them having a null dataset ID and no
        // accumulated data. This should be a pretty good test, but we haven't
        // formally specified what "empty" means
        if (_drop_empty && input_frame.dataset_id == dset_id_t::null && input_frame.fpga_seq_total == 0) {
            DEBUG2("Skipping empty frame.")
            // Mark the input buffer and move on
            mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);
            continue;
        }

        // Set the first FPGA timesample
        if (!fpga0) {
            fpga0 = std::get<0>(input_frame.time);
        }

        uint64_t fpga = std::get<0>(input_frame.time);
        INFO("{:s}", input_frame.summary());
        INFO("Diff {}, {}, {}", fpga, fpga0.value(), fpga - fpga0.value());
        auto frame_tp = start_tp + (fpga - fpga0.value()) * clock_tick;
        INFO("Frame time: {}", std::chrono::duration_cast<std::chrono::seconds>(frame_tp - start_tp));

        // Wait for the output buffer to be empty of data
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }
        // Copy input frame to output frame and create view
        allocate_new_metadata_object(out_buf, output_frame_id);
        auto output_frame = visFrameView(out_buf, output_frame_id, input_frame);
        (void)output_frame;

        // Mark the input and output buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);
        std::this_thread::sleep_until(frame_tp);
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);
    }
}