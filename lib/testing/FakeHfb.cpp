#include "FakeHfb.hpp"

#include "Config.hpp"          // for Config
#include "HFBFrameView.hpp"    // for HFBFrameView
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for allocate_new_metadata_object, mark_frame_full, register_p...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, dset_id_t, datasetManager
#include "datasetState.hpp"    // for eigenvalueState, freqState, inputState, metadataState
#include "errors.h"            // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "factory.hpp"         // for FACTORY
#include "kotekanLogging.hpp"  // for INFO, DEBUG
#include "version.h"           // for get_git_commit_hash
#include "visUtil.hpp"         // for prod_ctype, input_ctype, double_to_ts, current_time, freq...

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span

#include <algorithm>   // for max, fill, transform
#include <atomic>      // for atomic_bool
#include <complex>     // for complex
#include <cstdint>     // for uint32_t, int32_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function, placeholders
#include <iterator>    // for back_insert_iterator, back_inserter, begin, end
#include <memory>      // for allocator, unique_ptr
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <time.h>      // for nanosleep, timespec
#include <tuple>       // for get, make_tuple, tuple
#include <type_traits> // for __decay_and_strip<>::__type
#include <utility>     // for pair


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;


REGISTER_KOTEKAN_STAGE(FakeHfb);
REGISTER_KOTEKAN_STAGE(ReplaceHfb);


FakeHfb::FakeHfb(Config& config, const std::string& unique_name,
                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&FakeHfb::main_thread, this)) {

    // Fetch any simple configuration
    num_beams = config.get<size_t>(unique_name, "num_frb_total_beams");
    num_subfreq = config.get<size_t>(unique_name, "factor_upchan");
    sleep_before = config.get_default<float>(unique_name, "sleep_before", 0.0);
    sleep_after = config.get_default<float>(unique_name, "sleep_after", 1.0);

    // Get the output buffer
    std::string buffer_name = config.get<std::string>(unique_name, "out_buf");

    // Fetch the buffer, register it
    out_buf = buffer_container.get_buffer(buffer_name);
    register_producer(out_buf, unique_name.c_str());

    // Get frequency IDs from config
    freq = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");

    // Was a fixed dataset ID configured?
    if (config.exists(unique_name, "dataset_id")) {
        _dset_id = config.get<dset_id_t>(unique_name, "dataset_id");
        _fixed_dset_id = true;
    } else
        _fixed_dset_id = false;

    mode = config.get_default<std::string>(unique_name, "mode", "default");
    INFO("Using fill type: {:s}", mode);
    // pattern = FACTORY(FakeHfbPattern)::create_unique(mode, config, unique_name);

    // Get timing and frame params
    start_time = config.get_default<double>(unique_name, "start_time", current_time());
    cadence = config.get<float>(unique_name, "cadence");
    num_frames = config.get_default<int32_t>(unique_name, "num_frames", -1);
    wait = config.get_default<bool>(unique_name, "wait", true);

    // Get zero_weight option
    zero_weight = config.get_default<bool>(unique_name, "zero_weight", false);
}

void FakeHfb::main_thread() {

    unsigned int output_frame_id = 0, frame_count = 0;
    uint64_t fpga_seq = 0;

    timespec ts = double_to_ts(start_time);

    // Calculate the time increments in seq and ctime
    uint64_t delta_seq = (uint64_t)(800e6 / 2048 * cadence);
    uint64_t delta_ns = (uint64_t)(cadence * 1000000000);

    // Register datasetStates to describe the properties of the created stream
    dset_id_t ds_id = dset_id_t::null;
    auto& dm = datasetManager::instance();

    if (_fixed_dset_id) {
        ds_id = _dset_id;
    } else {
        std::vector<state_id_t> states;
        states.push_back(
            dm.create_state<metadataState>("not set", "FakeHfb", get_git_commit_hash()).first);

        std::vector<std::pair<uint32_t, freq_ctype>> fspec;
        // TODO: CHIME specific
        std::transform(std::begin(freq), std::end(freq), std::back_inserter(fspec),
                       [](const uint32_t& id) -> std::pair<uint32_t, freq_ctype> {
                           return {id, {800.0 - 400.0 / 1024 * id, 400.0 / 1024}};
                       });
        states.push_back(dm.create_state<freqState>(fspec).first);
        states.push_back(dm.create_state<beamState>(num_beams).first);
        states.push_back(dm.create_state<subfreqState>(num_subfreq).first);

        // Register a root state
        ds_id = dm.add_dataset(states);
    }

    // Sleep before starting up
    timespec ts_sleep = double_to_ts(sleep_before);
    nanosleep(&ts_sleep, nullptr);

    while (!stop_thread) {

        double start = current_time();

        for (auto f : freq) {

            DEBUG("Making fake visBuffer for freq={:d}, fpga_seq={:d}", f, fpga_seq);

            // Wait for the buffer frame to be free
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                break;
            }

            // Allocate metadata and get frame
            allocate_new_metadata_object(out_buf, output_frame_id);
            HFBFrameView::set_metadata((HFBMetadata*)out_buf->metadata[output_frame_id]->metadata,
                                       num_beams, num_subfreq);

            auto output_frame = HFBFrameView(out_buf, output_frame_id);

            output_frame.dataset_id = ds_id;

            // Set the frequency index
            output_frame.freq_id = f;

            // Set the time
            output_frame.time = ts;
            output_frame.fpga_seq_num = fpga_seq;

            // Set the length and total data
            output_frame.num_samples_expected = delta_seq;
            output_frame.num_samples_integrated = delta_seq;

            // Fill out the frame with the selected pattern
            // pattern->fill(output_frame);

            for (uint32_t i = 0; i < output_frame.num_beams * output_frame.num_subfreq; i++) {
                float data = (i % 2 == 0 ? output_frame.freq_id : output_frame.fpga_seq_num);

                output_frame.hfb[i] = data;
            }


            // Mark the buffers and move on
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

        // Increment time
        fpga_seq += delta_seq;
        frame_count++; // NOTE: frame count increase once for all freq

        // Increment the timespec
        ts.tv_sec += ((ts.tv_nsec + delta_ns) / 1000000000);
        ts.tv_nsec = (ts.tv_nsec + delta_ns) % 1000000000;

        // Cause kotekan to exit if we've hit the maximum number of frames
        if (num_frames > 0 && frame_count >= (unsigned)num_frames) {
            INFO("Reached frame limit [{:d} frames]. Sleeping and then exiting kotekan...",
                 num_frames);
            timespec ts = double_to_ts(sleep_after);
            nanosleep(&ts, nullptr);
            exit_kotekan(ReturnCode::CLEAN_EXIT);
            return;
        }

        // If requested sleep for the extra time required to produce a fake vis
        // at the correct cadence
        if (this->wait) {
            double diff = cadence - (current_time() - start);
            timespec ts_diff = double_to_ts(diff);
            nanosleep(&ts_diff, nullptr);
        }
    }
}

ReplaceHfb::ReplaceHfb(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ReplaceHfb::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void ReplaceHfb::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer to be empty of data
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }
        // Create view to input frame
        auto input_frame = HFBFrameView(in_buf, input_frame_id);

        // Copy input frame to output frame and create view
        auto output_frame =
            HFBFrameView::copy_frame(in_buf, input_frame_id, out_buf, output_frame_id);

        for (uint32_t i = 0; i < output_frame.num_beams * output_frame.num_subfreq; i++) {
            float data = (i % 2 == 0 ? output_frame.freq_id : output_frame.fpga_seq_num);

            output_frame.hfb[i] = data;
        }


        // Mark the output buffer and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

        // Mark the input buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
