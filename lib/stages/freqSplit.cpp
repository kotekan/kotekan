#include "freqSplit.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for operator!=
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for wait_for_full_frame, allocate_new_metadata_object, mark_f...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, datasetManager, state_id_t
#include "datasetState.hpp"    // for freqState, datasetState, state_uptr
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO
#include "visBuffer.hpp"       // for visFrameView
#include "visUtil.hpp"         // for freq_ctype

#include <algorithm>    // for max
#include <atomic>       // for atomic_bool
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <memory>       // for make_unique, allocator_traits<>::value_type
#include <regex>        // for match_results<>::_Base_type
#include <stddef.h>     // for size_t
#include <stdexcept>    // for out_of_range, runtime_error
#include <stdint.h>     // for uint32_t
#include <system_error> // for system_error
#include <tuple>        // for tie, tuple


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(freqSplit);


freqSplit::freqSplit(Config& config, const std::string& unique_name,
                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&freqSplit::main_thread, this)) {

    // Get the list of buffers that this stage shoud connect to
    std::vector<std::string> output_buffer_names =
        config.get<std::vector<std::string>>(unique_name, "out_bufs");

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Fetch the output buffers, register them, and store them in our buffer vector
    for (auto name : output_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_producer(buf, unique_name.c_str());
        out_bufs.push_back({buf, 0});
    }

    _split_freq = config.get_default<double>(unique_name, "split_freq", 512);
}

std::array<dset_id_t, 2> freqSplit::change_dataset_state(dset_id_t input_dset_id) {
    auto& dm = datasetManager::instance();

    // create new frequency dataset state
    const freqState* freq_state_ptr = dm.dataset_state<freqState>(input_dset_id);
    if (freq_state_ptr == nullptr) {
        FATAL_ERROR("Set to not use dataset_broker and couldn't find freqState ancestor of dataset "
                    "{}. Make sure there is a stage upstream in the config, that adds a "
                    "freqState.\nExiting...",
                    input_dset_id);
    }

    const std::vector<std::pair<uint32_t, freq_ctype>>& input_freqs = freq_state_ptr->get_freqs();
    std::vector<std::pair<uint32_t, freq_ctype>> output_freqs_lower, output_freqs_higher;

    for (size_t i = 0; i < input_freqs.size(); i++) {
        if (input_freqs.at(i).first < _split_freq)
            output_freqs_lower.push_back(input_freqs.at(i));
        else
            output_freqs_higher.push_back(input_freqs.at(i));
    }

    state_uptr fstate_lower = std::make_unique<freqState>(output_freqs_lower);
    state_uptr fstate_higher = std::make_unique<freqState>(output_freqs_higher);

    // add states (async)
    state_id_t freq_state_id_lower = dm.add_state(std::move(fstate_lower)).first;
    state_id_t freq_state_id_higher = dm.add_state(std::move(fstate_higher)).first;

    return {{dm.add_dataset(freq_state_id_lower, input_dset_id),
             dm.add_dataset(freq_state_id_higher, input_dset_id)}};
}

void freqSplit::main_thread() {

    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    unsigned int buf_ind;

    dset_id_t input_dset_id;
    std::array<dset_id_t, 2> output_dset_id = {{dset_id_t::null, dset_id_t::null}};

    // Wait for a frame in the input buffer in order to get the dataset ID
    if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
        return;
    }
    input_dset_id = visFrameView(in_buf, input_frame_id).dataset_id;
    _output_dset_id = std::async(&freqSplit::change_dataset_state, this, input_dset_id);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (input_dset_id != input_frame.dataset_id) {
            input_dset_id = input_frame.dataset_id;
            _output_dset_id = std::async(&freqSplit::change_dataset_state, this, input_dset_id);
        }

        // frequency index of this frame
        freq = input_frame.freq_id;

        // Choose output buffer for this frequency:
        if (freq < _split_freq) {
            buf_ind = 0;
        } else {
            buf_ind = 1;
        }

        auto& buffer_pair = out_bufs[buf_ind];
        std::tie(buf, frame_id) = buffer_pair;

        INFO("Buffer {:d} has frame_id={:d}", buf_ind, frame_id);

        // Wait for the output buffer to be empty of data
        if (wait_for_empty_frame(buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        allocate_new_metadata_object(buf, frame_id);

        // Copy frame and create view
        auto frame = visFrameView(buf, frame_id, input_frame);

        // Are we waiting for a new dataset ID?
        if (_output_dset_id.valid())
            output_dset_id = _output_dset_id.get();

        // set the dataset ID in the outgoing frame
        frame.dataset_id = output_dset_id.at(buf_ind);

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
