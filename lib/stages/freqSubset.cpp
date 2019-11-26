#include "freqSubset.hpp"

#include "StageFactory.hpp"
#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cxxabi.h>
#include <exception>
#include <functional>
#include <inttypes.h>
#include <map>
#include <memory>
#include <regex>
#include <signal.h>
#include <stdexcept>
#include <utility>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(freqSubset);


freqSubset::freqSubset(Config& config, const string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&freqSubset::main_thread, this)) {

    // Get list of frequencies to subset from config
    _subset_list = config.get<std::vector<uint32_t>>(unique_name, "subset_list");

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

dset_id_t freqSubset::change_dataset_state(dset_id_t input_dset_id,
                                           std::vector<uint32_t>& subset_list) {
    auto& dm = datasetManager::instance();

    // create new frequency dataset state
    const freqState* freq_state_ptr = dm.dataset_state<freqState>(input_dset_id);
    if (freq_state_ptr == nullptr) {
        FATAL_ERROR("Set to not use dataset_broker and couldn't find freqState ancestor of dataset "
                    "{}. Make sure there is a stage upstream in the config, that adds a "
                    "freqState.\nExiting...",
                    input_dset_id);
    }

    // put the input_freqs in a map and then pick the ones that are in the
    // subset list out of the map again into the output_freqs
    const std::vector<std::pair<uint32_t, freq_ctype>>& vec_input_freqs(
        freq_state_ptr->get_freqs());
    std::map<uint32_t, freq_ctype> input_freqs;

    for (auto const& i : vec_input_freqs) {
        input_freqs.insert(i);
    }
    std::vector<std::pair<uint32_t, freq_ctype>> output_freqs;

    for (uint32_t i = 0; i < subset_list.size(); i++)
        if (input_freqs.find(subset_list.at(i)) != input_freqs.end())
            output_freqs.push_back(std::pair<uint32_t, freq_ctype>(
                subset_list.at(i), input_freqs.at(subset_list.at(i))));

    state_uptr fstate = std::make_unique<freqState>(output_freqs);
    state_id_t freq_state_id = dm.add_state(std::move(fstate)).first;
    return dm.add_dataset(freq_state_id, input_dset_id);
}

void freqSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    dset_id_t output_dset_id = dset_id_t::null;
    dset_id_t input_dset_id;

    // Wait for a frame in the input buffer in order to get the dataset ID
    if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
        return;
    }
    input_dset_id = visFrameView(in_buf, input_frame_id).dataset_id;
    _output_dset_id =
        std::async(&freqSubset::change_dataset_state, this, input_dset_id, std::ref(_subset_list));

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
            _output_dset_id = std::async(&freqSubset::change_dataset_state, this, input_dset_id,
                                         std::ref(_subset_list));
        }

        // frequency index of this frame
        freq = input_frame.freq_id;

        // If this frame is part of subset
        // TODO: Apparently std::set can be used to speed up this search
        if (std::find(_subset_list.begin(), _subset_list.end(), freq) != _subset_list.end()) {

            // Wait for the output buffer to be empty of data
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                break;
            }

            allocate_new_metadata_object(out_buf, output_frame_id);

            // Copy frame and create view
            auto output_frame = visFrameView(out_buf, output_frame_id, input_frame);

            // set the dataset ID in the outgoing frame
            if (_output_dset_id.valid())
                output_dset_id = _output_dset_id.get();

            output_frame.dataset_id = output_dset_id;

            // Mark the output buffer and move on
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }
        // Mark the input buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        // Advance the current input frame id
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
