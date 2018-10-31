#include "freqSlicer.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "datasetManager.hpp"
#include <algorithm>
#include <signal.h>


REGISTER_KOTEKAN_PROCESS(freqSplit);
REGISTER_KOTEKAN_PROCESS(freqSubset);


freqSplit::freqSplit(Config& config,
                     const string& unique_name,
                     bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&freqSplit::main_thread, this)) {

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> output_buffer_names =
        config.get<std::vector<std::string>>(unique_name, "out_bufs");

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Fetch the output buffers, register them, and store them in our buffer vector
    for(auto name : output_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_producer(buf, unique_name.c_str());
        out_bufs.push_back({buf, 0});
    }

    _use_dataset_manager = config.get_default<bool>(
                unique_name, "use_dataset_manager", false);
}

void freqSplit::apply_config(uint64_t fpga_seq) {

}

std::array<dset_id_t, 2>
freqSplit::change_dataset_state(dset_id_t input_dset_id) {
    auto& dm = datasetManager::instance();

    // create new frequency dataset state
    const freqState* input_freq_ptr =
            dm.dataset_state<freqState>(input_dset_id);
    if (input_freq_ptr == nullptr)
        throw std::runtime_error("freqSplit: Could not find freqState " \
                                 "ancestor of dataset "
                                 + std::to_string(input_dset_id));

    const std::vector<std::pair<uint32_t, freq_ctype>>& input_freqs =
            input_freq_ptr->get_freqs();
    std::vector<std::pair<uint32_t, freq_ctype>> output_freqs_lower,
            output_freqs_higher;

    for (size_t i = 0; i < input_freqs.size(); i++) {
        if (input_freqs.at(i).first < SPLIT_FREQ)
            output_freqs_lower.push_back(input_freqs.at(i));
        else
            output_freqs_higher.push_back(input_freqs.at(i));
    }

    state_uptr fstate_lower = std::make_unique<freqState>(output_freqs_lower);
    state_uptr fstate_higher = std::make_unique<freqState>(output_freqs_higher);
    state_id_t freq_state_id_lower =
            dm.add_state(std::move(fstate_lower)).first;
    state_id_t freq_state_id_higher =
            dm.add_state(std::move(fstate_higher)).first;

    return {dm.add_dataset(dataset(freq_state_id_lower, input_dset_id)),
                dm.add_dataset(dataset(freq_state_id_higher, input_dset_id))};
}

void freqSplit::main_thread() {

    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    unsigned int buf_ind;

    dset_id_t input_dset_id;
    std::array<dset_id_t, 2> output_dset_id = {0, 0};

    // flag indicating if the communication with the ds broker should be retried
    bool broker_retry = false;

    if (_use_dataset_manager) {
        // Wait for the input buffer to be filled with data
        // in order to get the dataset ID
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            return;
        }
        input_dset_id = visFrameView(in_buf, input_frame_id).dataset_id;
        _output_dset_id = std::async(change_dataset_state, input_dset_id);
    }

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (_use_dataset_manager &&
            (input_dset_id != input_frame.dataset_id || broker_retry)) {
            input_dset_id = input_frame.dataset_id;
            _output_dset_id = std::async(change_dataset_state, input_dset_id);
        }

        // frequency index of this frame
        freq = input_frame.freq_id;

        // Choose output buffer for this frequency:
        if(freq < SPLIT_FREQ) {
            buf_ind = 0;
        } else {
            buf_ind = 1;
        }

        auto& buffer_pair = out_bufs[buf_ind];
        std::tie(buf, frame_id) = buffer_pair;

        INFO("Buffer %i has frame_id=%i", buf_ind, frame_id);

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(buf, unique_name.c_str(),
                                frame_id) == nullptr) {
            break;
        }

        allocate_new_metadata_object(buf, frame_id);

        // Copy frame and create view
        auto frame = visFrameView(buf, frame_id, input_frame);

        // set the dataset ID in the outgoing frame
        if (_use_dataset_manager) {
            if (_output_dset_id.valid()) {
                try {
                    output_dset_id = _output_dset_id.get();
                } catch (std::exception& e) {
                   WARN("freqSplit: Dropping frame, failure in " \
                        "datasetManager: %s", e.what());

                    // Mark the input buffer and move on
                    mark_frame_empty(in_buf, unique_name.c_str(),
                                     input_frame_id);
                    // Advance the current input frame ids
                    std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
                    input_frame_id = (input_frame_id + 1) % in_buf->num_frames;

                    broker_retry = true;
                    continue;
                }
                broker_retry = false;
            }
            frame.dataset_id = output_dset_id.at(buf_ind);
        } else
            frame.dataset_id = 0;

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }

}


freqSubset::freqSubset(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&freqSubset::main_thread, this)) {

    // Get list of frequencies to subset from config
    _subset_list = config.get<std::vector<uint32_t>>(unique_name,
                                                     "subset_list");

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    _use_dataset_manager = config.get_default(
                unique_name, "use_dataset_manager", false);
}

void freqSubset::apply_config(uint64_t fpga_seq) {

}

dset_id_t freqSubset::change_dataset_state(dset_id_t input_dset_id,
                                           std::vector<uint32_t>& subset_list) {
    auto& dm = datasetManager::instance();

    // create new frequency dataset state
    const freqState* input_freq_ptr =
            dm.dataset_state<freqState>(input_dset_id);
    if (input_freq_ptr == nullptr)
        throw std::runtime_error("freqSubset: Could not find freqState " \
                                 "ancestor of dataset "
                                 + std::to_string(input_dset_id));

    // put the input_freqs in a map and then pick the ones that are in the
    // subset list out of the map again into the output_freqs
    const std::vector<std::pair<uint32_t, freq_ctype>>&
            vec_input_freqs(input_freq_ptr->get_freqs());
    std::map<uint32_t, freq_ctype> input_freqs;

    for (auto const& i : vec_input_freqs) {
        input_freqs.insert(i);
    }
    std::vector<std::pair<uint32_t, freq_ctype>> output_freqs;

    for (uint32_t i = 0; i < subset_list.size(); i++)
        if (input_freqs.find(subset_list.at(i)) != input_freqs.end())
            output_freqs.push_back(std::pair<uint32_t, freq_ctype>(
                                       subset_list.at(i),
                                       input_freqs.at(subset_list.at(i))));

    state_uptr fstate = std::make_unique<freqState>(output_freqs);
    state_id_t freq_state_id = dm.add_state(std::move(fstate)).first;
    return dm.add_dataset(dataset(freq_state_id, input_dset_id));
}

void freqSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    dset_id_t output_dset_id = 0;
    dset_id_t input_dset_id;

    // flag indicating if the communication with the ds broker should be retried
    bool broker_retry = false;

    if (_use_dataset_manager) {
        // Wait for the input buffer to be filled with data
        // in order to get the dataset ID
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            return;
        }
        input_dset_id = visFrameView(in_buf, input_frame_id).dataset_id;
        _output_dset_id = std::async(change_dataset_state,
                                    input_dset_id, std::ref(_subset_list));
    }

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (_use_dataset_manager &&
            (input_dset_id != input_frame.dataset_id || broker_retry)) {
            input_dset_id = input_frame.dataset_id;
            _output_dset_id = std::async(change_dataset_state,
                                         input_dset_id,
                                         std::ref(_subset_list));
        }

        // frequency index of this frame
        freq = input_frame.freq_id;

        // If this frame is part of subset
        // TODO: Apparently std::set can be used to speed up this search
        if (std::find(_subset_list.begin(), _subset_list.end(), freq) !=
            _subset_list.end()) {

            // Wait for the output buffer to be empty of data
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }

            allocate_new_metadata_object(out_buf, output_frame_id);

            // Copy frame and create view
            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             input_frame);

            // set the dataset ID in the outgoing frame
            if (_use_dataset_manager) {
                if (_output_dset_id.valid()) {
                    try {
                        output_dset_id = _output_dset_id.get();
                    } catch (std::exception& e) {
                       WARN("freqSubset: Dropping frame, failure in " \
                            "datasetManager: %s",
                            e.what());

                        // Mark the input buffer and move on
                        mark_frame_empty(in_buf, unique_name.c_str(),
                                                                input_frame_id);
                        // Advance the current input frame id
                        input_frame_id =
                                (input_frame_id + 1) % in_buf->num_frames;

                        broker_retry = true;
                        continue;
                    }
                    broker_retry = false;
                }
            }
            output_frame.dataset_id = output_dset_id;

            // Mark the output buffer and move on
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

            // Advance the current frame ids
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }
        // Mark the input buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(),
                                                input_frame_id);
        // Advance the current input frame id
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
