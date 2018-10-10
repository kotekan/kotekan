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
        config.get_string_array(unique_name, "out_bufs");

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Fetch the output buffers, register them, and store them in our buffer vector
    for(auto name : output_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_producer(buf, unique_name.c_str());
        out_bufs.push_back({buf, 0});
    }
}

void freqSplit::apply_config(uint64_t fpga_seq) {

}

void freqSplit::main_thread() {

    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    unsigned int buf_ind;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);
        // frequency index of this frame
        freq = input_frame.freq_id;

        // Choose output buffer for this frequency:
        if(freq < 512) {
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
    for (auto ff : config.get_int_array(unique_name, "subset_list")) {
        subset_list.push_back((uint32_t) ff);
    }

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

}

void freqSubset::apply_config(uint64_t fpga_seq) {

}

void freqSubset::set_dataset_ids(dset_id_t input_frame_dset_id) {
    auto& dm = datasetManager::instance();

    // create new frequency dataset state
    input_dset_id = input_frame_dset_id;
    const freqState* input_freq_ptr =
           dm.closest_ancestor_of_type<freqState>(input_dset_id).second;
    if (input_freq_ptr == nullptr) {
        ERROR("freqSubset: Could not find freqState for incoming " \
              "dataset with ID %d.", input_dset_id);
        raise(SIGINT);
        return;
    }

    const vector<pair<uint32_t, freq_ctype>>& input_freqs =
            input_freq_ptr->get_freqs();
    vector<pair<uint32_t, freq_ctype>> output_freqs;

    for (uint32_t i = 0; i < subset_list.size(); i++) {
        try {
            output_freqs.push_back(input_freqs.at(subset_list[i]));
        } catch (std::out_of_range e) {
            WARN("freqSlicer: Could not find frequency with ID %d in " \
                 "incoming dataset %d: %s", subset_list[i],
                 input_dset_id, e.what());
        }
    }

    state_uptr fstate = std::make_unique<freqState>(output_freqs);
    state_id_t freq_state_id = dm.add_state(std::move(fstate)).first;
    output_dset_id = dm.add_dataset(dataset(freq_state_id,
                                            input_dset_id));
}

void freqSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;

    // Wait for the input buffer to be filled with data
    // in order to get the dataset ID
    if(wait_for_full_frame(in_buf, unique_name.c_str(),
                           input_frame_id) == nullptr) {
        return;
    }
    set_dataset_ids(visFrameView(in_buf, input_frame_id).dataset_id);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (input_dset_id != input_frame.dataset_id)
            set_dataset_ids(input_frame.dataset_id);

        // frequency index of this frame
        freq = input_frame.freq_id;

        // If this frame is part of subset
        // TODO: Apparently std::set can be used to speed up this search
        if (std::find(subset_list.begin(), subset_list.end(), freq) !=
            subset_list.end()) {

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
