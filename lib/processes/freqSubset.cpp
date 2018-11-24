#include "freqSubset.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "datasetManager.hpp"
#include <algorithm>
#include <signal.h>

REGISTER_KOTEKAN_PROCESS(freqSubset);


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

dset_id_t freqSubset::change_dataset_state(dset_id_t input_dset_id,
                                           std::vector<uint32_t>& subset_list) {
    auto& dm = datasetManager::instance();

    // create new frequency dataset state
    const freqState* freq_state_ptr =
            dm.dataset_state<freqState>(input_dset_id);
    if (freq_state_ptr == nullptr)
        throw std::runtime_error("freqSubset: Could not find freqState " \
                                 "ancestor of dataset "
                                 + std::to_string(input_dset_id));

    // put the input_freqs in a map and then pick the ones that are in the
    // subset list out of the map again into the output_freqs
    const std::vector<std::pair<uint32_t, freq_ctype>>&
            vec_input_freqs(freq_state_ptr->get_freqs());
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
    return dm.add_dataset(input_dset_id, freq_state_id);
}

void freqSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    dset_id_t output_dset_id = 0;
    dset_id_t input_dset_id;

    // number of errors when dealing with dataset manager
    uint32_t err_count = 0;

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
                       prometheusMetrics::instance().add_process_metric(
                           "kotekan_dataset_manager_dropped_frame_count",
                           unique_name, ++err_count);

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