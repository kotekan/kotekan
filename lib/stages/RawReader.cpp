#include "RawReader.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for Hash, operator<, operator==
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_full, wait_for_empty_frame, mark_frame_empty
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, dset_id_t, datasetManager, DS_UNIQUE_NAME
#include "datasetState.hpp"    // for freqState, timeState, metadataState
#include "kotekanLogging.hpp"  // for INFO, FATAL_ERROR, DEBUG, WARN, ERROR
#include "visBuffer.hpp"       // for VisFrameView
#include "visUtil.hpp"         // for time_ctype, freq_ctype, frameID, modulo, current_time

#include "json.hpp" // for basic_json<>::object_t, json, basic_json, basic_json<>::v...

#include <atomic>       // for atomic_bool
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for async, future
#include <memory>       // for allocator_traits<>::value_type
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error, invalid_argument, out_of_range
#include <system_error> // for system_error
#include <tuple>        // for get

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(ensureOrdered);

ensureOrdered::ensureOrdered(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ensureOrdered::main_thread, this)) {

    max_waiting = config.get_default<size_t>(unique_name, "max_waiting", 100);

    chunked = config.exists(unique_name, "chunk_size");
    if (chunked) {
        chunk_size = config.get<std::vector<int>>(unique_name, "chunk_size");
        if (chunk_size.size() != 3) {
            FATAL_ERROR("Chunk size needs exactly three elements (got {:d}).", chunk_size.size());
            return;
        }
        chunk_t = chunk_size[2];
        chunk_f = chunk_size[0];
        if (chunk_size[0] < 1 || chunk_size[1] < 1 || chunk_size[2] < 1) {
            FATAL_ERROR("Chunk dimensions need to be >= 1 (got ({:d}, {:d}, {:d}).", chunk_size[0],
                        chunk_size[1], chunk_size[2]);
            return;
        }
    }

    // Get the list of buffers that this stage should connect to
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

bool ensureOrdered::get_dataset_state(dset_id_t ds_id) {

    datasetManager& dm = datasetManager::instance();

    // Get the states synchronously.
    auto tstate_fut = std::async(&datasetManager::dataset_state<timeState>, &dm, ds_id);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);

    const timeState* tstate = tstate_fut.get();
    const freqState* fstate = fstate_fut.get();

    if (tstate == nullptr || fstate == nullptr) {
        ERROR("One of time or freq dataset states is null for dataset {}.", ds_id);
        return false;
    }

    auto times = tstate->get_times();
    auto freq_pairs = fstate->get_freqs();

    // construct map of times to axis index
    for (size_t i = 0; i < times.size(); i++) {
        time_map.insert({times.at(i), i});
    }
    // construct map of freq_ind to axis index
    for (size_t i = 0; i < freq_pairs.size(); i++) {
        freq_map.insert({freq_pairs.at(i).first, i});
    }

    return true;
}

void ensureOrdered::main_thread() {

    // The index to the current buffer frame
    frameID frame_id(in_buf);
    frameID output_frame_id(out_buf);
    // The index of the current frame relative to the first frame
    size_t output_ind = 0;

    // Frequency and time indices
    size_t fi, ti;
    time_ctype t;

    // The dataset ID we read from the frame
    dset_id_t ds_id;

    // Get axes from dataset state
    uint32_t first_ind = 0;
    while (true) {
        // Wait for a frame in the input buffer in order to get the dataset ID
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), first_ind)) == nullptr) {
            return;
        }
        auto frame = VisFrameView(in_buf, first_ind);
        if (frame.fpga_seq_length == 0) {
            INFO("Got empty frame ({:d}).", first_ind);
            first_ind++;
        } else {
            ds_id = frame.dataset_id;
            break;
        }
    }

    auto future_ds_state = std::async(&ensureOrdered::get_dataset_state, this, ds_id);
    if (!future_ds_state.get()) {
        FATAL_ERROR("Couldn't find ancestor of dataset {}. "
                    "Make sure there is a stage upstream in the config, that adds the dataset "
                    "states.\nExiting...",
                    ds_id);
        return;
    }

    // main loop:
    while (!stop_thread) {
        // Wait for a full frame in the input buffer
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }
        auto frame = VisFrameView(in_buf, frame_id);

        // Figure out the ordered index of this frame
        t = {std::get<0>(frame.time), ts_to_double(std::get<1>(frame.time))};
        ti = time_map.at(t);
        fi = freq_map.at(frame.freq_id);
        size_t ordered_ind = get_frame_ind(ti, fi);

        // Check if this is the index we are ready to send
        if (ordered_ind == output_ind) {
            // copy frame into output buffer
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                return;
            }
            auto output_frame =
                VisFrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);

            // release input frame
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

            // increment output index
            output_ind++;
        } else if (waiting.size() <= max_waiting) {
            INFO("Frame {:d} arrived out of order. Expected {:d}. Adding it to waiting buffer.",
                 ordered_ind, output_ind);
            // Add to waiting frames and move to next (without marking empty!)
            waiting.insert({ordered_ind, (int)frame_id});
            frame_id++;
        } else {
            FATAL_ERROR("Number of frames arriving out of order exceeded maximum buffer size.");
            return;
        }

        // Check if any of the waiting frames are ready
        auto ready = waiting.find(output_ind);
        while (ready != waiting.end()) {
            // remove this index from waiting map
            uint32_t waiting_id = ready->second;
            waiting.erase(ready);
            INFO("Frame {:d} is ready to be sent. Releasing buffer.", output_ind);
            // copy frame into output buffer
            auto past_frame = VisFrameView(in_buf, waiting_id);
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                return;
            }
            auto output_frame =
                VisFrameView::copy_frame(in_buf, waiting_id, out_buf, output_frame_id);
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);

            mark_frame_empty(in_buf, unique_name.c_str(), waiting_id);
            output_ind++;

            ready = waiting.find(output_ind);
        }
    }
}
