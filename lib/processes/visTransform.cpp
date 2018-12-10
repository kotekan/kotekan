#include "visTransform.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <tuple>

#include "gsl-lite.hpp"

#include "buffer.h"
#include "bufferContainer.hpp"
#include "chimeMetadata.h"
#include "datasetState.hpp"
#include "errors.h"
#include "metadata.h"
#include "processFactory.hpp"
#include "version.h"
#include "visBuffer.hpp"
#include "visUtil.hpp"



REGISTER_KOTEKAN_PROCESS(visTransform);

visTransform::visTransform(Config& config,
                           const std::string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTransform::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get<size_t>(unique_name, "num_elements");
    block_size = config.get<size_t>(unique_name, "block_size");
    num_eigenvectors =  config.get<size_t>(unique_name, "num_ev");

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> input_buffer_names =
        config.get<std::vector<std::string>>(unique_name, "in_bufs");

    // Fetch the input buffers, register them, and store them in our buffer vector
    for(auto name : input_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_consumer(buf, unique_name.c_str());
        in_bufs.push_back({buf, 0});
    }

    // Setup the output vector
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the indices for reordering
    auto input_reorder = parse_reorder_default(config, unique_name);

    // Get the indices for reordering
    input_remap = std::get<0>(input_reorder);

    // Get everything we need for registering dataset states

    // --> get metadata
    _instrument_name = config.get_default<std::string>(
                unique_name, "instrument_name", "chime");

    std::vector<uint32_t> freq_ids;

    // Get the frequency IDs that are on this stream, check the config or just
    // assume all CHIME channels
    // TODO: CHIME specific
    if (config.exists(unique_name, "freq_ids")) {
        freq_ids = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");
    }
    else {
        freq_ids.resize(1024);
        std::iota(std::begin(freq_ids), std::end(freq_ids), 0);
    }

    // Create the frequency specification
    // TODO: CHIME specific
    std::transform(std::begin(freq_ids), std::end(freq_ids), std::back_inserter(_freqs),
                   [] (uint32_t id) -> std::pair<uint32_t, freq_ctype> {
                       return {id, {800.0 - 400.0 / 1024 * id, 400.0 / 1024}};
                   });

    // The input specification from the config
    _inputs = std::get<1>(input_reorder);

    size_t num_elements = _inputs.size();

    // Create the product specification
    _prods.reserve(num_elements * (num_elements+1) / 2);
    for(uint16_t i = 0; i < num_elements; i++) {
        for(uint16_t j = i; j < num_elements; j++) {
            _prods.push_back({i, j});
        }
    }

    // Ask the broker for a dataset ID (blocking)
    _ds_id_out = change_dataset_state();
}

void visTransform::main_thread() {

    uint8_t* frame = nullptr;
    Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;

    while (!stop_thread) {

        // This is where all the main set of work happens. Iterate over the
        // available buffers, wait for data to appear and transform into
        // visBuffer style data
        for(auto& buffer_pair : in_bufs) {
            std::tie(buf, frame_id) = buffer_pair;

            // Calculate the timeout
            auto timeout = double_to_ts(current_time() + 0.1);

            // Find the next available buffer
            int status = wait_for_full_frame_timeout(buf, unique_name.c_str(),
                                                     frame_id, timeout);
            if(status == 1) continue;  // Timed out, try next buffer
            if(status == -1) break;  // Got shutdown signal

            INFO("Got full buffer %s with frame_id=%i", buf->buffer_name, frame_id);

            frame = buf->frames[frame_id];

            // Wait for the buffer to be filled with data
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }
            allocate_new_metadata_object(out_buf, output_frame_id);

            auto output_frame = visFrameView(out_buf, output_frame_id,
                                             num_elements, num_eigenvectors);

            // Copy over the metadata
            output_frame.fill_chime_metadata((const chimeMetadata *)buf->metadata[frame_id]->metadata);

            // Copy the visibility data into a proper triangle and write into
            // the file
            copy_vis_triangle((int32_t *)frame, input_remap, block_size,
                              num_elements, output_frame.vis);

            // Fill other datasets with reasonable values
            std::fill(output_frame.weight.begin(), output_frame.weight.end(), 1.0);
            std::fill(output_frame.flags.begin(), output_frame.flags.end(), 1.0);
            std::fill(output_frame.evec.begin(), output_frame.evec.end(), 0.0);
            std::fill(output_frame.eval.begin(), output_frame.eval.end(), 0.0);
            output_frame.erms = 0;
            std::fill(output_frame.gain.begin(), output_frame.gain.end(), 1.0);

            // Mark the buffers and move on
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

    }

}

dset_id_t visTransform::change_dataset_state() {

    // weight calculation is hardcoded, so is the weight type name
    const std::string weight_type = "none";
    const std::string git_tag = get_git_commit_hash();

    // create all the states
    state_uptr freq_state = std::make_unique<freqState>(_freqs);
    state_uptr input_state = std::make_unique<inputState>(
        _inputs, std::move(freq_state));
    state_uptr prod_state = std::make_unique<prodState>(
        _prods, std::move(input_state));
    state_uptr mstate = std::make_unique<metadataState>(weight_type,
                                                        _instrument_name,
                                                        git_tag,
                                                        std::move(prod_state));

    // register them with the datasetManager
    datasetManager& dm = datasetManager::instance();
    state_id_t mstate_id = dm.add_state(std::move(mstate)).first;
    //register root dataset
    return dm.add_dataset(mstate_id);
}