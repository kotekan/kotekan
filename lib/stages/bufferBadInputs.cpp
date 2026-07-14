#include "bufferBadInputs.hpp"

#include "Config.hpp"         // for Config
#include "N2Util.hpp"         // for frameID
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"      // for Telescope, station_id_t
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG, ERROR

#include <cstring>    // for memset
#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <json.hpp>   // for json
#include <mutex>      // for lock_guard, mutex
#include <stdint.h>   // for uint8_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferBadInputs);


bufferBadInputs::bufferBadInputs(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&bufferBadInputs::main_thread, this)) {

    num_elements = config.get<size_t>(unique_name, "num_elements");

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Construct the cylinder -> beamformer reorder table.
    // reorder[beamformer_idx] = cylinder_idx;
    reorder.resize(num_elements);

    // initialize the mask (1 == good)
    input_mask = std::vector<uint8_t>(num_elements, 1u);

    const Telescope& tel = Telescope::instance();

    for (size_t beamformer_idx = 0; beamformer_idx < num_elements; ++beamformer_idx) {
        station_id_t st_id =
            tel.element_index_to_station_id(beamformer_idx, ElementOrder::CHIMEBeamformer);
        reorder.at(beamformer_idx) =
            tel.station_id_to_element_index(st_id, ElementOrder::CHIMECylinder);
    }

    // Listen for bad input list updates
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {
    DEBUG("update_bad_inputs_callback(): Update to bad inputs list.");

    // hold lock for the entire update
    std::lock_guard<std::mutex> lock(mtx);

    // Reset the mask (1 == good)
    std::fill(input_mask.begin(), input_mask.end(), 1u);

    bool all_valid = true;

    try {
        bad_inputs = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list:\n{:s}", e.what());
        return false;
    }

    // Add current bad input to the mask
    for (int element : bad_inputs) {
        if (element < (int)num_elements && element >= 0) {
            input_mask[reorder[element]] = 0;
        } else {
            ERROR("Got input with invalid index: {:d}", element);
            all_valid = false;
        }
    }

    DEBUG("update_bad_inputs_callback(): Bad inputs reordered and buffered.");

    return all_valid;
}

void bufferBadInputs::main_thread() {
    N2::frameID frame_id(out_buf);

    while (!stop_thread) {
        // get an output frame
        uint8_t* out_frame = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);

        // Copy from the permanent buffer
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::copy_n(input_mask.begin(), num_elements, out_frame);
        }

        // Set metadata and release
        out_buf->allocate_new_metadata_object(frame_id);
        get_chord_metadata(out_buf, frame_id)->set_rfi_num_bad_inputs(num_bad_inputs);
        out_buf->mark_frame_full(unique_name, frame_id);
        frame_id++;
    }
}
