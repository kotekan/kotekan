#include "bufferBadInputs.hpp"

#include <stdint.h>            // for uint8_t
#include <json.hpp>            // for json
#include <exception>           // for exception
#include <functional>          // for bind, function, _1
#include <memory>              // for __shared_ptr_access, shared_ptr
#include <array>               // for array
#include <cstring>             // for memset

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "chordMetadata.hpp"   // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"   // for configUpdater
#include "kotekanLogging.hpp"  // for DEBUG, ERROR
#include "visUtil.hpp"         // for get_cylinder_to_beamformer_reorder_table
#include "fmt.hpp"             // for compile_string_to_view

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferBadInputs);


bufferBadInputs::bufferBadInputs(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&bufferBadInputs::main_thread, this)),
    // Required since `frame_id` is used outside of the main loop
    frame_id(get_buffer("out_buf")) {

    num_elements = config.get<size_t>(unique_name, "num_elements");

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Construct the cylinder -> beamformer reorder table.
    // reorder[beamformer_idx] = cylinder_idx;
    reorder.reserve(num_elements);

    const Telescope& tel = Telescope::instance();

    for(size_t beamformer_idx = 0; beamformer_idx < num_elements; ++beamformer_idx) {
        station_id_t st_id = tel.element_index_to_station_id(beamformer_idx, ElementOrder::CHIMEBeamformer);
        reorder.at(beamformer_idx) = tel.station_id_to_element_index(st_id, ElementOrder::CHIMECylinder);
    }
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {
    DEBUG("update_bad_inputs_callback(): Update to bad inputs list.");

    // Get the next output frame
    uint8_t* host_mask = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
    // Reset the mask (1 == good)
    std::memset(host_mask, 1U, num_elements);

    bool all_good = true;

    try {
        bad_inputs = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list:\n{:s}", e.what());
        return false;
    }

    // Add current bad input to the mask
    for (int element : bad_inputs) {
        if (element < (int)num_elements && element >= 0) {
            host_mask[reorder[element]] = 0;
        } else {
            ERROR("Got input with invalid index: {:d}", element);
            all_good = false;
        }
    }

    // Create new metadata
    out_buf->allocate_new_metadata_object(frame_id);
    // Set no. of bad inputs in the metadata
    get_chord_metadata(out_buf, frame_id)->set_rfi_num_bad_inputs(bad_inputs.size());

    out_buf->mark_frame_full(unique_name, frame_id);

    DEBUG("update_bad_inputs_callback(): Bad inputs reordered and buffered.");

    frame_id++;

    return all_good;
}

void bufferBadInputs::main_thread() {
    // Listen for bad input list updates
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}
