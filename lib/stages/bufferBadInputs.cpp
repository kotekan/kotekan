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
    INFO("update_bad_inputs_callback(): Received update to bad inputs list.");

    // hold lock for the entire update
    std::lock_guard<std::mutex> lock(mtx);

    try {
        bad_inputs = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list:\n{:s}", e.what());
        return false;
    }

    // validate all inputs before changing the mask
    for (int element : bad_inputs) {
        if (element >= (int)num_elements || element < 0) {
            ERROR("Received input with invalid index: {:d}", element);
            return false;
        }
    }

    // Reset the mask (1 == good)
    std::fill(input_mask.begin(), input_mask.end(), 1u);

    // now update the mask
    for (int element : bad_inputs) {
        input_mask[reorder[element]] = 0;
    }
    num_bad_inputs = bad_inputs.size();

    DEBUG("update_bad_inputs_callback(): Bad inputs reordered and buffered.");

    return true;
}

void bufferBadInputs::main_thread() {
    N2::frameID frame_id(out_buf);
    size_t nbad; // copy num bad inputs for access outside lock

    while (!stop_thread) {
        // get an output frame
        uint8_t* out_frame = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (out_frame == nullptr) {
            return;
        }

        // Copy from the permanent buffer
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::copy_n(input_mask.begin(), num_elements, out_frame);
            nbad = num_bad_inputs;
        }

        // Set metadata and release
        out_buf->allocate_new_metadata_object(frame_id);
        get_chord_metadata(out_buf, frame_id)->set_rfi_num_bad_inputs(nbad);
        out_buf->mark_frame_full(unique_name, frame_id);
        frame_id++;
    }
}
