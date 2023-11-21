#include "bufferBadInputs.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"         // for allocate_new_metadata_object, mark_frame_full, register_pr...
#include "chimeMetadata.hpp"  // for set_rfi_num_bad_inputs
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG, ERROR
#include "visUtil.hpp"        // for parse_reorder_default

#include <algorithm>  // for max
#include <cstdint>    // for uint32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, function
#include <regex>      // for match_results<>::_Base_type
#include <tuple>      // for get

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferBadInputs);

bufferBadInputs::bufferBadInputs(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&bufferBadInputs::main_thread, this)) {

    uint32_t num_elements = config.get<uint32_t>(unique_name, "num_elements");
    input_mask_len = sizeof(uint8_t) * num_elements;

    auto input_reorder = parse_reorder_default(config, unique_name);
    input_remap = std::get<0>(input_reorder);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    out_buffer_ID = 0;
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {

    DEBUG("update_bad_inputs_callback(): Update to bad inputs list.");

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* host_mask = out_buf->wait_for_empty_frame(unique_name, out_buffer_ID);

    try {
        bad_inputs_cylinder = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list {:s}", e.what());
        return false;
    }

    // Reorder list
    bad_inputs_correlator.clear();
    for (auto element : bad_inputs_cylinder)
        bad_inputs_correlator.push_back(input_remap[element]);

    // Zero bad inputs mask
    for (uint32_t i = 0; i < input_mask_len; ++i) {
        host_mask[i] = 1;
    }

    // Add current bad input mask
    for (auto element : bad_inputs_correlator) {
        if (element < (int)input_mask_len && element >= 0) {
            host_mask[element] = 0;
        } else {
            ERROR("Got a bad input with invalid index");
            return false;
        }
    }

    // Create new metadata
    out_buf->allocate_new_metadata_object(out_buffer_ID);

    // Set no. of bad inputs in the metadata
    set_rfi_num_bad_inputs(out_buf, out_buffer_ID, bad_inputs_correlator.size());

    out_buf->mark_frame_full(unique_name, out_buffer_ID);

    DEBUG("update_bad_inputs_callback(): Bad inputs reordered and buffered.");

    // Increment frame ID
    out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_frames;

    return true;
}

void bufferBadInputs::main_thread() {
    // Listen for bad input list updates
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}
