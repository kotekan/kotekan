#include "VisRawReader.hpp"

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"    // for eigenvalueState, inputState, prodState, stackState
#include "kotekanLogging.hpp"  // for DEBUG, WARN
#include "visBuffer.hpp"       // for VisFrameView, VisMetadata
#include "visUtil.hpp"         // for prod_ctype, rstack_ctype, stack_ctype, input_ctype, frameID

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span
#include "json.hpp"     // for basic_json<>::object_t, basic_json, json, basic_json<>::v...

#include <algorithm>   // for fill, max
#include <cstdint>     // for uint32_t
#include <stddef.h>    // for size_t
#include <stdexcept>   // for runtime_error
#include <time.h>      // for timespec
#include <tuple>       // for make_tuple, tuple
#include <type_traits> // for __decay_and_strip<>::__type
#include <utility>     // for pair, move

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(VisRawReader);

VisRawReader::VisRawReader(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    RawReader(config, unique_name, buffer_container) {

    // Extract data specific indices
    _inputs = metadata_json["index_map"]["input"].get<std::vector<input_ctype>>();
    _prods = metadata_json["index_map"]["prod"].get<std::vector<prod_ctype>>();
    _ev = metadata_json["index_map"]["ev"].get<std::vector<uint32_t>>();
    if (metadata_json.at("index_map").find("stack") != metadata_json.at("index_map").end()) {
        _stack = metadata_json.at("index_map").at("stack").get<std::vector<stack_ctype>>();
        _rstack = metadata_json.at("reverse_map").at("stack").get<std::vector<rstack_ctype>>();
        _num_stack = metadata_json.at("structure").at("num_stack").get<uint32_t>();
    }

    // Check metadata is the correct size
    if (sizeof(VisMetadata) != metadata_size) {
        std::string msg = fmt::format(fmt("Metadata in file {:s} is larger ({:d} bytes) than "
                                          "VisMetadata ({:d} bytes)."),
                                      filename, metadata_size, sizeof(VisMetadata));
        throw std::runtime_error(msg);
    }

    // Register a state for the time axis if using comet, or register the replacement dataset ID if
    // using
    if (update_dataset_id) {

        datasetManager& dm = datasetManager::instance();

        if (!use_comet) {
            // Add data specific states
            if (!_stack.empty())
                states.push_back(dm.create_state<stackState>(_num_stack, std::move(_rstack)).first);
            states.push_back(dm.create_state<inputState>(_inputs).first);
            states.push_back(dm.create_state<eigenvalueState>(_ev).first);
            states.push_back(dm.create_state<prodState>(_prods).first);

            // register it as root dataset
            static_out_dset_id = dm.add_dataset(states);

            WARN("Updating the dataset IDs without comet is not recommended "
                 "as it will not preserve dataset ID changes.");
        }
    }
}

VisRawReader::~VisRawReader() {}

void VisRawReader::create_empty_frame(frameID frame_id) {

    // Create frame and set structural metadata
    size_t num_vis = _stack.size() > 0 ? _stack.size() : _prods.size();

    auto frame =
        VisFrameView::create_frame_view(out_buf, frame_id, _inputs.size(), num_vis, _ev.size());

    frame.zero_frame();

    DEBUG("VisRawReader: Reading empty frame: {:d}", frame_id);
}
