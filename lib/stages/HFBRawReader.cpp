#include "HFBRawReader.hpp"

#include "Config.hpp"          // for Config
#include "HFBFrameView.hpp"    // for HFBFrameView
#include "HFBMetadata.hpp"     // for HFBMetadata
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"    // for beamState, subfreqState
#include "kotekanLogging.hpp"  // for DEBUG, WARN
#include "visUtil.hpp"         // for frameID

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span
#include "json.hpp"     // for basic_json<>::object_t, json, basic_json, basic_json<>::v...

#include <algorithm> // for fill, max
#include <cstdint>   // for uint32_t
#include <stdexcept> // for runtime_error
#include <time.h>    // for timespec
#include <utility>   // for pair

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(HFBRawReader);

HFBRawReader::HFBRawReader(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    RawReader(config, unique_name, buffer_container) {

    // Extract data specific indices
    _beams = metadata_json["index_map"]["beam"].get<std::vector<uint32_t>>();
    _subfreqs = metadata_json["index_map"]["subfreq"].get<std::vector<uint32_t>>();

    // Check metadata is the correct size
    if (sizeof(HFBMetadata) != metadata_size) {
        std::string msg = fmt::format(fmt("Metadata in file {:s} is larger ({:d} bytes) than "
                                          "HFBMetadata ({:d} bytes)."),
                                      filename, metadata_size, sizeof(HFBMetadata));
        throw std::runtime_error(msg);
    }

    // Register a state for the time axis if using comet, or register the replacement dataset ID if
    // using
    if (update_dataset_id) {

        datasetManager& dm = datasetManager::instance();

        if (!use_comet) {
            // Add data specific states
            states.push_back(dm.create_state<beamState>(_beams).first);
            states.push_back(dm.create_state<subfreqState>(_subfreqs).first);

            // register it as root dataset
            static_out_dset_id = dm.add_dataset(states);

            WARN("Updating the dataset IDs without comet is not recommended "
                 "as it will not preserve dataset ID changes.");
        }
    }
}

HFBRawReader::~HFBRawReader() {}

void HFBRawReader::create_empty_frame(frameID frame_id) {

    // Create frame and set structural metadata
    auto frame =
        HFBFrameView::create_frame_view(out_buf, frame_id, _beams.size(), _subfreqs.size());

    frame.zero_frame();

    DEBUG("HFBRawReader: Reading empty frame: {:d}", frame_id);
}
