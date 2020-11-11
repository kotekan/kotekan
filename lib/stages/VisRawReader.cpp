#include "VisRawReader.hpp"

#include "Config.hpp"       // for Config
#include "Hash.hpp"         // for Hash, operator<, operator==
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.h"            // for allocate_new_metadata_object, mark_frame_full, wait_for_e...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, dset_id_t, datasetManager, DS_UNIQUE_NAME
#include "datasetState.hpp"    // for freqState, timeState, eigenvalueState, inputState, metada...
#include "errors.h"            // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"  // for INFO, DEBUG, FATAL_ERROR, ERROR, WARN
#include "metadata.h"          // for metadataContainer
#include "version.h"           // for get_git_commit_hash
#include "visBuffer.hpp"       // for VisFrameView, VisMetadata
#include "visUtil.hpp"         // for time_ctype, frameID, freq_ctype, prod_ctype, rstack_ctype

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span
#include "json.hpp"     // for basic_json<>::object_t, json, basic_json, basic_json<>::v...

#include <algorithm>    // for fill, min, max
#include <atomic>       // for atomic_bool
#include <cstdint>      // for uint32_t, uint8_t
#include <cstring>      // for strerror, memcpy
#include <cxxabi.h>     // for __forced_unwind
#include <errno.h>      // for errno
#include <exception>    // for exception
#include <fcntl.h>      // for open, O_RDONLY
#include <fstream>      // for ifstream, ios_base::failure, ios_base, basic_ios, basic_i...
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for async, future
#include <memory>       // for allocator_traits<>::value_type
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error, invalid_argument, out_of_range
#include <sys/mman.h>   // for madvise, mmap, munmap, MADV_DONTNEED, MADV_WILLNEED, MAP_...
#include <sys/stat.h>   // for stat
#include <system_error> // for system_error
#include <time.h>       // for nanosleep, timespec
#include <tuple>        // for get, make_tuple, tuple
#include <type_traits>  // for __decay_and_strip<>::__type
#include <unistd.h>     // for close, off_t

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

    auto frame = VisFrameView::create_frame_view(out_buf, frame_id, _inputs.size(), num_vis,
                                                 _ev.size());

    // Fill data with zeros
    std::fill(frame.vis.begin(), frame.vis.end(), 0.0);
    std::fill(frame.weight.begin(), frame.weight.end(), 0.0);
    std::fill(frame.eval.begin(), frame.eval.end(), 0.0);
    std::fill(frame.evec.begin(), frame.evec.end(), 0.0);
    std::fill(frame.gain.begin(), frame.gain.end(), 0.0);
    std::fill(frame.flags.begin(), frame.flags.end(), 0.0);
    frame.erms = 0;

    // Set non-structural metadata
    frame.freq_id = 0;
    frame.dataset_id = dset_id_t::null;
    frame.time = std::make_tuple(0, timespec{0, 0});

    // mark frame as empty by ensuring this is 0
    frame.fpga_seq_length = 0;
    frame.fpga_seq_total = 0;

    DEBUG("VisRawReader: Reading empty frame: {:d}", frame_id);
}

dset_id_t& VisRawReader::get_dataset_id(frameID frame_id) {
    auto frame = VisFrameView(out_buf, frame_id);
    return frame.dataset_id;
}
