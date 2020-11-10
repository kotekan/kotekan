#include "HFBRawReader.hpp"

#include "Config.hpp"       // for Config
#include "Hash.hpp"         // for Hash, operator<, operator==
#include "HFBFrameView.hpp" // for HFBFrameView, HFBMetadata
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

REGISTER_KOTEKAN_STAGE(HFBRawReader);

HFBRawReader::HFBRawReader(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&HFBRawReader::main_thread, this)) {

    filename = config.get<std::string>(unique_name, "infile");
    readahead_blocks = config.get<size_t>(unique_name, "readahead_blocks");
    max_read_rate = config.get_default<double>(unique_name, "max_read_rate", 0.0);
    sleep_time = config.get_default<float>(unique_name, "sleep_time", -1);
    update_dataset_id = config.get_default<bool>(unique_name, "update_dataset_id", true);
    use_comet = config.get_default<bool>(DS_UNIQUE_NAME, "use_dataset_broker", true);
    local_dm = config.get_default<bool>(unique_name, "use_local_dataset_man", false);
    if (local_dm && use_comet)
        FATAL_ERROR("Cannot use local dataset manager and dataset broker together.")

    chunked = config.exists(unique_name, "chunk_size");
    if (chunked) {
        chunk_size = config.get<std::vector<int>>(unique_name, "chunk_size");
        if (chunk_size.size() != 5)
            throw std::invalid_argument("Chunk size needs exactly five "
                                        "elements (has "
                                        + std::to_string(chunk_size.size()) + ").");
        chunk_t = chunk_size[2];
        chunk_f = chunk_size[0];
        if (chunk_size[0] < 1 || chunk_size[1] < 1 || chunk_size[2] < 1 || chunk_size[3] < 1 || chunk_size[4] < 1)
            throw std::invalid_argument("HFBRawReader: config: Chunk size "
                                        "needs to be greater or equal to (1,1,1,1,1) (is ("
                                        + std::to_string(chunk_size[0]) + ","
                                        + std::to_string(chunk_size[1]) + ","
                                        + std::to_string(chunk_size[2]) + ","
                                        + std::to_string(chunk_size[3]) + ","
                                        + std::to_string(chunk_size[4]) + ")).");
    }

    // Get the list of buffers that this stage should connect to
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Read the metadata
    std::string md_filename = (filename + ".meta");
    INFO("Reading metadata file: {:s}", md_filename);
    struct stat st;
    if (stat(md_filename.c_str(), &st) == -1)
        throw std::ios_base::failure(
            fmt::format(fmt("HFBRawReader: Error reading from metadata file: {:s}"), md_filename));
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    if (metadata_file) // only read if no error
        metadata_file.read((char*)&packed_json[0], filesize);
    if (!metadata_file) // check if open and read successful
        throw std::ios_base::failure("HFBRawReader: Error reading from "
                                     "metadata file: "
                                     + md_filename);
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    _metadata = _t["attributes"];
    _times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    auto freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    _beams = _t["index_map"]["beam"].get<std::vector<uint32_t>>();
    _subfreqs = _t["index_map"]["subfreq"].get<std::vector<uint32_t>>();

    // Match frequencies to IDs in the Telescope...
    auto& tel = Telescope::instance();
    std::map<double, uint32_t> inv_freq_map;

    // ... first construct a map of central frequencies to IDs known by the
    // telescope object
    for (uint32_t id = 0; id < tel.num_freq(); id++) {
        inv_freq_map[tel.to_freq(id)] = id;
    }

    // ... then use this to match the central frequencies given in the file
    for (auto f : freqs) {

        auto it = inv_freq_map.find(f.centre);

        if (it == inv_freq_map.end()) {
            FATAL_ERROR("Could not match a frequency ID to channel in file at {} MHz. "
                        "Check you are specifying the correct telescope.",
                        f.centre);
            return;
        }

        DEBUG("restored freq_id for f_centre={:.2f} : {:d}", f.centre, it->second);
        _freqs.push_back({it->second, f});
    }

    // check git version tag
    // TODO: enforce that they match if build type == "release"?
    if (_metadata.at("git_version_tag").get<std::string>() != std::string(get_git_commit_hash()))
        INFO("Git version tags don't match: dataset in file {:s} has tag {:s}, while the local git "
             "version tag is {:s}",
             filename, _metadata.at("git_version_tag").get<std::string>(), get_git_commit_hash());

    // Extract the structure
    file_frame_size = _t["structure"]["frame_size"].get<size_t>();
    metadata_size = _t["structure"]["metadata_size"].get<size_t>();
    data_size = _t["structure"]["data_size"].get<size_t>();
    nfreq = _t["structure"]["nfreq"].get<size_t>();
    ntime = _t["structure"]["ntime"].get<size_t>();
    nbeam = _t["structure"]["num_beams"].get<size_t>();
    nsubfreq = _t["structure"]["num_subfreq"].get<size_t>();

    INFO("Metadata fields. frame_size: {}, metadata_size: {}, data_size: {}, nfreq: {}, ntime: {}, "
        "nbeam: {}, nsubfreq: {}",
        file_frame_size, metadata_size, data_size, nfreq, ntime, nbeam, nsubfreq);

    if (chunked) {
        // Special case if dimensions less than chunk size
        chunk_f = std::min(chunk_f, nfreq);
        chunk_t = std::min(chunk_t, ntime);

        // Number of elements in a chunked row
        row_size = chunk_t * nfreq;
    }

    // Check metadata is the correct size
    if (sizeof(HFBMetadata) != metadata_size) {
        std::string msg = fmt::format(fmt("Metadata in file {:s} is larger ({:d} bytes) than "
                                          "HFBMetadata ({:d} bytes)."),
                                      filename, metadata_size, sizeof(HFBMetadata));
        throw std::runtime_error(msg);
    }

    // Check that buffer is large enough
    if ((unsigned int)(out_buf->frame_size) < data_size || out_buf->frame_size < 0) {
        std::string msg =
            fmt::format(fmt("Data in file {:s} is larger ({:d} bytes) than buffer size "
                            "({:d} bytes)."),
                        filename, data_size, out_buf->frame_size);
        throw std::runtime_error(msg);
    }

    // Register a state for the time axis if using comet, or register the replacement dataset ID if
    // using
    if (update_dataset_id) {

        datasetManager& dm = datasetManager::instance();
        tstate_id = dm.create_state<timeState>(_times).first;

        if (!use_comet) {
            // Add the states: metadata, time, prod, freq, input,
            // eigenvalue and stack.
            states.push_back(dm.create_state<freqState>(_freqs).first);
            states.push_back(dm.create_state<beamState>(_beams).first);
            states.push_back(dm.create_state<subfreqState>(_subfreqs).first);
            states.push_back(tstate_id);
            states.push_back(dm.create_state<metadataState>(_metadata.at("weight_type"),
                                                            _metadata.at("instrument_name"),
                                                            _metadata.at("git_version_tag"))
                                 .first);
            // register it as root dataset
            static_out_dset_id = dm.add_dataset(states);

            WARN("Updating the dataset IDs without comet is not recommended "
                 "as it will not preserve dataset ID changes.");
        }
    }

    // Open up the data file and mmap it
    INFO("Opening data file: {:s}.data", filename);
    if ((fd = open((filename + ".data").c_str(), O_RDONLY)) == -1) {
        throw std::runtime_error(
            fmt::format(fmt("Failed to open file {:s}.data: {:s}."), filename, strerror(errno)));
    }
    mapped_file =
        (uint8_t*)mmap(nullptr, ntime * nfreq * file_frame_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_file == MAP_FAILED)
        throw std::runtime_error(fmt::format(fmt("Failed to map file {:s}.data to memory: {:s}."),
                                             filename, strerror(errno)));
}

HFBRawReader::~HFBRawReader() {
    if (munmap(mapped_file, ntime * nfreq * file_frame_size) == -1) {
        // Make sure kotekan is exiting...
        FATAL_ERROR("Failed to unmap file {:s}.data: {:s}.", filename, strerror(errno));
    }

    close(fd);
}

dset_id_t HFBRawReader::get_dataset_state(dset_id_t ds_id) {

    // See if we've already processed this ds_id...
    auto got_it = ds_in_file.find(ds_id);

    // ... if we have, just return the corresponding output
    if (got_it != ds_in_file.end())
        return got_it->second;

    dset_id_t new_id;

    if (!update_dataset_id || ds_id == dset_id_t::null) {
        new_id = ds_id;
    } else if (local_dm) {
        INFO("Registering new dataset with local DM based on {}.", ds_id);
        datasetManager& dm = datasetManager::instance();
        new_id = dm.add_dataset(states, ds_id);
    } else if (use_comet) {
        INFO("Registering new dataset with broker based on {}.", ds_id);
        datasetManager& dm = datasetManager::instance();
        new_id = dm.add_dataset(tstate_id, ds_id);
    } else {
        new_id = static_out_dset_id;
    }

    ds_in_file[ds_id] = new_id;
    return new_id;
}

void HFBRawReader::read_ahead(int ind) {

    off_t offset = position_map(ind) * file_frame_size;

    if (madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED) == -1)
        DEBUG("madvise failed: {:s}", strerror(errno));
}

int HFBRawReader::position_map(int ind) {
    if (chunked) {
        // chunked row index
        int ri = ind / row_size;
        // Special case at edges of time*freq array
        int t_width = std::min(ntime - ri * chunk_t, chunk_t);
        // chunked column index
        int ci = (ind % row_size) / (t_width * chunk_f);
        // edges case
        int f_width = std::min(nfreq - ci * chunk_f, chunk_f);
        // time and frequency indices
        // frequency is fastest varying
        int fi = ci * chunk_f + ((ind % row_size) % (t_width * chunk_f)) % f_width;
        int ti = ri * chunk_t + ((ind % row_size) % (t_width * chunk_f)) / f_width;

        return ti * nfreq + fi;
    }
    return ind;
}

void HFBRawReader::main_thread() {

    double start_time, end_time;
    frameID frame_id(out_buf);
    uint8_t* frame_ptr;

    size_t ind = 0, read_ind = 0, file_ind;

    size_t nframe = nfreq * ntime;

    // Calculate the minimum time we should take to read the data to satisfy the
    // rate limiting
    double min_read_time =
        (max_read_rate > 0 ? file_frame_size / (max_read_rate * 1024 * 1024) : 0.0);
    DEBUG("Minimum read time per frame {}s", min_read_time);

    readahead_blocks = std::min(nframe, readahead_blocks);
    // Initial readahead for frames
    for (read_ind = 0; read_ind < readahead_blocks; read_ind++) {
        read_ahead(read_ind);
    }

    while (!stop_thread && ind < nframe) {

        // Get the start time of the loop for rate limiting
        start_time = current_time();

        // Wait for an empty frame in the output buffer
        if ((frame_ptr = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }

        // Issue the read ahead request
        if (read_ind < (ntime * nfreq)) {
            read_ahead(read_ind);
        }

        // Get the index into the file
        file_ind = position_map(ind);

        // Allocate the metadata space
        allocate_new_metadata_object(out_buf, frame_id);

        DEBUG("file_ind: {}, file_frame_size: {}", file_ind, file_frame_size);

        // Check first byte indicating empty frame
        if (*(mapped_file + file_ind * file_frame_size) != 0) {
            // Copy the metadata from the file
            std::memcpy(out_buf->metadata[frame_id]->metadata,
                        mapped_file + file_ind * file_frame_size + 1, metadata_size);

            // Copy the data from the file
            std::memcpy(frame_ptr, mapped_file + file_ind * file_frame_size + metadata_size + 1,
                        data_size);
            //DEBUG("hfbRawReader: Reading full frame: {:d}, data: {}", frame_id, frame[0]);
        } else {
            // Create frame and set structural metadata
            auto frame = HFBFrameView::create_frame_view(out_buf, frame_id, _beams.size(), _subfreqs.size());

            // Fill data with zeros
            std::fill(frame.hfb.begin(), frame.hfb.end(), 0.0);
            std::fill(frame.weight.begin(), frame.weight.end(), 0.0);

            // Set non-structural metadata
            frame.freq_id = 0;
            frame.dataset_id = dset_id_t::null;
            frame.time = timespec{0, 0};

            // mark frame as empty by ensuring this is 0
            frame.fpga_seq_length = 0;
            frame.fpga_seq_total = 0;

            DEBUG("HFBRawReader: Reading empty frame: {:d}", frame_id);
        }

        // Set the dataset ID to the updated value
        dset_id_t& ds_id = ((HFBMetadata*)(out_buf->metadata[frame_id]->metadata))->dataset_id;
        ds_id = get_dataset_state(ds_id);

        // Try and clear out the cached data as we don't need it again
        if (madvise(mapped_file + file_ind * file_frame_size, file_frame_size, MADV_DONTNEED) == -1)
            DEBUG("madvise failed: {:s}", strerror(errno));

        // Release the frame and advance all the counters
        mark_frame_full(out_buf, unique_name.c_str(), frame_id++);
        read_ind++;
        ind++;

        // Get the end time for the loop and sleep for long enough to satisfy
        // the max rate
        end_time = current_time();
        double sleep_time_this_frame = min_read_time - (end_time - start_time);
        DEBUG("Sleep time {}", sleep_time_this_frame);
        if (sleep_time > 0) {
            auto ts = double_to_ts(sleep_time_this_frame);
            nanosleep(&ts, nullptr);
        }
    }

    if (sleep_time > 0) {
        INFO("Read all data. Sleeping and then exiting kotekan...");
        timespec ts = double_to_ts(sleep_time);
        nanosleep(&ts, nullptr);
        exit_kotekan(ReturnCode::CLEAN_EXIT);
    } else {
        INFO("Read all data. Exiting stage, but keeping kotekan alive.");
    }
}
