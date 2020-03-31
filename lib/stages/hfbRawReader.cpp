#include "hfbRawReader.hpp"

#include "Config.hpp"          // for Config
#include "HfbFrameView.hpp"    // for HfbFrameView, hfbMetadata
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"    // for acqDatasetIdState, eigenvalueState, freqState, inputState
#include "errors.h"            // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"  // for DEBUG, INFO, FATAL_ERROR
#include "metadata.h"          // for metadataContainer
#include "version.h"           // for get_git_commit_hash
#include "visUtil.hpp"         // for freq_ctype, prod_ctype, rstack_ctype, stack_ctype, time_c...

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span
#include "json.hpp"     // for basic_json<>::object_t, json, basic_json, basic_json<>::v...

#include <algorithm>  // for fill, min, max
#include <atomic>     // for atomic_bool
#include <cstdint>    // for uint32_t, uint8_t
#include <cstring>    // for strerror, memcpy
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, O_RDONLY
#include <fstream>    // for ifstream, ios_base::failure, ios_base, basic_ios, basic_i...
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error, invalid_argument
#include <sys/mman.h> // for madvise, mmap, munmap, MADV_DONTNEED, MADV_WILLNEED, MAP_...
#include <sys/stat.h> // for stat
#include <time.h>     // for nanosleep, timespec
#include <unistd.h>   // for close, off_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(hfbRawReader);

hfbRawReader::hfbRawReader(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&hfbRawReader::main_thread, this)) {

    filename = config.get<std::string>(unique_name, "infile");
    readahead_blocks = config.get<size_t>(unique_name, "readahead_blocks");
    max_read_rate = config.get_default<double>(unique_name, "max_read_rate", 0.0);
    sleep_time = config.get_default<float>(unique_name, "sleep_time", -1);

    chunked = config.exists(unique_name, "chunk_size");
    if (chunked) {
        chunk_size = config.get<std::vector<int>>(unique_name, "chunk_size");
        if (chunk_size.size() != 3)
            throw std::invalid_argument("Chunk size needs exactly three "
                                        "elements (has "
                                        + std::to_string(chunk_size.size()) + ").");
        chunk_t = chunk_size[2];
        chunk_f = chunk_size[0];
        if (chunk_size[0] < 1 || chunk_size[1] < 1 || chunk_size[2] < 1)
            throw std::invalid_argument("hfbRawReader: config: Chunk size "
                                        "needs to be greater or equal to (1,1,1) (is ("
                                        + std::to_string(chunk_size[0]) + ","
                                        + std::to_string(chunk_size[1]) + ","
                                        + std::to_string(chunk_size[2]) + ")).");
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
            fmt::format(fmt("hfbRawReader: Error reading from metadata file: {:s}"), md_filename));
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    if (metadata_file) // only read if no error
        metadata_file.read((char*)&packed_json[0], filesize);
    if (!metadata_file) // check if open and read successful
        throw std::ios_base::failure("hfbRawReader: Error reading from "
                                     "metadata file: "
                                     + md_filename);
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    _metadata = _t["attributes"];
    _times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    auto freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    auto beams = _t["index_map"]["beam"].get<std::vector<uint32_t>>();
    auto subfreqs = _t["index_map"]["subfreq"].get<std::vector<freq_ctype>>();
    if (_t.at("index_map").find("stack") != _t.at("index_map").end()) {
        _stack = _t.at("index_map").at("stack").get<std::vector<stack_ctype>>();
        _rstack = _t.at("reverse_map").at("stack").get<std::vector<rstack_ctype>>();
        _num_stack = _t.at("structure").at("num_stack").get<uint32_t>();
    }

    // for (auto f : freqs) {
    //    // TODO: add freq IDs to raw file format instead of restoring them here
    //    // TODO: CHIME specific.
    //    uint32_t freq_id = 1024.0 / 400.0 * (800.0 - f.centre);
    //    DEBUG("restored freq_id for f_centre={:.2f} : {:d}", f.centre, freq_id);
    //    _freqs.push_back({freq_id, f});
    //}

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
        row_size = (chunk_f * chunk_t) * (nfreq / chunk_f) + (nfreq % chunk_f) * chunk_t;
    }

    // Check metadata is the correct size
    if (sizeof(hfbMetadata) != metadata_size) {
        std::string msg = fmt::format(fmt("Metadata in file {:s} is larger ({:d} bytes) than "
                                          "hfbMetadata ({:d} bytes)."),
                                      filename, metadata_size, sizeof(hfbMetadata));
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

hfbRawReader::~hfbRawReader() {
    if (munmap(mapped_file, ntime * nfreq * file_frame_size) == -1) {
        // Make sure kotekan is exiting...
        FATAL_ERROR("Failed to unmap file {:s}.data: {:s}.", filename, strerror(errno));
    }

    close(fd);
}

void hfbRawReader::change_dataset_state(dset_id_t ds_id) {
    datasetManager& dm = datasetManager::instance();

    // Add the states: acq_dataset_id, metadata, time, prod, freq, input,
    // eigenvalue and stack.
    std::vector<state_id_t> states;
    if (!_stack.empty())
        states.push_back(dm.create_state<stackState>(_num_stack, std::move(_rstack)).first);
    // states.push_back(dm.create_state<inputState>(_inputs).first);
    // states.push_back(dm.create_state<eigenvalueState>(_ev).first);
    // states.push_back(dm.create_state<freqState>(_freqs).first);
    // states.push_back(dm.create_state<prodState>(_prods).first);
    states.push_back(dm.create_state<timeState>(_times).first);
    // states.push_back(dm.create_state<metadataState>(_metadata.at("instrument_name"),
    //                                                _metadata.at("git_version_tag"))
    //                     .first);
    states.push_back(dm.create_state<acqDatasetIdState>(ds_id).first);
    // register it as root dataset
    _dataset_id = dm.add_dataset(states);
}

void hfbRawReader::read_ahead(int ind) {

    off_t offset = position_map(ind) * file_frame_size;

    if (madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED) == -1)
        DEBUG("madvise failed: {:s}", strerror(errno));
}

int hfbRawReader::position_map(int ind) {
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
        // time is fastest varying
        int fi = ci * chunk_f + ((ind % row_size) % (t_width * f_width)) / t_width;
        int ti = ri * chunk_t + ((ind % row_size) % (t_width * f_width)) % t_width;

        return ti * nfreq + fi;
    } else {
        return ind;
    }
}

void hfbRawReader::main_thread() {

    double start_time, end_time;
    size_t frame_id = 0;
    uint8_t* frame;

    size_t ind = 0, read_ind = 0, file_ind;

    size_t nframe = nfreq * ntime;

    // Calculate the minimum time we should take to read the data to satisfy the
    // rate limiting
    double min_read_time =
        (max_read_rate > 0 ? file_frame_size / (max_read_rate * 1024 * 1024) : 0.0);
    DEBUG("Minimum read time per frame {}s", min_read_time);

    // Initial readahead for frames
    for (read_ind = 0; read_ind < readahead_blocks; read_ind++) {
        read_ahead(read_ind);
    }

    while (!stop_thread && ind < nframe) {

        // Get the start time of the loop for rate limiting
        start_time = current_time();

        // Wait for an empty frame in the output buffer
        if ((frame = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id)) == nullptr) {
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

        INFO("file_ind: {}, file_frame_size: {}", file_ind, file_frame_size);

        // Check first byte indicating empty frame
        if (*(mapped_file + file_ind * file_frame_size) != 0) {
            // Copy the metadata from the file
            std::memcpy(out_buf->metadata[frame_id]->metadata,
                        mapped_file + file_ind * file_frame_size + 1, metadata_size);

            // Copy the data from the file
            std::memcpy(frame, mapped_file + file_ind * file_frame_size + metadata_size + 1,
                        data_size);
        } else {
            // Set metadata if file contained an empty frame
            //((hfbMetadata*)(out_buf->metadata[frame_id]->metadata))->num_prod = _prods.size();
            //((hfbMetadata*)(out_buf->metadata[frame_id]->metadata))->num_ev = _ev.size();
            //((hfbMetadata*)(out_buf->metadata[frame_id]->metadata))->num_elements =
            //_inputs.size();
            // Fill data with zeros
            // size_t num_vis = _stack.size() > 0 ? _stack.size() : _prods.size();
            // JSW: Do we want num_prod to be _prods.size() or _stack.size()?
            auto frame = HfbFrameView(out_buf, frame_id);
            std::fill(frame.hfb.begin(), frame.hfb.end(), 0.0);
            frame.freq_id = 0;
            DEBUG("hfbRawReader: Reading empty frame: {:d}", frame_id);
        }

        // Read first frame to get true dataset ID
        if (ind == 0) {
            change_dataset_state(
                ((hfbMetadata*)(out_buf->metadata[frame_id]->metadata))->dataset_id);
        }

        // Set the dataset ID to one associated with this file
        ((hfbMetadata*)(out_buf->metadata[frame_id]->metadata))->dataset_id = _dataset_id;

        // Try and clear out the cached data as we don't need it again
        if (madvise(mapped_file + file_ind * file_frame_size, file_frame_size, MADV_DONTNEED) == -1)
            DEBUG("madvise failed: {:s}", strerror(errno));

        // Release the frame and advance all the counters
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % out_buf->num_frames;
        read_ind++;
        ind++;

        // Get the end time for the loop and sleep for long enough to satisfy
        // the max rate
        end_time = current_time();
        double sleep_time = min_read_time - (end_time - start_time);
        DEBUG("Sleep time {}", sleep_time);
        if (sleep_time > 0) {
            auto ts = double_to_ts(sleep_time);
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
