#include "visRawReader.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for Hash, operator!=, operator<
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"    // for acqDatasetIdState, eigenvalueState, freqState, inputState
#include "errors.h"            // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"  // for DEBUG, INFO, FATAL_ERROR
#include "metadata.h"          // for metadataContainer
#include "version.h"           // for get_git_commit_hash
#include "visBuffer.hpp"       // for visFrameView, visMetadata
#include "visUtil.hpp"         // for freq_ctype, prod_ctype, rstack_ctype, stack_ctype, time_c...

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span
#include "json.hpp"     // for basic_json<>::object_t, json, basic_json, basic_json<>::v...

#include <algorithm>  // for fill, min, max
#include <atomic>     // for atomic_bool
#include <cstdint>    // for uint32_t, uint8_t
#include <cstring>    // for strerror, memcpy
#include <cxxabi.h>   // for __forced_unwind
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, O_RDONLY
#include <fstream>    // for ifstream, ios_base::failure, ios_base, basic_ios, basic_i...
#include <functional> // for _Bind_helper<>::type, bind, function
#include <future>
#include <memory>       // for allocator_traits<>::value_type
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error, invalid_argument
#include <sys/mman.h>   // for madvise, mmap, munmap, MADV_DONTNEED, MADV_WILLNEED, MAP_...
#include <sys/stat.h>   // for stat
#include <system_error> // for system_error
#include <time.h>       // for nanosleep, timespec
#include <tuple>        // for get
#include <unistd.h>     // for close, off_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(visRawReader);
REGISTER_KOTEKAN_STAGE(ensureOrdered);

visRawReader::visRawReader(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visRawReader::main_thread, this)) {


    filename = config.get<std::string>(unique_name, "infile");
    ring = config.get_default<bool>(unique_name, "ring", false);
    readahead_blocks = config.get<size_t>(unique_name, "readahead_blocks");
    max_read_rate = config.get_default<double>(unique_name, "max_read_rate", 0.0);
    sleep_time = config.get_default<float>(unique_name, "sleep_time", -1);
    update_dataset_id = config.get_default<bool>(unique_name, "update_dataset_id", true);
    use_comet = config.get_default<bool>(DS_UNIQUE_NAME, "use_dataset_broker", true);

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
            throw std::invalid_argument("visRawReader: config: Chunk size "
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
            fmt::format(fmt("visRawReader: Error reading from metadata file: {:s}"), md_filename));
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    if (metadata_file) // only read if no error
        metadata_file.read((char*)&packed_json[0], filesize);
    if (!metadata_file) // check if open and read successful
        throw std::ios_base::failure("visRawReader: Error reading from "
                                     "metadata file: "
                                     + md_filename);
    json _t = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    _metadata = _t["attributes"];
    _times = _t["index_map"]["time"].get<std::vector<time_ctype>>();
    auto freqs = _t["index_map"]["freq"].get<std::vector<freq_ctype>>();
    _inputs = _t["index_map"]["input"].get<std::vector<input_ctype>>();
    _prods = _t["index_map"]["prod"].get<std::vector<prod_ctype>>();
    _ev = _t["index_map"]["ev"].get<std::vector<uint32_t>>();
    if (_t.at("index_map").find("stack") != _t.at("index_map").end()) {
        _stack = _t.at("index_map").at("stack").get<std::vector<stack_ctype>>();
        _rstack = _t.at("reverse_map").at("stack").get<std::vector<rstack_ctype>>();
        _num_stack = _t.at("structure").at("num_stack").get<uint32_t>();
    }

    for (auto f : freqs) {
        // TODO: add freq IDs to raw file format instead of restoring them here
        // TODO: CHIME specific.
        uint32_t freq_id = 1024.0 / 400.0 * (800.0 - f.centre);
        DEBUG("restored freq_id for f_centre={:.2f} : {:d}", f.centre, freq_id);
        _freqs.push_back({freq_id, f});
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

    if (chunked) {
        // Special case if dimensions less than chunk size
        chunk_f = std::min(chunk_f, nfreq);
        chunk_t = std::min(chunk_t, ntime);

        // Number of elements in a chunked row
        row_size = chunk_t * nfreq;
    }

    // Check metadata is the correct size
    if (sizeof(visMetadata) != metadata_size) {
        std::string msg = fmt::format(fmt("Metadata in file {:s} is larger ({:d} bytes) than "
                                          "visMetadata ({:d} bytes)."),
                                      filename, metadata_size, sizeof(visMetadata));
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

    // Register a state for the time axis if using comet, or register the replacement dataset ID if using
    if (update_dataset_id) {

        datasetManager& dm = datasetManager::instance();
        tstate_id = dm.create_state<timeState>(_times).first;

        if (!use_comet) {
            // Add the states: metadata, time, prod, freq, input,
            // eigenvalue and stack.
            std::vector<state_id_t> states;
            if (!_stack.empty())
                states.push_back(dm.create_state<stackState>(_num_stack, std::move(_rstack)).first);
            states.push_back(dm.create_state<inputState>(_inputs).first);
            states.push_back(dm.create_state<eigenvalueState>(_ev).first);
            states.push_back(dm.create_state<freqState>(_freqs).first);
            states.push_back(dm.create_state<prodState>(_prods).first);
            states.push_back(tstate_id);
            states.push_back(
                dm.create_state<metadataState>(
                    _metadata.at("weight_type"),
                    _metadata.at("instrument_name"),
                    _metadata.at("git_version_tag")
                ).first
            );
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

    if (ring)
        ring_offset = 0;
}

visRawReader::~visRawReader() {
    if (munmap(mapped_file, ntime * nfreq * file_frame_size) == -1) {
        // Make sure kotekan is exiting...
        FATAL_ERROR("Failed to unmap file {:s}.data: {:s}.", filename, strerror(errno));
    }

    close(fd);
}

dset_id_t visRawReader::get_dataset_state(dset_id_t ds_id) {

    // See if we've already processed this ds_id...
    auto got_it = ds_in_file.find(ds_id);

    // ... if we have, just return the corresponding output
    if (got_it != ds_in_file.end())
        return got_it->second;

    dset_id_t new_id;

    if (!update_dataset_id || ds_id == dset_id_t::null) {
        new_id = ds_id;
    }
    else if (use_comet) {
        INFO("Registering new dataset with broker based on {}.", ds_id);
        datasetManager& dm = datasetManager::instance();
        new_id = dm.add_dataset(tstate_id, ds_id);
    }
    else {
        new_id = static_out_dset_id;
    }

    ds_in_file[ds_id] = new_id;
    return new_id;
}

void visRawReader::read_ahead(int ind) {

    off_t offset = position_map(ind) * file_frame_size;

    if (madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED) == -1)
        DEBUG("madvise failed: {:s}", strerror(errno));
}

int visRawReader::position_map(int ind) {
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

void visRawReader::main_thread() {

    double start_time, end_time;
    frameID frame_id(out_buf);
    uint8_t* frame;

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

        // Check first byte indicating empty frame
        if (*(mapped_file + file_ind * file_frame_size) != 0) {
            // Copy the metadata from the file
            std::memcpy(out_buf->metadata[frame_id]->metadata,
                        mapped_file + file_ind * file_frame_size + 1, metadata_size);

            // Copy the data from the file
            std::memcpy(frame, mapped_file + file_ind * file_frame_size + metadata_size + 1,
                        data_size);

        } else {
            // Create frame and set structural metadata
            size_t num_vis = _stack.size() > 0 ? _stack.size() : _prods.size();
            auto frame = visFrameView(out_buf, frame_id, _inputs.size(), num_vis, _ev.size());

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

            DEBUG("visRawReader: Reading empty frame: {:d}", frame_id);
        }

        // Set the dataset ID to the updated value
        dset_id_t& ds_id = ((visMetadata*)(out_buf->metadata[frame_id]->metadata))->dataset_id;
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

ensureOrdered::ensureOrdered(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ensureOrdered::main_thread, this)) {

    max_waiting = config.get_default<size_t>(unique_name, "max_waiting", 100);

    chunked = config.exists(unique_name, "chunk_size");
    if (chunked) {
        chunk_size = config.get<std::vector<int>>(unique_name, "chunk_size");
        if (chunk_size.size() != 3) {
            FATAL_ERROR("Chunk size needs exactly three elements (got {:d}).", chunk_size.size());
            return;
        }
        chunk_t = chunk_size[2];
        chunk_f = chunk_size[0];
        if (chunk_size[0] < 1 || chunk_size[1] < 1 || chunk_size[2] < 1) {
            FATAL_ERROR("Chunk dimensions need to be >= 1 (got ({:d}, {:d}, {:d}).", chunk_size[0],
                        chunk_size[1], chunk_size[2]);
            return;
        }
    }

    // Get the list of buffers that this stage should connect to
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

bool ensureOrdered::get_dataset_state(dset_id_t ds_id) {

    datasetManager& dm = datasetManager::instance();

    // Get the states synchronously.
    auto tstate_fut = std::async(&datasetManager::dataset_state<timeState>, &dm, ds_id);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);

    const timeState* tstate = tstate_fut.get();
    const freqState* fstate = fstate_fut.get();

    if (tstate == nullptr || fstate == nullptr) {
        ERROR("One of time or freq dataset states is null for dataset {}.", ds_id);
        return false;
    }

    auto times = tstate->get_times();
    auto freq_pairs = fstate->get_freqs();

    // construct map of times to axis index
    for (size_t i = 0; i < times.size(); i++) {
        time_map.insert({times.at(i), i});
    }
    // construct map of freq_ind to axis index
    for (size_t i = 0; i < freq_pairs.size(); i++) {
        freq_map.insert({freq_pairs.at(i).first, i});
    }

    return true;
}

void ensureOrdered::main_thread() {

    // The index to the current buffer frame
    frameID frame_id(in_buf);
    frameID output_frame_id(out_buf);
    // The index of the current frame relative to the first frame
    size_t output_ind = 0;

    // Frequency and time indices
    size_t fi, ti;
    time_ctype t;

    // The dataset ID we read from the frame
    dset_id_t ds_id;

    // Get axes from dataset state
    uint32_t first_ind = 0;
    while (true) {
        // Wait for a frame in the input buffer in order to get the dataset ID
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), first_ind)) == nullptr) {
            return;
        }
        auto frame = visFrameView(in_buf, first_ind);
        if (frame.fpga_seq_length == 0) {
            INFO("Got empty frame ({:d}).", first_ind);
            first_ind++;
        } else {
            ds_id = frame.dataset_id;
            break;
        }
    }

    auto future_ds_state = std::async(&ensureOrdered::get_dataset_state, this, ds_id);
    if (!future_ds_state.get()) {
        FATAL_ERROR("Couldn't find ancestor of dataset {}. "
                    "Make sure there is a stage upstream in the config, that adds the dataset "
                    "states.\nExiting...",
                    ds_id);
        return;
    }

    // main loop:
    while (!stop_thread) {
        // Wait for a full frame in the input buffer
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }
        auto frame = visFrameView(in_buf, frame_id);

        // Figure out the ordered index of this frame
        t = {std::get<0>(frame.time), ts_to_double(std::get<1>(frame.time))};
        ti = time_map.at(t);
        fi = freq_map.at(frame.freq_id);
        size_t ordered_ind = get_frame_ind(ti, fi);

        // Check if this is the index we are ready to send
        if (ordered_ind == output_ind) {
            // copy frame into output buffer
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                return;
            }
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame = visFrameView(out_buf, output_frame_id, frame);
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);

            // release input frame
            mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

            // increment output index
            output_ind++;
        } else if (waiting.size() <= max_waiting) {
            INFO("Frame {:d} arrived out of order. Expected {:d}. Adding it to waiting buffer.",
                 ordered_ind, output_ind);
            // Add to waiting frames and move to next (without marking empty!)
            waiting.insert({ordered_ind, (int)frame_id});
            frame_id++;
        } else {
            FATAL_ERROR("Number of frames arriving out of order exceeded maximum buffer size.");
            return;
        }

        // Check if any of the waiting frames are ready
        auto ready = waiting.find(output_ind);
        while (ready != waiting.end()) {
            // remove this index from waiting map
            uint32_t waiting_id = ready->second;
            waiting.erase(ready);
            INFO("Frame {:d} is ready to be sent. Releasing buffer.", output_ind);
            // copy frame into output buffer
            auto past_frame = visFrameView(in_buf, waiting_id);
            if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
                return;
            }
            allocate_new_metadata_object(out_buf, output_frame_id);
            auto output_frame = visFrameView(out_buf, output_frame_id, past_frame);
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);

            mark_frame_empty(in_buf, unique_name.c_str(), waiting_id);
            output_ind++;

            ready = waiting.find(output_ind);
        }
    }
}
