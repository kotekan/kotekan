#include "visRawReader.hpp"

#include "StageFactory.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "metadata.h"
#include "version.h"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include "fmt.hpp"
#include "gsl-lite.hpp"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <regex>
#include <signal.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(visRawReader);

visRawReader::visRawReader(Config& config, const string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visRawReader::main_thread, this)) {

    filename = config.get<std::string>(unique_name, "infile");
    readahead_blocks = config.get<size_t>(unique_name, "readahead_blocks");

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
    INFO("Reading metadata file: %s", md_filename.c_str());
    struct stat st;
    if (stat(md_filename.c_str(), &st) == -1)
        throw std::ios_base::failure("visRawReader: Error reading from "
                                     "metadata file: "
                                     + md_filename);
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
        DEBUG("restored freq_id for f_centre=%.2f : %d", f.centre, freq_id);
        _freqs.push_back({freq_id, f});
    }

    // check git version tag
    // TODO: enforce that they match if build type == "release"?
    if (_metadata.at("git_version_tag").get<std::string>() != std::string(get_git_commit_hash()))
        INFO("Git version tags don't match: dataset in file %s has tag %s, "
             "while the local git version tag is %s",
             filename.c_str(), _metadata.at("git_version_tag").get<std::string>().c_str(),
             get_git_commit_hash());

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
        row_size = (chunk_f * chunk_t) * (nfreq / chunk_f) + (nfreq % chunk_f) * chunk_t;
    }

    // Check metadata is the correct size
    if (sizeof(visMetadata) != metadata_size) {
        std::string msg = fmt::format("Metadata in file {} is larger "
                                      "({} bytes) than visMetadata ({} bytes).",
                                      filename, metadata_size, sizeof(visMetadata));
        throw std::runtime_error(msg);
    }

    // Check that buffer is large enough
    if ((unsigned int)(out_buf->frame_size) < data_size || out_buf->frame_size < 0) {
        std::string msg =
            fmt::format("Data in file {} is larger ({} bytes) than buffer size ({} bytes).",
                        filename, data_size, out_buf->frame_size);
        throw std::runtime_error(msg);
    }

    // Open up the data file and mmap it
    INFO("Opening data file: %s", (filename + ".data").c_str());
    if ((fd = open((filename + ".data").c_str(), O_RDONLY)) == -1) {
        throw std::runtime_error(
            fmt::format("Failed to open file {}: {}.", filename + ".data", strerror(errno)));
    }
    mapped_file =
        (uint8_t*)mmap(NULL, ntime * nfreq * file_frame_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_file == MAP_FAILED)
        throw std::runtime_error(fmt::format("Failed to map file {} to memory: {}.",
                                             filename + ".data", strerror(errno)));

    // tell the dataset manager and get a dataset ID for the data coming from
    // this file
    change_dataset_state();
}

visRawReader::~visRawReader() {
    if (munmap(mapped_file, ntime * nfreq * file_frame_size) == -1) {
        ERROR(fmt::format("Failed to unmap file {}: {}.", filename + ".data", strerror(errno))
                  .c_str());
        // Make sure kotekan is exiting...
        raise(SIGINT);
    }

    close(fd);
}

void visRawReader::change_dataset_state() {
    datasetManager& dm = datasetManager::instance();

    // Add the states: metadata, time, prod, freq, input, eigenvalue and stack.
    state_uptr sstate = nullptr;
    if (!_stack.empty())
        sstate = std::make_unique<stackState>(_num_stack, std::move(_rstack));
    state_uptr istate = std::make_unique<inputState>(_inputs, std::move(sstate));
    state_uptr evstate = std::make_unique<eigenvalueState>(_ev, std::move(istate));
    state_uptr fstate = std::make_unique<freqState>(_freqs, std::move(evstate));
    state_uptr pstate = std::make_unique<prodState>(_prods, std::move(fstate));
    state_uptr tstate = std::make_unique<timeState>(_times, std::move(pstate));

    state_id_t mstate_id =
        dm.add_state(std::make_unique<metadataState>(
                         _metadata.at("weight_type"), _metadata.at("instrument_name"),
                         _metadata.at("git_version_tag"), std::move(tstate)))
            .first;

    // register it as root dataset
    _dataset_id = dm.add_dataset(mstate_id);
}

void visRawReader::read_ahead(int ind) {

    off_t offset = position_map(ind) * file_frame_size;

    if (madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED) == -1)
        DEBUG("madvise failed: %s", strerror(errno));
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
        // time is fastest varying
        int fi = ci * chunk_f + ((ind % row_size) % (t_width * f_width)) / t_width;
        int ti = ri * chunk_t + ((ind % row_size) % (t_width * f_width)) % t_width;

        return ti * nfreq + fi;
    } else {
        return ind;
    }
}

void visRawReader::main_thread() {

    size_t frame_id = 0;
    uint8_t* frame;

    size_t ind = 0, read_ind = 0, file_ind;

    size_t nframe = nfreq * ntime;

    // Initial readahead for frames
    for (read_ind = 0; read_ind < readahead_blocks; read_ind++) {
        read_ahead(read_ind);
    }

    while (!stop_thread && ind < nframe) {

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
            // Set metadata if file contained an empty frame
            ((visMetadata*)(out_buf->metadata[frame_id]->metadata))->num_prod = _prods.size();
            ((visMetadata*)(out_buf->metadata[frame_id]->metadata))->num_ev = _ev.size();
            ((visMetadata*)(out_buf->metadata[frame_id]->metadata))->num_elements = _inputs.size();
            // Fill data with zeros
            size_t num_vis = _stack.size() > 0 ? _stack.size() : _prods.size();
            auto frame = visFrameView(out_buf, frame_id, _inputs.size(), num_vis, _ev.size());
            std::fill(frame.vis.begin(), frame.vis.end(), 0.0);
            std::fill(frame.weight.begin(), frame.weight.end(), 0.0);
            std::fill(frame.eval.begin(), frame.eval.end(), 0.0);
            std::fill(frame.evec.begin(), frame.evec.end(), 0.0);
            std::fill(frame.gain.begin(), frame.gain.end(), 0.0);
            std::fill(frame.flags.begin(), frame.flags.end(), 0.0);
            frame.freq_id = 0;
            frame.erms = 0;
            DEBUG("visRawReader: Reading empty frame: %d", frame_id);
        }

        // Set the dataset ID
        ((visMetadata*)(out_buf->metadata[frame_id]->metadata))->dataset_id = _dataset_id;

        // Try and clear out the cached data as we don't need it again
        if (madvise(mapped_file + file_ind * file_frame_size, file_frame_size, MADV_DONTNEED) == -1)
            DEBUG("madvise failed: %s", strerror(errno));

        // Release the frame and advance all the counters
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % out_buf->num_frames;
        read_ind++;
        ind++;
    }
}
