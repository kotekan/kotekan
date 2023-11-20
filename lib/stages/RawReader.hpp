/*****************************************
@file
@brief Base class for reading raw files.
- RawReader : public kotekan::Stage
*****************************************/
#ifndef _RAW_READER_HPP
#define _RAW_READER_HPP

#include "Config.hpp"
#include "Hash.hpp"      // for Hash, operator<, operator==
#include "Stage.hpp"     // for Stage
#include "Telescope.hpp" // for Telescope
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t
#include "datasetState.hpp"   // for freqState, timeState, metadataState
#include "errors.h"           // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp" // for INFO, FATAL_ERROR, DEBUG, WARN, ERROR
#include "metadata.h"         // for metadataContainer
#include "version.h"          // for get_git_commit_hash
#include "visUtil.hpp"        // for freq_ctype (ptr only), input_ctype, prod_ctype, rstack_ctype

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json

#include <cstring>    // for strerror, memcpy
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, posix_fadvise, O_RDONLY, POSIX_FADV_DONTNEED
#include <fstream>    // for ifstream, ios_base::failure, ios_base, basic_ios, basic_i...
#include <functional> // for _Bind_helper<>::type, bind, function
#include <map>        // for map
#include <regex>      // for match_results<>::_Base_type
#include <stddef.h>   // for size_t
#include <stdexcept>  // for runtime_error, invalid_argument, out_of_range
#include <stdint.h>   // for uint32_t, uint8_t
#include <string>     // for string
#include <sys/mman.h> // for madvise, mmap, munmap, MADV_DONTNEED, MADV_WILLNEED, MAP_...
#include <sys/stat.h> // for stat
#include <time.h>     // for nanosleep, timespec
#include <unistd.h>   // for close, off_t
#include <utility>    // for pair
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

/**
 * @class RawReader
 * @brief Generic class to read and stream a raw file.
 *
 * All classes which inherit from this should provide the following API:
 *
 * create_empty_frame(frameID frame_id);
 * get_dataset_id(frameID frame_id);
 *
 * This stage will divide the file up into time-frequency chunks of set size and
 * stream out the frames with time as the *fastest* index. The dataset ID
 * will be restored from the dataset broker if `use_comet` is set. Otherwise
 * a new dataset will be created and the original ID stored in the frames
 * will be lost.
 *
 * The chunking strategy aids the downstream Transpose stage that writes to a HDF5
 * file with a chunked layout. The file writing is most efficient when writing entire
 * chunks on exact chunk boundaries. The reason for using chunking in the first place
 * is to enable compression and try to optimise for certain IO patterns
 * (i.e. read a few frequencies for many times).
 *
 * @par Buffers
 * @buffer out_buf The data read from the raw file.
 *         @buffer_format Buffer structured
 *         @buffer_metadata Metadata
 *
 * @conf    readahead_blocks       Int. Number of blocks to advise OS to read ahead
 *                                 of current read.
 * @conf    chunk_size             Array of [int, int, int]. Read chunk size (freq,
 *                                 prod, time). If not specified will read file
 *                                 contiguously.
 * @conf    infile                 String. Path to the (data-meta-pair of) files to
 *                                 read (e.g. "/path/to/0000_000", without .data or
 *                                 .meta).
 * @conf    max_read_rate          Float. Maximum read rate for the process in MB/s.
 *                                 If the value is zero (default), then no rate
 *                                 limiting is applied.
 * @conf    sleep_time             Float. After the data is read pause this long in
 *                                 seconds before sending shutdown. If < 0, never
 *                                 send a shutdown signal. Default is -1.
 * @conf    update_dataset_id      Bool. Update the dataset ID with information about the
                                   file, for example which time samples does it contain.
 * @conf    use_dataset_broker     Bool. Restore dataset ID from dataset broker (i.e. comet).
 *                                 Should be disabled only for testing. Default is true.
 * @conf    use_local_dataset_man  Bool. Instead of using comet, register metadata on top
 *                                 of dataset ID in the file. Should only be used for
 *                                 testing or if original dataset IDs can be lost.
 *                                 Default is False.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche, James Willis
 */
template<typename T>
class RawReader : public kotekan::Stage {

public:
    /// default constructor
    RawReader(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

    ~RawReader();

    /// Main loop over buffer frames
    void main_thread() override;

    /**
     * @brief Get the times in the file.
     **/
    const std::vector<time_ctype>& times() {
        return _times;
    }

    /**
     * @brief Get the frequencies in the file.
     **/
    const std::vector<std::pair<uint32_t, freq_ctype>>& freqs() {
        return _freqs;
    }

    /**
     * @brief Get the metadata saved into the file.
     **/
    const nlohmann::json& metadata() {
        return _metadata;
    }

protected:
    // Dataset states constructed from metadata
    std::vector<state_id_t> states;

    // The input file
    std::string filename;

    // Metadata size
    size_t metadata_size;

    // whether to update the dataset ID with info about the file
    bool update_dataset_id;

    // whether to use comet to track dataset IDs
    bool use_comet;

    // Dataset ID to assign to output frames if not using comet
    dset_id_t static_out_dset_id;

    // Metadata file in json format
    json metadata_json;

    Buffer* out_buf;

private:
    // Create an empty frame
    virtual void create_empty_frame(frameID frame_id) = 0;

    /**
     * @brief Get the new dataset ID.
     *
     * If not using change the ID this just returns its input ID. If using the
     * broker, this will simply append a timeState to the current state.
     * Otherwise, we just use a static dataset_id constructed from the file
     * metadata.
     *
     * @param  ds_id  The ID of the read frame.
     * @returns       The replacement ID
     */
    dset_id_t get_dataset_state(dset_id_t ds_id);

    /**
     * @brief Read the next frame.
     *
     * The exact frame read is dependent on whether the reads are time ordered
     * or not.
     *
     * @param ind The frame index to read.
     **/
    void read_ahead(int ind);

    /**
     * @brief Map the index into a frame position in the file.
     *
     * The exact frame read dependent on whether the reads are time ordered
     * or not.
     *
     * @param ind The frame index.
     * @returns The frame index into the file.
     **/
    int position_map(int ind);

    // The metadata
    nlohmann::json _metadata;
    std::vector<time_ctype> _times;
    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;

    // whether to read in chunks
    bool chunked;

    // whether to use the local dataset manager
    bool local_dm;

    // Read chunk size (freq, prod, time)
    std::vector<int> chunk_size;
    // time chunk size
    size_t chunk_t;
    // freq chunk size
    size_t chunk_f;

    // Number of elements in a chunked row
    size_t row_size;

    // the input file
    int fd;
    uint8_t* mapped_file;

    size_t file_frame_size, data_size, nfreq, ntime;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;

    // the dataset state for the time axis
    state_id_t tstate_id;

    // Map input to output dataset IDs for quick access
    std::map<dset_id_t, dset_id_t> ds_in_file;

    // The read rate
    double max_read_rate;

    // Sleep time after reading
    double sleep_time;
};

template<typename T>
RawReader<T>::RawReader(Config& config, const std::string& unique_name,
                        bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RawReader::main_thread, this)) {

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
        if (chunk_size.size() < 3)
            throw std::invalid_argument("Chunk size needs at least three "
                                        "elements (has "
                                        + std::to_string(chunk_size.size()) + ").");
        chunk_t = chunk_size[2];
        chunk_f = chunk_size[0];
        if (chunk_size[0] < 1 || chunk_size[1] < 1 || chunk_size[2] < 1)
            throw std::invalid_argument("RawReader: config: Chunk size "
                                        "needs to be greater or equal to (1,1,1) (is ("
                                        + std::to_string(chunk_size[0]) + ","
                                        + std::to_string(chunk_size[1]) + ","
                                        + std::to_string(chunk_size[2]) + ")).");
    }

    DEBUG("Chunked: {}, chunk_t: {}, chunk_f: {}", chunked, chunk_t, chunk_f);

    // Get the list of buffers that this stage should connect to
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Read the metadata
    std::string md_filename = (filename + ".meta");
    INFO("Reading metadata file: {:s}", md_filename);
    struct stat st;
    if (stat(md_filename.c_str(), &st) == -1)
        throw std::ios_base::failure(
            fmt::format(fmt("RawReader: Error reading from metadata file: {:s}"), md_filename));
    size_t filesize = st.st_size;
    std::vector<uint8_t> packed_json(filesize);

    std::ifstream metadata_file(md_filename, std::ios::binary);
    if (metadata_file) // only read if no error
        metadata_file.read((char*)&packed_json[0], filesize);
    if (!metadata_file) // check if open and read successful
        throw std::ios_base::failure("RawReader: Error reading from "
                                     "metadata file: "
                                     + md_filename);
    metadata_json = json::from_msgpack(packed_json);
    metadata_file.close();

    // Extract the attributes and index maps
    _metadata = metadata_json["attributes"];
    _times = metadata_json["index_map"]["time"].template get<std::vector<time_ctype>>();
    auto freqs = metadata_json["index_map"]["freq"].template get<std::vector<freq_ctype>>();

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
    if (_metadata.at("git_version_tag").template get<std::string>()
        != std::string(get_git_commit_hash()))
        INFO("Git version tags don't match: dataset in file {:s} has tag {:s}, while the local git "
             "version tag is {:s}",
             filename, _metadata.at("git_version_tag").template get<std::string>(),
             get_git_commit_hash());

    // Extract the structure
    file_frame_size = metadata_json["structure"]["frame_size"].template get<size_t>();
    metadata_size = metadata_json["structure"]["metadata_size"].template get<size_t>();
    data_size = metadata_json["structure"]["data_size"].template get<size_t>();
    nfreq = metadata_json["structure"]["nfreq"].template get<size_t>();
    ntime = metadata_json["structure"]["ntime"].template get<size_t>();

    DEBUG("Metadata fields. frame_size: {}, metadata_size: {}, data_size: {}, nfreq: {}, ntime: {}",
          file_frame_size, metadata_size, data_size, nfreq, ntime);

    if (chunked) {
        // Special case if dimensions less than chunk size
        chunk_f = std::min(chunk_f, nfreq);
        chunk_t = std::min(chunk_t, ntime);

        // Number of elements in a chunked row
        row_size = chunk_t * nfreq;
    }

    // Check that buffer is large enough
    if (out_buf->frame_size < data_size) {
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
            // Add the states: metadata, time, freq
            states.push_back(tstate_id);
            states.push_back(dm.create_state<freqState>(_freqs).first);
            states.push_back(dm.create_state<metadataState>(_metadata.at("weight_type"),
                                                            _metadata.at("instrument_name"),
                                                            _metadata.at("git_version_tag"))
                                 .first);
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

template<typename T>
RawReader<T>::~RawReader() {
    if (munmap(mapped_file, ntime * nfreq * file_frame_size) == -1) {
        // Make sure kotekan is exiting...
        FATAL_ERROR("Failed to unmap file {:s}.data: {:s}.", filename, strerror(errno));
    }

    close(fd);
}

template<typename T>
void RawReader<T>::main_thread() {

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
            // Create empty frame and set structural metadata
            create_empty_frame(frame_id);
        }

        // Set the dataset ID to the updated value
        auto frame = T(out_buf, frame_id);
        dset_id_t& ds_id = frame.dataset_id;
        ds_id = get_dataset_state(ds_id);

        // Try and clear out the cached data from the memory map as we don't need it again
        if (madvise(mapped_file + file_ind * file_frame_size, file_frame_size, MADV_DONTNEED) == -1)
            WARN("madvise failed: {:s}", strerror(errno));
#ifdef __linux__
        // Try and clear out the cached data from the page cache as we don't need it again
        // NOTE: unless we do this in addition to the above madvise the kernel will try and keep as
        // much of the file in the page cache as possible and it will fill all the available memory
        if (posix_fadvise(fd, file_ind * file_frame_size, file_frame_size, POSIX_FADV_DONTNEED)
            == -1)
            WARN("fadvise failed: {:s}", strerror(errno));
#endif

        // Release the frame and advance all the counters
        mark_frame_full(out_buf, unique_name.c_str(), frame_id++);
        read_ind++;
        ind++;

        // Get the end time for the loop and sleep for long enough to satisfy
        // the max rate
        end_time = current_time();
        double sleep_time_this_frame = min_read_time - (end_time - start_time);
        DEBUG2("Sleep time {}", sleep_time_this_frame);
        if (sleep_time_this_frame > 0) {
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

template<typename T>
dset_id_t RawReader<T>::get_dataset_state(dset_id_t ds_id) {

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

template<typename T>
void RawReader<T>::read_ahead(int ind) {

    off_t offset = position_map(ind) * file_frame_size;

    if (madvise(mapped_file + offset, file_frame_size, MADV_WILLNEED) == -1)
        DEBUG("madvise failed: {:s}", strerror(errno));
}

template<typename T>
int RawReader<T>::position_map(int ind) {
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


/**
 * @class ensureOrdered
 * @brief Check frames are coming through in order and reorder them otherwise.
 *        Not used presently.
 */
class ensureOrdered : public kotekan::Stage {

public:
    ensureOrdered(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

    ~ensureOrdered() = default;

    /// Main loop over buffer frames
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    // Map of buffer frames waiting for their turn
    std::map<size_t, size_t> waiting;
    size_t max_waiting;

    // time and frequency axes
    std::map<time_ctype, size_t> time_map;
    std::map<size_t, size_t> freq_map;
    size_t ntime;
    size_t nfreq;

    // HDF5 chunk size
    std::vector<int> chunk_size;
    // size of time dimension of chunk
    size_t chunk_t;
    // size of frequency dimension of chunk
    size_t chunk_f;
    bool chunked;

    bool get_dataset_state(dset_id_t ds_id);

    // map from time and freq index to frame index
    // RawReader reads chunks with frequency as fastest varying index
    // within a chunk, frequency is also the fastest varying index.
    inline size_t get_frame_ind(size_t ti, size_t fi) {
        size_t ind = 0;
        // chunk row and column
        size_t row = ti / chunk_t;
        size_t col = fi / chunk_f;
        // special dimension at array edges
        size_t this_chunk_t = chunk_t ? row * chunk_t + chunk_t < ntime : ntime - row * chunk_t;
        size_t this_chunk_f = chunk_f ? col * chunk_f + chunk_f < nfreq : nfreq - col * chunk_f;
        // number of frames in previous rows
        ind += nfreq * chunk_t * row;
        // number of frames in chunks in this row
        ind += (this_chunk_t * chunk_f) * col;
        // within a chunk, frequency is fastest varying
        ind += (ti % chunk_t) * this_chunk_f + (fi % chunk_f);

        return ind;
    };
};

#endif
