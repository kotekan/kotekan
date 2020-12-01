/*****************************************
@file
@brief Base class for reading raw files.
- RawReader : public kotekan::Stage
*****************************************/
#ifndef _RAW_READER_HPP
#define _RAW_READER_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t
#include "visUtil.hpp"        // for freq_ctype (ptr only), input_ctype, prod_ctype, rstack_ctype

#include "json.hpp" // for json

#include <map>      // for map
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint8_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

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
 * This will divide the file up into time-frequency chunks of set size and
 * stream out the frames with time as the *fastest* index. The dataset ID
 * will be restored from the dataset broker if `use_comet` is set. Otherwise
 * a new dataset will be created and the original ID stored in the frames
 * will be lost.
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
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
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

    // Get dataset ID
    virtual dset_id_t& get_dataset_id(frameID frame_id) = 0;

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
