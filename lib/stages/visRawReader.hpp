/*****************************************
@file
@brief Read visFileRaw data.
- visRawReader : public kotekan::Stage
*****************************************/
#ifndef _VIS_RAW_READER_HPP
#define _VIS_RAW_READER_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp"

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

/**
 * @class visRawReader
 * @brief Read and stream a raw visibility file.
 *
 * This will divide the file up into time-frequency chunks of set size and
 * stream out the frames with time as the *fastest* index. The dataset ID
 * will be restored from the dataset broker if `use_comet` is set. Otherwise
 * a new dataset will be created and the original ID stored in the frames
 * will be lost.
 *
 * @par Buffers
 * @buffer out_buf The data read from the raw file.
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf    readahead_blocks    Int. Number of blocks to advise OS to read ahead
 *                              of current read.
 * @conf    chunk_size          Array of [int, int, int]. Read chunk size (freq,
 *                              prod, time). If not specified will read file
 *                              contiguously.
 * @conf    infile              String. Path to the (data-meta-pair of) files to
 *                              read (e.g. "/path/to/0000_000", without .data or
 *                              .meta).
 * @conf    max_read_rate       Float. Maximum read rate for the process in MB/s.
 *                              If the value is zero (default), then no rate
 *                              limiting is applied.
 * @conf    sleep_time          Float. After the data is read pause this long in
 *                              seconds before sending shutdown. If < 0, never
 *                              send a shutdown signal. Default is -1.
 * @conf    use_dataset_broker  Bool. Restore dataset ID from dataset broker (i.e. comet).
 *                              Should be disabled only for testing. Default is true.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class visRawReader : public kotekan::Stage {

public:
    /// default constructor
    visRawReader(kotekan::Config& config, const string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    ~visRawReader();

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
     * @brief Get the products in the file.
     **/
    const std::vector<prod_ctype>& prods() {
        return _prods;
    }

    /**
     * @brief Get the stack in the file.
     **/
    const std::vector<stack_ctype>& stack() {
        return _stack;
    }

    /**
     * @brief Get the inputs in the file.
     **/
    const std::vector<input_ctype>& inputs() {
        return _inputs;
    }

    /**
     * @brief Get the ev axis in the file.
     **/
    const std::vector<uint32_t>& ev() {
        return _ev;
    }

    /**
     * @brief Get the metadata saved into the file.
     **/
    const json& metadata() {
        return _metadata;
    }

private:
    /**
     * @brief Tells the datasetManager about all the datasetStates of the data
     * that is read.
     *
     * Adds the following states: metadata, time, prod, freq, input, eigenvalue
     * and, if the data is stacked, stack.
     * Sets the dataset ID that should be given to the dataset coming from
     * the file that is read.
     *
     * If the dataset broker is being used, add a timeState to existing dataset
     * and pass that on.
     */
    void get_dataset_state(dset_id_t ds_id);

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

    Buffer* out_buf;

    // The metadata
    json _metadata;
    std::vector<time_ctype> _times;
    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;
    std::vector<prod_ctype> _prods;
    std::vector<input_ctype> _inputs;
    std::vector<stack_ctype> _stack;
    std::vector<rstack_ctype> _rstack;
    std::vector<uint32_t> _ev;
    uint32_t _num_stack;

    // whether to read in chunks
    bool chunked;

    // whether to use comet to track dataset IDs
    bool use_comet;

    // Read chunk size (freq, prod, time)
    std::vector<int> chunk_size;
    // time chunk size
    size_t chunk_t;
    // freq chunk size
    size_t chunk_f;

    // Number of elements in a chunked row
    size_t row_size;

    // the input file
    std::string filename;
    int fd;
    uint8_t* mapped_file;

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;

    // Dataset ID to assign to output frames
    dset_id_t out_dset_id;

    // the dataset state for the time axis
    state_id_t tstate_id;

    // Store dataset IDs present in file for quick access
    std::map<dset_id_t, dset_id_t> ds_in_file;

    // The read rate
    double max_read_rate;

    // Sleep time after reading
    double sleep_time;
};

class ensureOrdered : public kotekan::Stage {

public:
    ensureOrdered(kotekan::Config& config, const string& unique_name,
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
    // visRawReader reads chunks with frequency as fastest varying index
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
