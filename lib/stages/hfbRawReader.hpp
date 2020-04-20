/*****************************************
@file
@brief Read hfbFileRaw data.
- hfbRawReader : public kotekan::Stage
*****************************************/
#ifndef _HFB_RAW_READER_HPP
#define _HFB_RAW_READER_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t
#include "visUtil.hpp"        // for freq_ctype (ptr only), input_ctype, prod_ctype, rstack_ctype

#include "json.hpp" // for json

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint8_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector


/**
 * @class hfbRawReader
 * @brief Read and stream a raw 21cm absorber file.
 *
 * This will divide the file up into time-frequency chunks of set size and
 * stream out the frames with time as the *fastest* index.
 *
 * @par Buffers
 * @buffer out_buf The data read from the raw file.
 *         @buffer_format hfbBuffer structured
 *         @buffer_metadata hfbMetadata
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
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class hfbRawReader : public kotekan::Stage {

public:
    /// default constructor
    hfbRawReader(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    ~hfbRawReader();

    /// Main loop over buffer frames
    void main_thread() override;

    /**
     * @brief Get the times in the file.
     **/
    const std::vector<time_ctype>& times() {
        return _times;
    }

    /**
     * @brief Get the stack in the file.
     **/
    const std::vector<stack_ctype>& stack() {
        return _stack;
    }

    /**
     * @brief Get the metadata saved into the file.
     **/
    const nlohmann::json& metadata() {
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
     */
    void change_dataset_state(dset_id_t ds_id);

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
    nlohmann::json _metadata;
    std::vector<time_ctype> _times;
    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;
    std::vector<uint32_t> _beams;
    std::vector<uint32_t> _subfreqs;
    std::vector<stack_ctype> _stack;
    std::vector<rstack_ctype> _rstack;
    uint32_t _num_stack;

    // whether to read in chunks
    bool chunked;

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

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime, nbeam, nsubfreq;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;

    // The ID for the data coming from the file that is read.
    dset_id_t _dataset_id;

    // The read rate
    double max_read_rate;

    // Sleep time after reading
    double sleep_time;
};

#endif
