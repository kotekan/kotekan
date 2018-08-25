/*****************************************
@file
@brief Read visFileRaw data.
- visRawReader : public KotekanProcess
*****************************************/
#ifndef _VIS_RAW_READER_HPP
#define _VIS_RAW_READER_HPP

#include "json.hpp"
#include "buffer.h"
#include "visUtil.hpp"
#include "visBuffer.hpp"
#include "KotekanProcess.hpp"

using json = nlohmann::json;

/**
 * @class visRawReader
 * @brief Read and stream a raw visibility file.
 *
 * This will divide the file up into time-frequency chunks of set size and
 * stream out the frames with time as the *fastest* index.
 *
 * @par Buffers
 * @buffer out_buf The data read from the raw file.
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf   readahead_blocks		Int. Number of blocks to advise OS to read ahead of current read.
 * @conf   chunk_size			Array of [int, int, int]. Read chunk size (freq, prod, time). If not specified will read file contiguously.
 * @conf   infile				String. Path to the (data-meta-pair of) files to read (e.g. "/path/to/0000_000", without .data or .meta).
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class visRawReader : public KotekanProcess {

public:
    /// default constructor
    visRawReader(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container);

    ~visRawReader();

    void apply_config(uint64_t fpga_seq);

    /// Main loop over buffer frames
    void main_thread();

    /**
     * @brief Get the times in the file.
     **/
    const std::vector<time_ctype>& times() { return _times; }

    /**
     * @brief Get the frequencies in the file.
     **/
    const std::vector<freq_ctype>& freqs() { return _freqs; }

    /**
     * @brief Get the products in the file.
     **/
    const std::vector<prod_ctype>& prods() { return _prods; }

    /**
     * @brief Get the stack in the file.
     **/
    const std::vector<stack_pair>& stack() { return _stack; }

    /**
     * @brief Get the inputs in the file.
     **/
    const std::vector<input_ctype>& inputs() { return _inputs; }

    /**
     * @brief Get the ev axis in the file.
     **/
    const std::vector<uint32_t>& ev() { return _ev; }

    /**
     * @brief Get the metadata saved into the file.
     **/
    const json& metadata() { return _metadata; }

private:

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

    Buffer * out_buf;

    // The metadata
    json _metadata;
    std::vector<time_ctype> _times;
    std::vector<freq_ctype> _freqs;
    std::vector<prod_ctype> _prods;
    std::vector<input_ctype> _inputs;
    std::vector<stack_pair> _stack;
    std::vector<uint32_t> _ev;

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
    uint8_t * mapped_file;

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;
};

#endif