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
 * @brief Reads recorded data from .data and .meta files.
 *
 * Data file is mmaped to memory for performance reasons.
 *
 * @par Buffers
 * @buffer out_buf The output stream as read from file.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata
 *
 * @conf   readahead_blocks		Int. Number of blocks to advise OS to read ahead of current read.
 * @conf   chunk_size			Array of [int, int, int]. Read chunk size (freq, prod, time).
 * @conf   infile				String. Path to the (data-meta-pair of) files to read (e.g. "/path/to/0000_000", without .data or .meta).
 * @conf   time_ordered			Bool.
 *
 * @author Tristan Pinsonneault-Marotte
 */
class visRawReader : public KotekanProcess {

public:
    // default constructor
    visRawReader(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container);

    ~visRawReader();

    void apply_config(uint64_t fpga_seq);

    void main_thread();

    /**
     * @brief Get time values from the metadata.
     *
     * @returns A vector of time values from the metadata.
     **/
    const std::vector<time_ctype>& times() { return _times; }

    /**
     * @brief Get freq values from the metadata.
     *
     * @returns A vector of freq values from the metadata.
     **/
    const std::vector<freq_ctype>& freqs() { return _freqs; }

    /**
     * @brief Get prod values from the metadata.
     *
     * @returns A vector of prod values from the metadata.
     **/
    const std::vector<prod_ctype>& prods() { return _prods; }

    /**
     * @brief Get input values from the metadata.
     *
     * @returns A vector of input values from the metadata.
     **/
    const std::vector<input_ctype>& inputs() { return _inputs; }

    /**
     * @brief Get ev values from the metadata.
     *
     * @returns A vector of ev values from the metadata.
     **/
    const std::vector<uint32_t>& ev() { return _ev; }

    /**
     * @brief Get the metadata.
     *
     * @returns All metadata as a reference to a json object.
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
    std::vector<uint32_t> _ev;

    // Read chunk size (freq, prod, time)
    std::vector<int> chunk_size;
    // time chunk size
    size_t chunk_t;
    // freq chunk size
    size_t chunk_f;

    // Number of elements in a chunked row
    size_t row_size;

    bool time_ordered = true;

    // the input file
    std::string filename;
    int fd;
    uint8_t * mapped_file;

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;

    // Timing to measure I/O read speed
    double read_time = 0.;
    double last_time;
    // count read data size
    float mbytes_read;
};

#endif
