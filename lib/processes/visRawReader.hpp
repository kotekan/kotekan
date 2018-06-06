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
 * @conf  filename    Int. The raw file to read.
 * @conf  chunk_size  Int. The size of chunks to read in.
 *
 * @author Richard Shaw
 */
class visRawReader : public KotekanProcess {

public:
    visRawReader(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container);

    ~visRawReader();

    void apply_config(uint64_t fpga_seq);

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

    json _metadata;

    std::vector<time_ctype> _times;
    std::vector<freq_ctype> _freqs;
    std::vector<prod_ctype> _prods;
    std::vector<input_ctype> _inputs;
    std::vector<uint32_t> _ev;

    std::vector<int> chunk_size;

    bool time_ordered = true;

    std::string filename;

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime;

    int fd;
    uint8_t * mapped_file;
};

#endif