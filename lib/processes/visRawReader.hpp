#ifndef _VIS_RAW_READER_HPP
#define _VIS_RAW_READER_HPP

#include "json.hpp"

#include "buffer.h"
#include "visUtil.hpp"
#include "visBuffer.hpp"
#include "KotekanProcess.hpp"

using json = nlohmann::json;

class visRawReader : public KotekanProcess {

public:
    visRawReader(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container);

    ~visRawReader();

    void apply_config(uint64_t fpga_seq);

    void main_thread();

    const std::vector<time_ctype>& times() { return _times; }
    const std::vector<freq_ctype>& freqs() { return _freqs; }
    const std::vector<prod_ctype>& prods() { return _prods; }
    const std::vector<input_ctype>& inputs() { return _inputs; }
    const std::vector<uint32_t>& ev() { return _ev; }

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

    // Read chunk size (freq, prod, time)
    std::vector<int> chunk_size;
    size_t chunk_t;
    size_t chunk_f;

    // Number of elements in a chunked row
    size_t row_size;

    bool time_ordered = true;

    std::string filename;

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime;

    int fd;
    uint8_t * mapped_file;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;

    // Timing
    double read_time = 0.;
    double last_time;

    // count read data size
    float mbytes_read;
};

#endif
