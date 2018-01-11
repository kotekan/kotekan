#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include <unistd.h>
#include "fpga_header_functions.h"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visFile.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"

// Define an ordering on stream ids so they can be used in a std::map
/*struct compareStream {
    bool operator() (const stream_id_t& lhs, const stream_id_t& rhs) const;
};*/


// Merge multiple gpu streams and transform into a visibility buffer
class visTransform : public KotekanProcess {

public:
    visTransform(Config &config,
                const string& unique_name,
                bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:

    // Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> input_buffers;
    Buffer * output_buffer;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

};


// Provide some useful debugging out on a visibility buffer
class visDebug : public KotekanProcess {

public:
    visDebug(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:

    Buffer * buffer;
};


/// Kotekan Process for writing data out into an HDF5 file.
class visWriter : public KotekanProcess {
public:
    visWriter(Config &config,
              const string& unique_name,
              bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();


private:

    /// Setup the acquisition
    void init_acq();

    /// Given a list of stream_ids infer the frequencies in the file, and create
    /// a mapping from id to frequency index
    void setup_freq(const std::vector<uint32_t>& freq_ids);

    // Parameters saved from the config files
    size_t num_freq;
    std::string root_path;
    std::string instrument_name;

    // The current file of visibilities that we are writing
    std::unique_ptr<visFileBundle> file_bundle;

    // Input buffer to read from
    Buffer * buffer;

    // The list of frequencies and inputs that gets written into the index maps
    // of the HDF5 files
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

    // The mapping from frequency bin id to output frequency index
    std::map<uint32_t, uint32_t> freq_map;

    // A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;

    // TODO: remove me
    // Legacy params for supporting old HDF5 writing scheme
    bool node_mode;
    std::vector<int> enabled_chunks;
    bool enabled;
};

#endif
