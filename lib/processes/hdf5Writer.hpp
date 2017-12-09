#ifndef HDF5_WRITER_HPP
#define HDF5_WRITER_HPP

#include <unistd.h>
#include "fpga_header_functions.h"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visFile.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"

// Define an ordering on stream ids so they can be used in a std::map
struct compareStream {
    bool operator() (const stream_id_t& lhs, const stream_id_t& rhs) const;
};


/// Kotekan Process for writing data out into an HDF5 file.
class hdf5Writer : public KotekanProcess {
public:
    hdf5Writer(Config &config,
               const string& unique_name,
               bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();


private:

    /// Setup the acquisition
    void init_acq();

    /// Given a list of stream_ids infer the frequencies in the file, and create
    /// a mapping from id to frequency index
    void setup_freq(const std::vector<stream_id_t>& stream_ids);

    /// Figure out the acqusition start
    void setup_acq_start(const std::vector<timeval>& start_times);

    // Parameters saved from the config files
    size_t num_freq;
    size_t num_elements;
    bool reorder_freq;
    std::string root_path;
    std::vector<int32_t> enabled_chunks;

    std::string instrument_name;

    // The current file of visibilities that we are writing
    std::unique_ptr<visFileBundle> file_bundle;

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<struct Buffer*, unsigned int>> buffers;

    // The list of frequencies and inputs that gets written into the index maps
    // of the HDF5 files
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    // The mapping from stream_id to local frequency index
    std::map<stream_id_t, uint32_t, compareStream> freq_stream_map;

    // A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;
    bool enabled;

};

#endif
