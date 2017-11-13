#ifndef HDF5_WRITER_HPP
#define HDF5_WRITER_HPP

#include "fpga_header_functions.h"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>
#include <highfive/H5File.hpp>

// Structs to represent the datatypes of the index maps
typedef struct {
    double centre;
    double width;
} freq_ctype;

typedef struct {
    uint16_t chan_id;
    char correlator_input[32];
} input_ctype;

typedef struct {
    uint64_t fpga_count;
    double ctime;
} time_ctype;

typedef struct {
    uint16_t input_a;
    uint16_t input_b;
} prod_ctype;

typedef struct {
    int32_t r;
    int32_t i;
} complex_int;


// Define an ordering on stream ids so they can be used in a std::map
struct compareStream {
    bool operator() (const stream_id_t& lhs, const stream_id_t& rhs) const;
};


/// Class to manage access to a CHIME correlator file
///
/// This is only designed with writing data in mind.
class visFile {

public:

    /// Create file (and lock file).
    /// \param name Name of the file to write
    /// \param freqs Frequencies channels that will be in the file
    /// \param inputs Inputs that are in the file
    visFile(const std::string& name,
            const std::string& acq_name,
            const std::string& inst_name,
            const std::string& notes,
            const std::vector<freq_ctype>& freqs,
            const std::vector<input_ctype>& inputs);
    ~visFile();


    /// Write a new time sample into this file
    /// \param new_time Time of sample
    /// \param freq_ind Index of the frequency we are writing
    /// \param new_vis Visibility data for this frequency
    /// \param new_weight Visibility weights for this frequency
    /// \param new_gcoeff Gain coefficient data
    /// \param new_gexp Gain exponent data
    /// \return The number of entries in the time axis
    size_t addSample(time_ctype new_time, uint32_t freq_ind,
                   std::vector<complex_int> new_vis,
                   std::vector<uint8_t> new_weight,
                   std::vector<complex_int> new_gcoeff,
                   std::vector<int32_t> new_gexp);

private:


    // Create the index maps from the frequencies and the inputs
    void createIndex(const std::vector<freq_ctype>& freqs,
                     const std::vector<input_ctype>& inputs);

    // Create the main visibility holding datasets
    void createDatasets(size_t nfreq, size_t ninput, size_t nprod);


    // Pointer to the underlying HighFive file
    std::unique_ptr<HighFive::File> file;

    std::string lock_filename;

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
    std::string acq_name;
    double acq_start_time;

    // The current file of visibilities that we are writing
    std::unique_ptr<visFile> current_file;

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

};


// These templated functions are needed in order to tell HighFive how the
// various structs are converted into HDF5 datatypes
namespace HighFive {
template <> DataType create_datatype<freq_ctype>();
template <> DataType create_datatype<time_ctype>();
template <> DataType create_datatype<input_ctype>();
template <> DataType create_datatype<prod_ctype>();
template <> DataType create_datatype<complex_int>();
};


#endif
