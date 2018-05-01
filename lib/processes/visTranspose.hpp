#ifndef VISTRANSPOSE
#define VISTRANSPOSE

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "visFileArchive.hpp"

class visTranspose : public KotekanProcess {
public:
    /// Constructor; loads parameters from config
    visTranspose(Config &config, const string& unique_name, bufferContainer &buffer_container);
    ~visTranspose();

    /// Main loop over buffer frames
    void main_thread() override;

    void apply_config(uint64_t fpga_seq) override;
private:
    // Buffers
    Buffer * in_buf;

    // Size of a chunk
    size_t chunk_t;
    size_t chunk_f;

    // Datasets to be stored until ready to write
    std::span<time_ctype> time;
    std::span<cfloat> vis;
    std::span<float> vis_weight;
    std::span<cfloat> gain_coeff;
    std::span<int32_t> gain_exp;
    std::span<float> eval;
    std::span<cfloat> evec;
    std::span<float> erms;

    /// The list of frequencies and inputs that gets written into the index maps
    /// of the HDF5 files
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

    /// A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;

    // Vector of products if options to restrict them are present
    std::vector<prod_ctype> prods;

    /// Number of products to write
    size_t num_prod;
    size_t num_input;
    size_t num_time;
    size_t num_freq;
    size_t num_ev;

    // set up and write to files
    void mk_file();
    void transpose_write();

    std::shared_ptr<visFileArchive> file;

};

#endif
