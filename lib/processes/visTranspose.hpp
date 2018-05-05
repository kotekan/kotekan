#ifndef VISTRANSPOSE
#define VISTRANSPOSE

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "visFileArchive.hpp"

using json = nlohmann::json;

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

    std::string filename;

    // Datasets to be stored until ready to write
    std::vector<time_ctype> time;
    std::vector<cfloat> vis;
    std::vector<float> vis_weight;
    std::vector<cfloat> gain_coeff;
    std::vector<int32_t> gain_exp;
    std::vector<float> eval;
    std::vector<cfloat> evec;
    std::vector<float> erms;

    /// The list of frequencies and inputs that get written into the index maps
    /// of the HDF5 files
    std::vector<time_ctype> times;
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;
    std::vector<prod_ctype> prods;
    std::vector<uint32_t> ev;
    json metadata;

    /// A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;

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

    // Buffer for writing
    std::vector<char> write_buf;

    size_t f_ind = 0;
    size_t t_ind = 0;

};

#endif
