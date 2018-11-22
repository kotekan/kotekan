#ifndef VISTRANSPOSE
#define VISTRANSPOSE

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "visFileArchive.hpp"

using json = nlohmann::json;

/**
 * @class visTranspose
 * @brief Transposes the data to make time the fastest varying value, compresses
 * it and writes it to a file.
 *
 * The data (vis, weight, eval and evac) is received as one-dimensional arrays
 * that represent flattened-out time-X-frequency matrices. These are transposed
 * and flattened out again to be written to a file. In other words,
 * the transposition makes time the fastest-varying for the data values,
 * where it was frequency before.
 * This process expects the data to be ordered like visRawReader does.
 * Other processes might not guarentee this same order.
 *
 * @warning Don't run this anywhere but on the transpose (gossec) node.
 * The OpenMP calls could cause issues on systems using kotekan pin
 * priority threads (likely the GPU nodes).
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata
 *
 * @conf   chunk_size           Array of [int, int, int]. Chunk size of the data (freq, prod, time).
 * @conf   outfile              String. Path to the (data-meta-pair of) files to write to (e.g. "/path/to/0000_000", without .h5).
 * @conf   use_dataset_manager  Bool. If set to `true`, the metadata will be
 *                              fetched from the datasetManager, otherwise
 *                              infile has to be set and the metadata will be
 *                              read from there.
 * @conf   infile               String. Path to the data files to read (e.g.
 *                              "/path/to/0000_000", without .data/meta).
 *                              Only needed if use_dataset_manager is `False`.
 *
 * @par Metrics
 * @metric kotekan_vistranspose_data_transposed_bytes
 *         The total amount of data processed in bytes.
 *
 * @author Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class visTranspose : public KotekanProcess {
public:
    /// Constructor; loads parameters from config
    visTranspose(Config &config, const string& unique_name, bufferContainer &buffer_container);
    ~visTranspose() = default;

    /// Main loop over buffer frames
    void main_thread() override;

private:
    /// Wait for the first frames dataset ID, request dataset states from the
    /// datasetManager and prepare all metadata that is not already set in the
    /// constructor.
    void gather_metadata();

    // Buffers
    Buffer * in_buf;

    // HDF5 chunk size
    std::vector<int> chunk;
    //size of time dimension of chunk
    size_t chunk_t;
    //size of frequency dimension of chunk
    size_t chunk_f;

    // Config values
    std::string filename;
    bool _use_dataset_manager;

    // Datasets to be stored until ready to write
    std::vector<time_ctype> time;
    std::vector<cfloat> vis;
    std::vector<float> vis_weight;
    std::vector<float> eval;
    std::vector<cfloat> evec;
    std::vector<float> erms;
    std::vector<cfloat> gain;
    std::vector<float> frac_lost;
    std::vector<float> input_flags;
    std::vector<rstack_ctype> reverse_stack;

    // Keep track of the size to write out
    // size of frequency and time dimension of chunk when written to file
    size_t write_f, write_t;
    // flags to indicate incomplete chunks
    bool t_edge = false;
    bool f_edge = false;
    void increment_chunk();

    // keep track of the non-zero flags found so far
    std::vector<bool> found_flags;

    /// The list of frequencies and inputs that get written into the index maps
    /// of the HDF5 files
    std::vector<time_ctype> times;
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;
    std::vector<prod_ctype> prods;
    std::vector<uint32_t> ev;
    std::vector<stack_ctype> stack;
    json metadata;

    /// A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;

    /// Number of products to write
    size_t num_prod;
    size_t num_input;
    size_t num_time;
    size_t num_freq;
    size_t num_ev;
    size_t eff_prod_dim;

    // write datasets to file
    void write();

    std::shared_ptr<visFileArchive> file;

    // Buffer for writing
    std::vector<char> write_buf;

    size_t f_ind = 0;
    size_t t_ind = 0;

    const size_t BLOCK_SIZE = 32;
};

template<typename T>
inline void strided_copy(T* in, T* out, size_t offset, size_t stride, size_t n_val) {
    #pragma omp parallel for
    for (size_t i = 0; i < n_val; i++) {
        out[offset + i * stride] = in[i];
    }
}

#endif
