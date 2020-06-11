#ifndef VISTRANSPOSE
#define VISTRANSPOSE

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "visFileArchive.hpp"  // for visFileArchive
#include "visFileH5.hpp"
#include "visUtil.hpp" // for cfloat, time_ctype, freq_ctype, input_ctype, prod_ctype

#include "json.hpp" // for json

#include <chrono>
#include <memory>   // for shared_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


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
 * This stage expects the data to be ordered like visRawReader does.
 * Other stages might not guarentee this same order.
 *
 * @warning Don't run this anywhere but on the transpose (gossec) node.
 * The OpenMP calls could cause issues on systems using kotekan pin
 * priority threads (likely the GPU nodes).
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format VisBuffer.
 *         @buffer_metadata VisMetadata
 *
 * @conf   chunk_size           Array of [int, int, int]. Chunk size of the data
 *                              (freq, prod, time).
 * @conf   outfile              String. Path to the (data-meta-pair of) files to
 *                              write to (e.g. "/path/to/0000_000", without .h5).
 * @conf   comet_timeout        Float, default 60. Timeout for communications with
 *                              dataset broker.
 *
 * @par Metrics
 * @metric kotekan_vistranspose_data_transposed_bytes
 *         The total amount of data processed in bytes.
 *
 * @author Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class visTranspose : public kotekan::Stage {
public:
    /// Constructor; loads parameters from config
    visTranspose(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~visTranspose() = default;

    /// Main loop over buffer frames
    void main_thread() override;

private:
    /// Request dataset states from the datasetManager and prepare all metadata
    /// that is not already set in the constructor.
    bool get_dataset_state(dset_id_t ds_id);

    /// Extract the base dataset ID
    dset_id_t base_dset(dset_id_t ds_id);

    // Buffers
    Buffer* in_buf;

    // HDF5 chunk size
    std::vector<int> chunk;
    // size of time dimension of chunk
    size_t chunk_t;
    // size of frequency dimension of chunk
    size_t chunk_f;

    // Config values
    std::string filename;
    std::chrono::duration<float> timeout;

    // Datasets to be stored until ready to write
    std::vector<time_ctype> time;
    std::vector<cfloat> vis;
    std::vector<float> vis_weight;
    std::vector<float> eval;
    std::vector<cfloat> evec;
    std::vector<float> erms;
    std::vector<cfloat> gain;
    std::vector<float> frac_lost;
    std::vector<float> frac_rfi;
    std::vector<dset_id_str> dset_id;
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
    nlohmann::json metadata;

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
};

template<typename T>
inline void strided_copy(T* in, T* out, size_t offset, size_t stride, size_t n_val) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n_val; i++) {
        out[offset + i * stride] = in[i];
    }
}

#endif
