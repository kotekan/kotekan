/*****************************************
@file
@brief Base class for transposing raw files into HDF5 files.
- Transpose : public kotekan::Stage
*****************************************/
#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "dset_id.hpp"
#include "visUtil.hpp" // for cfloat, time_ctype, freq_ctype, input_ctype, prod_ctype

#include "json.hpp" // for json

#include <chrono>
#include <memory>   // for shared_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class Transpose
 * @brief Generic base class that transposes the data to make time the fastest varying value,
 * compresses it and writes it to a file.
 *
 * All classes which inherit from this should provide the following API:
 *
 *  get_dataset_state(dset_id_t ds_id);
 *  get_frame_data();
 *  create_hdf5_file();
 *  copy_frame_data(uint32_t freq_index, uint32_t time_index);
 *  copy_flags(uint32_t time_index);
 *  write_chunk();
 *  increment_chunk();
 *
 * The data is received as one-dimensional arrays
 * that represent flattened-out time-X-frequency matrices. These are transposed
 * and flattened out again to be written to a file. In other words,
 * the transposition makes time the fastest-varying for the data values,
 * where it was frequency before.
 * This stage expects the data to be ordered like RawReader does.
 * Other stages might not gaurentee this same order.
 *
 * @warning Don't run this anywhere but on the transpose (gossec) node.
 * The OpenMP calls could cause issues on systems using kotekan pin
 * priority threads (likely the GPU nodes).
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format Buffer.
 *         @buffer_metadata Metadata
 *
 * @conf   chunk_size           Array of [int, int, int]. Chunk size of the data
 *                              (freq, prod, time).
 * @conf   outfile              String. Path to the (data-meta-pair of) files to
 *                              write to (e.g. "/path/to/0000_000", without .h5).
 * @conf   comet_timeout        Float, default 60. Timeout for communications with
 *                              dataset broker.
 *
 * @par Metrics
 * @metric kotekan_transpose_data_transposed_bytes
 *         The total amount of data processed in bytes.
 *
 * @author Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class Transpose : public kotekan::Stage {
public:
    /// Constructor; loads parameters from config
    Transpose(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);
    ~Transpose() = default;

    /// Main loop over buffer frames
    void main_thread() override;

protected:
    // Config values
    std::string filename;
    std::chrono::duration<float> timeout;

    // Buffers
    Buffer* in_buf;

    // Input buffer frame ID
    frameID frame_id;

    nlohmann::json metadata;

    // HDF5 chunk size
    std::vector<int> chunk;

    // Size of time dimension of chunk
    size_t chunk_t;

    // Size of frequency dimension of chunk
    size_t chunk_f;

    // Keep track of the non-zero flags found so far
    std::vector<bool> found_flags;

    // Keep track of the size to write out
    // size of frequency and time dimension of chunk when written to file
    size_t write_f, write_t;

    size_t f_ind = 0;
    size_t t_ind = 0;

    // Flags to indicate incomplete chunks
    bool t_edge = false;
    bool f_edge = false;

    // Effective dimension of data
    size_t eff_data_dim;

    /// Number of products to write
    size_t num_time;
    size_t num_freq;

    // Datasets to be stored until ready to write
    std::vector<dset_id_str> dset_id;

private:
    /// Request dataset states from the datasetManager and prepare all metadata
    /// that is not already set in the constructor.
    virtual bool get_dataset_state(dset_id_t ds_id) = 0;

    // Get frame size, fpga_seq_total and dataset_id from FrameView
    virtual std::tuple<size_t, uint64_t, dset_id_t> get_frame_data() = 0;

    // Create FileArchive
    virtual void create_hdf5_file() = 0;

    // Copy data into local vectors
    virtual void copy_frame_data(uint32_t freq_index, uint32_t time_index) = 0;

    // Copy flags into local vectors
    virtual void copy_flags(uint32_t time_index) = 0;

    // Write datasets to file
    virtual void write_chunk() = 0;

    // Increment between chunks
    virtual void increment_chunk() = 0;

    /// Extract the base dataset ID
    dset_id_t base_dset(dset_id_t ds_id);
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
