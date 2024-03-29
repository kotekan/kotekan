/*****************************************
@file
@brief Stage for eigen-factoring the visibilities
- eigenVis : public kotekan::Stage
*****************************************/
#ifndef EIGENVIS_HPP
#define EIGENVIS_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, state_id_t
#include "visUtil.hpp"         // for movingAverage

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class eigenVis
 * @brief Perform eigen factorization of the visibilities
 *
 * This task performs the factorization of the visibility matrix into
 * ``num_ev`` eigenvectors and eigenvalues and stores them in reserve space
 * in the ``VisBuffer``. They are stored in descending order of the eigenvalue.
 *
 * @par Buffers
 * @buffer in_buf The set of buffers coming out the GPU buffers
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in the
 *                              correlator data.
 * @conf  block_size            Int. The block size of the packed data.
 * @conf  num_ev                UInt. The number of eigenvectors to be calculated as
 *                              an approximation to the visibilities.
 * @conf  num_diagonals_filled  Int, default 0. Number of diagonals to fill with
 *                              the previous time step's solution prior to
 *                              factorization. For example, setting to 1 will replace
 *                              the main diagonal only. Filled with zero on the first
 *                              time step.
 * @conf  exclude_inputs        List of UInts, optional. Inputs to exclude (rows and
 *                              columns to set to zero) in visibilities prior to
 *                              factorization.
 *
 * @par Metrics
 * @metric kotekan_eigenvis_comp_time_seconds
 *         Time required to find eigenvectors. An exponential moving average over
 *         ~10 samples.
 * @metric kotekan_eigenvis_eigenvalue
 *         The value of each eigenvalue calculated, or the RMS.
 * @metric kotekan_eigenvis_lapack_failure_total
 *         The number of frames skipped due to LAPACK failing (because of bad input data
 *         or other reasons).
 *
 * @author Kiyoshi Masui
 */
class eigenVis : public kotekan::Stage {

public:
    eigenVis(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);
    virtual ~eigenVis() = default;
    void main_thread() override;

private:
    dset_id_t change_dataset_state(dset_id_t input_dset_id);

    Buffer* input_buffer;
    Buffer* output_buffer;

    uint32_t num_eigenvectors;
    uint32_t num_diagonals_filled;
    /// List of input indeces to zero prior to decomposition.
    std::vector<uint32_t> exclude_inputs;

    /// Keep track of the average write time
    movingAverage calc_time;

    state_id_t ev_state_id;
    dset_id_t input_dset_id = dset_id_t::null;
};

#endif
