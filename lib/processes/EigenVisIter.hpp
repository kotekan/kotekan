/*****************************************
@file
@brief Stage for eigen-factoring the visibilities using an iterative method
- EigenVisIter : public kotekan::Stage
*****************************************/
#ifndef EIGENVISITER_HPP
#define EIGENVISITER_HPP

#include "Stage.hpp"
#include "buffer.h"
#include "datasetManager.hpp"
#include "visUtil.hpp"
#include "LinearAlgebra.hpp" // TODO: figure out how to forward declare eig_t


/**
 * @class EigenVisIter
 * @brief Perform eigen factorization of the visibilities
 *
 * This task performs the factorization of the visibility matrix into
 * ``num_ev`` eigenvectors and eigenvalues and stores them in reserve space
 * in the ``visBuffer``. They are stored in descending order of the eigenvalue.
 *
 * This is performed by using a subspace iteration method with an augmented
 * Rayleigh-Ritz step and a progressive matrix completion of masked values.
 *
 * @par Buffers
 * @buffer in_buf The set of buffers coming out the GPU buffers
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
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
 * @metric kotekan_eigenvisiter_comp_time_seconds
 *         Time required to find eigenvectors. An exponential moving average over
 *         ~10 samples.
 * @metric kotekan_eigenvisiter_eigenvalue
 *         The value of each eigenvalue calculated, or the RMS.
 * @metric kotekan_eigenvisiter_lapack_failure_total
 *         The number of frames skipped due to LAPACK failing (because of bad input data
 *         or other reasons).
 *
 * @author Richard Shaw, Kiyoshi Masui
 */
class EigenVisIter : public kotekan::Stage {

public:
    EigenVisIter(kotekan::Config& config, const string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~EigenVisIter() = default;
    void main_thread() override;

private:

    // Update the dataset ID when we receive a new input dataset
    dset_id_t change_dataset_state(dset_id_t input_dset_id) const;

    // Update the prometheus metrics
    void update_metrics(uint32_t freq_id, dset_id_t dset_id,
                        double elapsed_time, const eig_t<cfloat>& eigpair,
                        const EigConvergenceStats& stats);

    // Calculate the mask to apply from the object parameters
    DynamicHermitian<float> calculate_mask(uint32_t num_elements) const;

    Buffer* input_buffer;
    Buffer* output_buffer;

    uint32_t num_eigenvectors;
    uint32_t num_diagonals_filled;

    // Parameters for convergence
    double tol_eval;
    double tol_evec;
    uint32_t num_ev_conv;
    uint32_t max_iterations;

    /// List of input indices to zero prior to decomposition.
    std::vector<uint32_t> exclude_inputs;

    /// Keep track of the average write time, per frequency and dataset ID
    std::map<std::pair<uint32_t, dset_id_t>, movingAverage> calc_time_map;

    state_id_t ev_state_id;
    dset_id_t input_dset_id = 0;
};

#endif
