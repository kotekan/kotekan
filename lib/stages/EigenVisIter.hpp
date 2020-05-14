/*****************************************
@file
@brief Stage for eigen-factoring the visibilities using an iterative method
- EigenVisIter : public kotekan::Stage
*****************************************/
#ifndef EIGENVISITER_HPP
#define EIGENVISITER_HPP

#include <blaze/Blaze.h> // for HermitianMatrix
#include <map>           // for map
#include <stdint.h>      // for uint32_t, int32_t
#include <string>        // for string
#include <utility>       // for pair
#include <vector>        // for vector

// TODO: figure out how to forward declare eig_t
#include "Config.hpp"            // for Config
#include "LinearAlgebra.hpp"     // for EigConvergenceStats
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, state_id_t
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily
#include "visUtil.hpp"           // for movingAverage, cfloat


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
 * @buffer in_buf The stream to eigen decompose.
 *         @buffer_format visBuffer structured
 *         @buffer_metadata VisMetadata
 * @buffer out_buf Output stream with the calculated eigen-pairs.
 *         @buffer_format visBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf  num_elements     Int. The number of elements (i.e. inputs) in the
 *                         correlator data.
 * @conf  block_size       Int. The block size of the packed data.
 * @conf  num_ev           UInt. The number of eigenvectors to be calculated as
 *                         an approximation to the visibilities.
 * @conf  bands_filled     List of pairs of ints, default empty. Ranges of diagonal
 *                         bands to mask out before the factorization. These are
 *                         iteratively filled within the eigen decomposition code.
 * @conf  block_fill_size  UInt, default 0. Mask out blocks of this size on the diagonal.
 * @conf  exclude_inputs   List of UInts, optional. Inputs to exclude (rows and
 *                         columns to set to zero) in visibilities prior to
 *                         factorization.
 * @conf  tol_eval         Float, default 1e-6. Fractional change in evals must be less
 *                         than this for convergence.
 * @conf  tol_evec         Float, default 1e-5. Total eigenvector overlap must be less
 *                         than this.
 * @conf  max_iterations   UInt. Maximum number of iterations to compute.
 * @conf  num_ev_conv      UInt. Test only the top `num_ev_conv` eigenpairs for convergence.
 * @conf  krylov           UInt, default 2. Size of the Krylov basis to use.
 * @conf  subspace         UInt, default 3. Number of subspace iteration substeps.
 *
 * @par Metrics
 * @metric kotekan_eigenvisiter_comp_time_seconds
 *         Time required to find eigenvectors. An exponential moving average over
 *         ~10 samples.
 * @metric kotekan_eigenvisiter_eigenvalue
 *         The value of each eigenvalue calculated, or the RMS.
 * @metric kotekan_eigenvisiter_iterations
 *         Number of iterations required to compute the last sample.
 * @metric kotekan_eigenvisiter_eigenvalue_convergence
 *         Eigenvalue convergence parameter of the last sample.
 * @metric kotekan_eigenvisiter_eigenvector_convergence
 *         Eigenvector convergence parameter of the last sample.
 *
 * @author Richard Shaw, Kiyoshi Masui
 */
class EigenVisIter : public kotekan::Stage {

public:
    EigenVisIter(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    virtual ~EigenVisIter() = default;
    void main_thread() override;

private:
    // Update the dataset ID when we receive a new input dataset
    dset_id_t change_dataset_state(dset_id_t input_dset_id) const;

    // Update the prometheus metrics
    void update_metrics(uint32_t freq_id, dset_id_t dset_id, double elapsed_time,
                        const eig_t<cfloat>& eigpair, const EigConvergenceStats& stats);

    // Calculate the mask to apply from the object parameters
    DynamicHermitian<float> calculate_mask(uint32_t num_elements) const;

    Buffer* in_buf;
    Buffer* out_buf;

    uint32_t _num_eigenvectors;

    // Parameters for convergence
    double _tol_eval;
    double _tol_evec;
    uint32_t _num_ev_conv;
    uint32_t _max_iterations;
    uint32_t _krylov;
    uint32_t _subspace;

    /// Parameters for masking the matrix
    std::vector<uint32_t> _exclude_inputs;
    uint32_t _block_fill_size;
    std::vector<std::pair<int32_t, int32_t>> _bands_filled;

    /// Keep track of the average write time, per frequency and dataset ID
    std::map<std::pair<uint32_t, dset_id_t>, movingAverage> calc_time_map;

    state_id_t ev_state_id;
    dset_id_t input_dset_id = dset_id_t::null;

    kotekan::prometheus::Gauge& comp_time_seconds_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& eigenvalue_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& iterations_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& eigenvalue_convergence_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& eigenvector_convergence_metric;
};

#endif
