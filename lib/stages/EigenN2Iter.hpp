/*****************************************
@file
@brief Stage for eigen-factoring the visibilities using an iterative method
- EigenN2Iter : public kotekan::Stage
*****************************************/
#ifndef EIGENN2ITER_HPP
#define EIGENN2ITER_HPP

#include <blaze/Blaze.h> // for HermitianMatrix
#include <map>           // for map
#include <stdint.h>      // for uint32_t, int32_t
#include <string>        // for string
#include <utility>       // for pair
#include <vector>        // for vector

// TODO: figure out how to forward declare eig_t
#include "Config.hpp"            // for Config
#include "LinearAlgebra.hpp"     // for EigConvergenceStats
#include "N2Util.hpp"            // for movingAverage, cfloat
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily


/**
 * @class EigenN2Iter
 * @brief Perform eigen factorization of the visibilities
 *
 * This task performs the factorization of the visibility matrix into
 * ``num_ev`` eigenvectors and eigenvalues and stores them in reserve space
 * in the ``VisBuffer``. They are stored in descending order of the eigenvalue.
 *
 * This is performed by using a subspace iteration method with an augmented
 * Rayleigh-Ritz step and a progressive matrix completion of masked values.
 *
 * This stage is similar to EigenVisIter, but works on N2Buffer inputs/outputs,
 * and without the dataset tracking functionality.
 *
 * @par Buffers
 * @buffer in_buf The stream to eigen decompose.
 *         @buffer_format N2Buffer structured
 *         @buffer_metadata N2Metadata
 * @buffer out_buf Output stream with the calculated eigen-pairs.
 *         @buffer_format N2Buffer structured
 *         @buffer_metadata N2Metadata
 * @buffer failed_buf Output stream to which buffers for which eigenvalues
 *         cannot be computed are sent. Leave empty to drop these frames.
 *         @buffer_format N2Buffer structured
 *         @buffer_metadata N2Metadata
 *
 * @conf  num_elements     Int. The number of elements (i.e. inputs) in the
 *                         correlator data.
 * @conf  block_size       Int. The block size of the packed data.
 * @conf  num_ev           UInt. The number of eigenvectors to be calculated as
 *                         an approximation to the visibilities.
 * @conf  diagonal_bands_filled     List of pairs of UInts, default empty. Ranges of diagonal
 *                         bands to mask out before the factorization. These are
 *                         iteratively filled within the eigen decomposition code.
 *                         Each pair is [start, end) indices of the band to be masked,
 *                         where 0 is the main diagonal, 1 is the first super-diagonal, etc.
 *                         Positive values should be provided as parameters, but both
 *                         super- and sub-diagonals are masked symmetrically.
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
 * @conf  num_blaze_workers UInt, default 0. If greater than 0 limit number of
 *                          blaze worker threads (kotekan wide)
 *
 * @par Metrics
 * @metric kotekan_eigenN2iter_comp_time_seconds
 *         Time required to find eigenvectors. An exponential moving average over
 *         ~10 samples.
 * @metric kotekan_eigenN2iter_eigenvalue
 *         The value of each eigenvalue calculated, or the RMS.
 * @metric kotekan_eigenN2iter_iterations
 *         Number of iterations required to compute the last sample.
 * @metric kotekan_eigenN2iter_eigenvalue_convergence
 *         Eigenvalue convergence parameter of the last sample.
 * @metric kotekan_eigenN2iter_eigenvector_convergence
 *         Eigenvector convergence parameter of the last sample.
 * @metric kotekan_eigenN2iter_num_failed_eigencalc
 *         The number of failed eigenvector decompositions.
 *
 *
 * @author Richard Shaw, Kiyoshi Masui
 */
class EigenN2Iter : public kotekan::Stage {

public:
    EigenN2Iter(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~EigenN2Iter() = default;
    void main_thread() override;

private:
    // Update the prometheus metrics
    void update_metrics(int freq_id, double elapsed_time, const eig_t<cfloat>& eigpair,
                        const EigConvergenceStats& stats);

    // Calculate the mask to apply from the object parameters
    DynamicHermitian<float> calculate_mask(size_t num_elements) const;

    Buffer* in_buf;
    Buffer* out_buf;
    Buffer* failed_buf;

    const size_t _num_eigenvectors;

    // Parameters for convergence
    const double _tol_eval;
    const double _tol_evec;
    const size_t _num_ev_conv;
    const size_t _max_iterations;
    const size_t _krylov;
    const size_t _subspace;

    /// Blaze number of threads
    uint32_t _num_blaze_workers;

    /// Parameters for masking the matrix
    std::vector<size_t> _exclude_inputs;
    const size_t _block_fill_size;
    std::vector<std::pair<size_t, size_t>> _diagonal_bands_filled;

    /// Keep track of the average write time, per frequency
    std::map<int, N2::movingAverage> calc_time_map;

    kotekan::prometheus::Gauge& comp_time_seconds_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& eigenvalue_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& iterations_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& eigenvalue_convergence_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& eigenvector_convergence_metric;
    kotekan::prometheus::Counter& num_failed_eigencalc;
};

#endif
