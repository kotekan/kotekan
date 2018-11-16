/*****************************************
@file
@brief Processes for eigen-factoring the visibilities
- eigenVis : public KotekanProcess
*****************************************/
#ifndef EIGENVIS_HPP
#define EIGENVIS_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"

/**
 * @class eigenVis
 * @brief Perform eigen factorization of the visibilities
 *
 * This task performs the factorization of the visibility matrix into
 * ``num_ev`` eigenvectors and eigenvalues and stores them in reserve space
 * in the ``visBuffer``. They are stored in descending order of the eigenvalue.
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
 * @conf  num_ev                Int. The number of eigenvectors to be calculated as
 *                              an approximation to the visibilities.
 * @conf  num_diagonals_filled  Int, default 0. Number of diagonals to fill with
 *                              the previous time step's solution prior to
 *                              factorization. For example, setting to 1 will replace
 *                              the main diagonal only. Filled with zero on the first
 *                              time step.
 * @conf  exclude_inputs        List of Ints, optional. Inputs to exclude (rows and
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
class eigenVis : public KotekanProcess {

public:
    eigenVis(Config& config,
             const string& unique_name,
             bufferContainer &buffer_container);
    ~eigenVis();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    struct Buffer *input_buffer;
    struct Buffer *output_buffer;

    int32_t num_eigenvectors;
    int32_t num_diagonals_filled;
    /// List of input indeces to zero prior to decomposition.
    std::vector<int32_t> exclude_inputs;

    /// Keep track of the average write time
    movingAverage calc_time;

};

#endif
