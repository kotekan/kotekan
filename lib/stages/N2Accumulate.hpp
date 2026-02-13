/*****************************************
@file
@brief Accumulation and gating of visibility data.
- N2Accumulate : public kotekan::Stage
*****************************************/
#ifndef N2_ACCUMULATE_HPP
#define N2_ACCUMULATE_HPP

#include "Config.hpp"            // for Config
#include "N2Util.hpp"            // for frameID
#include "Stage.hpp"             // for Stage
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter, MetricFamily

#include <cstdint> // for int64_t, int32_t
#include <string>  // for string
#include <vector>  // for vector

using N2::frameID;

/**
 * @class N2Accumulate
 * @brief Accumulate the high rate GPU output into integrated N2Buffers.
 *
 * This stage accumulates output from the N2k GPU correlator into integrated
 * visibility buffers.
 *
 * TODO:    - ERA binning
 *          - fringestopping
 *          - radiometer_chi2
 *          - improved variance estimator
 *
 * num_integrations := samples_per_dataset / sub_integration_ntime
 * num_freq := num_freq_per_n2k_frame
 *
 * num_corr_blocks_lin := num_elements / 16
 * num_corr_blocks := num_corr_blocks_lin * (num_corr_blocks_lin + 1) / 2
 *
 * num_count_blocks_lin := num_elements / 8 / 8     # 8 for blocksize,
 *                                                  # 8 for downsampling
 * num_count_blocks := num_count_blocks_lin * (num_count_blocks_lin + 1) / 2
 *
 * @par Buffers
 * @buffer  in_buf          Correlation buffer from n2k. Blocked Lower Triangular.
 *      @buffer_format      NDArray int32 [num_integrations, num_freq, num_corr_blocks, 16, 16, 2]
 *      @buffer_metadata    chordMetadata
 * @buffer  in_counts_buf   Counts buffer from n2k. Blocked Lower Triangular
 *      @buffer_format      NDArray int32 [num_integrations, num_freq, num_count_blocks, 8, 8]
 *      @buffer_metadata    chordMetadata
 * @buffer  in_rfimask_buf  RFImask buffer from n2k, the same mask used to compute
 *                          the correlation.
 *         @buffer_format   NDArray uint1x8 [samples_per_dataset / 128 / 8, num_freq, 128]
 *         @buffer_metadata chordMetadata
 * @buffer  out_buf         The accumulated and tagged data.
 *      @buffer_format N2Buffer. layout=FullUpperTri, num_ev=0
 *      @buffer_metadata N2Metadata
 *
 * @conf    num_freq_per_n2k_frame          int64_t Number of frequencies in
 *                                          buffers, required.
 * @conf    num_n2k_samples_to_accumulate   int64_t Number of samples (subintegrations)
 *                                          to accumulate in each output frame. Default: 0.
 * @conf    packet_loss_is_scalar           bool    Whether the packet loss (ie. the counts
 *                                          matrix) is a scalar in dish element or not.  If so,
 *                                          all baselines use the same value from `counts`, the
 *                                          first element in the buffer. The `false` case has
 *                                          not been implemented.
 * @conf    samples_per_data_set            int64_t Total number of time samples covered by each
 *                                          input frame. nt_outer in n2k.
 * @conf    sub_integration_ntime           int64_t Number of time samples integrated in each
 *                                          entry in correlation and counts buffers.  n2_inner
 *                                          in n2k.
 * @conf    rfi_downsampling_factor         int64_t Number of time samples used to compute
 *                                          RFImask.  The values in the RFIMask buffer are
 *                                          repeated this many times.
 * @conf    num_elements                    int64_t Number of elements (num_dish x num_pol) in
 *                                          the buffers.
 * @conf    do_fringestop                   bool    Whether to fringestop incoming correlations.
 *                                          Default: False
 */
class N2Accumulate : public kotekan::Stage {
public:
    N2Accumulate(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~N2Accumulate() = default;

    /**
     * @brief The main thread function for N2Accumulate.
     *
     * This function is responsible for the main logic of the N2Accumulate class.
     */
    void main_thread() override;

    /**
     * @brief   Return the sequence number for the start of the next accumulation bin edge,
     *          must be after the given seq.
     *
     * @param   seq     The current sequence number, the returned seq will be that of the first
     *                  bin edge after this seq.
     *
     * @return  The seq for the next bin edge.
     */
    int64_t get_next_accum_start_tick(int64_t seq);

    /**
     * @brief   Return a montonic index (counter) for the accumulation bin beginning at seq_start.
     * Will increase by 1 for each accumulation performed. May not begin at 0. Restart-safe.
     *
     * @param   seq_start   Sequence tick for the start of the bin in question
     *
     * @return  The index value for this accumulation bin.
     */
    int64_t get_abs_accum_idx(int64_t seq_start);

    /**
     * @brief Accumulate the rfimask over the given n2k integration into _n_rfi_samples_in_vis
     *
     * @param   rfimask The raw rfimask used in n2k to compute the correlation.
     * @param   t_vis   Time index denoting the current sample being accumulated,
     *                  in [0, samples_per_dataset / sub_integration_ntime)
     */
    void accumulate_rfimask_in_sample(const uint8_t* rfimask, int64_t t_vis);

    /**
     * @brief Copy accumulated visibility matrix and weights to the output buffer,
     * reset the visibility and weights matrices.
     *
     * Helper function to keep code a bit more readable.
     *
     * @param in_frame_id The input frame ID.
     * @param out_frame_id The output frame ID.
     * @return bool True if successful, false otherwise.
     */
    bool output_and_reset(frameID& in_frame_id, frameID& out_frame_id);

private:
    // Buffers to read/write
    Buffer* in_buf;         /// Buffer containing input correlations
    Buffer* in_counts_buf;  /// Buffer containing input counts
    Buffer* in_rfimask_buf; /// Buffer containing input rfimask
    Buffer* out_buf;        /// Output for the main vis dataset only

    // Parameters saved from the config files
    const int64_t _num_freq_per_n2k_frame;
    const bool _bin_in_ERA;
    const int64_t _num_n2k_samples_to_accumulate;
    const uint32_t _num_bins_per_rotation;

    const bool _packet_loss_is_scalar;

    const int64_t _n_fpga_samples_per_n2k_frame;
    const int64_t _n_fpga_samples_per_n2k_correlation;
    int64_t _n_integrations_per_n2k_frame;

    const int64_t _rfi_downsampling_factor; ///< Downsampling factor for RFI mask

    const int64_t _num_elements; ///< Total number of telescope elements (~2 * num dishes)

    const int _num_workers; ///< number of OpenMP threads to use to process data

    const bool _do_fringestop; ///< Whether to fringestop
    const int _debug_accum_mode;

    // Some derived parameters

    int64_t _N2_num_products; ///< Number of products produced by the N2 correlator

    /// The correlation and counts matrices are "block" triangular matrices.
    /// The size of the blocks in the matrices are fixed.

    static constexpr int64_t _n2k_correlation_blocksize = 16; // THIS IS ALWAYS 16
    int64_t
        _n2k_correlation_lin_blocks; ///< Number blocks in the blocked correlation matrix from n2k
    int64_t _n2k_correlation_num_blocks;   ///< Total number of blocks in n2k's correlation matrix
    int64_t _n2k_correlation_num_products; ///< Total number of products in n2k's correlation matrix

    static constexpr int64_t _n2k_counts_blocksize = 8; // THIS IS ALWAYS 8
    int64_t _n2k_counts_lin_blocks;   ///< Linear number of blocks in the counts matrix
    int64_t _n2k_counts_num_blocks;   ///< Total number of blocks in the counts matrix
    int64_t _n2k_counts_num_products; ///< Total number of products in n2k's counts matrix

    // The below vectors are initialized in the constructor after _num_vis_products
    // and _num_freq_in_frame are known.
    std::vector<int32_t> _vis;
    std::vector<float> _weights;
    // number of fpga samples, per frequency, in frame
    std::vector<int32_t> _n_valid_fpga_samples_in_vis;
    std::vector<int32_t> _n_valid_fpga_samples_in_vis_even;
    std::vector<float> _n_valid_sample_diff_sq_sum;
    std::vector<int32_t> _n_rfi_samples_in_vis;
    int64_t _vis_samples_in_out_frame;
    int64_t _accum_fpga_start_tick;

    // The telescope
    const Telescope& _tel;

    // Reference to the prometheus metric that we will use for counting skipped
    // frames
    // TODO ...
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& skipped_frame_counter;
};

#endif
