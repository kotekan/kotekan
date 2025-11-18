/*****************************************
@file
@brief Accumulation and gating of visibility data.
- N2Accumulate : public kotekan::Stage
*****************************************/
#ifndef N2_ACCUMULATE_HPP
#define N2_ACCUMULATE_HPP

#include "CHORDTelescope.hpp"    // for CHORDTelescope
#include "Config.hpp"            // for Config
#include "N2Util.hpp"            // for frameID
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter, MetricFamily

#include <cstdint> // for int64_t, int32_t
#include <string>  // for string
#include <vector>  // for vector

using N2::frameID;

/**
 * @class N2Accumulate
 * @brief Accumulate the high rate GPU output into integrated VisBuffers.
 *
 * This stage accumulates output from the N2k GPU correlator into integrated
 * visibility buffers.
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed information
 *         @buffer_metadata chordMetadata
 * @buffer out_buf The accumulated and tagged data.
 *         @buffer_format VisBuffer structured.
 *         @buffer_metadata VisMetadata
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
    int64_t _num_freq_per_n2k_frame;
    int64_t _num_n2k_samples_to_accumulate;

    bool _packet_loss_is_scalar;

    int64_t _n_fpga_samples_per_n2k_frame;
    int64_t _n_fpga_samples_per_n2k_correlation;
    int64_t _n_integrations_per_n2k_frame;

    int64_t _rfi_downsampling_factor; ///< Downsampling factor for RFI mask

    int64_t _num_elements; ///< Total number of telescope elements (~2 * num dishes)
    int64_t _num_ev;       ///< Number of eigenvalues/vectors

    // Absolute frame counter (TODO: determine this another way)
    uint64_t _abs_frame_count;

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
    std::vector<int32_t> _vis_even;
    std::vector<float> _weights;
    // number of fpga samples, per frequency, in frame
    std::vector<int32_t> _n_valid_fpga_samples_in_vis;
    std::vector<int32_t> _n_valid_fpga_samples_in_vis_even;
    std::vector<float> _n_valid_sample_diff_sq_sum;
    std::vector<int32_t> _n_rfi_samples_in_vis;
    int64_t _vis_samples_in_out_frame;
    int64_t _accum_fpga_start_tick;

    // The telescope
    const CHORDTelescope& _tel;

    // Reference to the prometheus metric that we will use for counting skipped
    // frames
    // TODO ...
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& skipped_frame_counter;
};

#endif
