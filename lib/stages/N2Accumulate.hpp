/*****************************************
@file
@brief Accumulation and gating of visibility data.
- N2Accumulate : public kotekan::Stage
*****************************************/
#ifndef N2_ACCUMULATE_HPP
#define N2_ACCUMULATE_HPP

#include <time.h>                 // for timespec
#include <json.hpp>               // for json
#include <cstdint>                // for int64_t, int32_t, uint64_t, uint32_t
#include <string>                 // for string
#include <vector>                 // for vector
#include <complex>                // for complex
#include <iosfwd>                 // for ostream

#include "Config.hpp"             // for Config
#include "N2Util.hpp"             // for frameID
#include "Stage.hpp"              // for Stage
#include "Telescope.hpp"          // for Telescope
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "prometheusMetrics.hpp"  // for Counter, MetricFamily
#include "timeUtil.hpp"           // for EOP

using N2::frameID;

enum class N2VarianceMode : uint32_t { CHIMEv1 = 0, EvenOddPosDef = 1 };

std::string N2VarianceMode_to_string(const N2VarianceMode& m);
N2VarianceMode N2VarianceMode_from_string(const std::string& s);
std::ostream& operator<<(std::ostream& os, const N2VarianceMode& m);
std::string format_as(const N2VarianceMode& m);
void to_json(nlohmann::json& j, const N2VarianceMode& m);
void from_json(const nlohmann::json& j, N2VarianceMode& m);


/**
 * @class N2Accumulate
 * @brief Accumulate the high rate GPU output into integrated N2Buffers.
 *
 * This stage accumulates output from the N2k GPU correlator into integrated
 * visibility buffers.
 *
 * ACCUMULATION BINNING STRATEGY
 *
 * The GPU correlation frames are accumulated into bins. The binning strategy is
 * either by "fixed number of frames" (bin_in_ERA false) or by Earth Rotation Angle (ERA, bin_in_ERA
 * true). Both binning strategies ingest frames two at a time, so bins will always contain an even
 * number of frames, and bin boundaries will always occur at an even frame count. If second-stage
 * RFI excision (via the input RFIFrameMask) signals a frame should be dropped, both frames in the
 * pair are dropped.
 *
 * In "fixed number of frames" the beginning of bin 0 is at instrument start (seq = 0),
 * and each bin covers exactly sub_integration_ntime * num_subintegrations_per_bin fpga seq
 * numbers.
 *
 * In "ERA" binning the beginning of bin 0 is when ERA = 0 just before 2000 Jan 1 Noon. Every Earth
 * rotation (sidereal day) is divided into num_bins_per_rotation bins equally sized in ERA, with a
 * bin edge at ERA = 0.
 *
 * This stage will often come online after there is data in the pipeline (because of a node restart,
 * an X-engine restart, or simply a delay between F-engine activation and X-engine).  When started,
 * it will check if the first frame it sees is the start of a bin. If so it will immediately begin
 * accumulating. If not, it will drop frames until it sees the start of a bin, and then begin
 * accumulating. An accumulation ends when the stage sees that the next frame will be in a new bin.
 * At this point the accumulated visibility matrix is output, internal buffers and counters are
 * reset, and a new accumulation begins.
 *
 * The stage output is single-frequency N2FrameViews. At the end of an accumulation the stage loops
 * through the input frequencies, for each one produces an N2FrameView with associated metadata. The
 * frequencies are always released into the output buffer in the order they appear in the input
 * arrays (ie. round robin).  The output stream will see frames in the order: [T0F0 T0F1 T0F2 T1F0
 * T1F1 T1F2 ...]
 *
 * VARIANCE ESTIMATION
 *
 * The output N2FrameViews include the visibility matrix (normalized by the number of good fpga
 * samples in the accumulation) and the `weights`: the reciprocal of the estimated variance of the
 * visibilities.  Because the visibilities may have a linear drift (due to fringes, etc) we cannot
 * use the standard estimator, we want to measure the variance of the visibility apart from linear
 * drift. To accomplish this, both variance estimators currently implemented follow an "even-odd"
 * pattern, building up an estimate of the variance by differencing adjacent samples.  Schematically
 * for a data stream [x0 x1 x2 x3 x4...] both estimators build up a sum like:
 *
 *   S2 = (x1 - x0)^2 + (x3 - x2)^2 + (x5 - x4)^2 + ...
 *
 * Only using even-odd pairs in this manner produces a noisier estimate of the variance, but allows
 * for relatively simple normalization and improves memory bandwidth as frames can be loaded in
 * chunks of 2 and then dropped.
 *
 * The implemented estimators are "CHIMEv1" and "EvenOddPosDef".
 *
 * The "CHIMEv1" estimator is the original estimator used in CHIME. It differences the unnormalized
 * correlations directly, which produces a biased estimate if the number of samples in each frame
 * changes. A bias correction is applied, however, it is not guaranteed the resulting variance value
 * is positive definite.
 *
 * The "EvenOddPosDef" estimator differences normalized visibility samples and accounts for the
 * number of samples in each. It produces an unbiased estimate of the variance and is positive
 * definite, it only produces 0 if the visibilities in each pair are identical or there are no
 * samples in the accumulation bin. On Gaussian data it has the same variance as the CHIMEv1
 * estimator.
 *
 * TODO:    - radiometer_chi2
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
 * @buffer  in_rficounts_buf  RFIcounts buffer, the count of bad (rfimask=0) samples in the RFI mask
 * in each n2k correlation.
 *         @buffer_format   NDArray int32 [num_integrations, num_freq]
 *         @buffer_metadata chordMetadata
 * @buffer  in_plcounts_buf  PLcounts buffer, the count of bad (plmask=0) samples in the PL mask
 * in each n2k correlation.
 *         @buffer_format   NDArray uint64 [num_integrations, num_freq]
 *         @buffer_metadata chordMetadata
 * @buffer  in_rfiframemask_buf  RFIFrameMask buffer, a mask marking specific correlator samples to
 * excise.
 *         @buffer_format   NDArray uint8 [num_integrations, num_freq]
 *         @buffer_metadata chordMetadata
 * @buffer  out_buf         The accumulated and tagged data.
 *      @buffer_format N2Buffer. layout=FullUpperTri, num_ev=0
 *      @buffer_metadata N2Metadata
 *
 * @conf    num_freq_per_n2k_frame          int64_t Number of frequencies in
 *                                          buffers, required.
 * @conf    num_subintegrations_per_bin   int64_t Number of samples (subintegrations)
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
     * @brief   Return a montonic index (counter) for the accumulation bin including seq.
     * Will increase by 1 for each accumulation bin. If bins are smaller than input frames, bins
     * will be skipped on output. The index may or may not begin at 0. Restart-safe.
     *
     * @param   seq   Sequence tick for the moment in question
     *
     * @return  The index value for this accumulation bin.
     */
    int64_t get_accum_abs_bin_idx(uint64_t seq);

    /**
     * @brief   Calculate the accumulation bin index at a given time for ERA-binning.
     * @param   t_inst  Instrument time to calculate the bin index.
     *
     * @return  The index value for this accumulation bin.
     */
    int64_t calculate_ERA_bin_idx_from_time(const timespec& t_inst);

    /**
     * @brief   Return the Earth Orientation Parameters (EOP) for this
     *          accumulation bin. This will be the EOP at the center of the bin.
     *
     * @param   accum_bin_idx   Index for the accumulation bin.
     *
     * @return  The EOP for the center of the bin.
     */
    EOP get_accum_bin_EOP(int64_t accum_bin_idx);

    /**
     * @brief   Check if the given sequence number is the beginning of the given accumulation bin.
     *
     * @param   seq     Sequence number of a visibility sample
     * @param   bin_idx Index of the accumulation bin to check.
     *
     * @return  true if seq is the start of the bin.
     */
    bool is_seq_start_of_bin(uint64_t seq, int64_t bin_idx);

    void accum_corr_and_var(int32_t* vis_f, float* weight_f, const int32_t* corr_t0_f,
                            const int32_t* corr_t1_f, double freq_MHz, EOP& target_eop, EOP& eop_t0,
                            EOP& eop_t1, int32_t count_t0, int32_t count_t1,
                            std::vector<std::complex<float>>& fringe_phase_t0,
                            std::vector<std::complex<float>>& fringe_phase_t1);

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
    bool output_and_reset(frameID& in_frame_id, frameID& in_rfiframemask_frame_id,
                          frameID& out_frame_id);

private:
    // Buffers to read/write
    Buffer* in_buf;              /// Buffer containing input correlations
    Buffer* in_counts_buf;       /// Buffer containing input counts
    Buffer* in_rficounts_buf;    /// Buffer containing input rficounts
    Buffer* in_plcounts_buf;     /// Buffer containing input plcounts
    Buffer* in_rfiframemask_buf; /// Buffer containing input rfiframemask
    Buffer* out_buf;             /// Output for the main vis dataset only

    // Parameters saved from the config files
    const int64_t _num_freq_per_n2k_frame;
    const bool _bin_in_ERA;
    const int64_t _num_subintegrations_per_bin;
    const uint32_t _num_bins_per_rotation;

    const bool _packet_loss_is_scalar;

    const int64_t _n_fpga_samples_per_n2k_frame;
    const int64_t _n_fpga_samples_per_n2k_correlation;
    int64_t _n_integrations_per_n2k_frame;

    const int64_t _num_polarizations; ///< Total number of telescope elements (~2 * num dishes)
    const int64_t _num_dishes;        ///< Total number of telescope elements (~2 * num dishes)
    const int64_t _num_elements;      ///< Total number of telescope elements (~2 * num dishes)

    const int _num_workers; ///< number of OpenMP threads to use to process data

    const bool _do_fringestop; ///< Whether to fringestop
    const ElementOrder _input_order; ///< Order label of the inputs
    const N2VarianceMode _variance_mode;
    const bool _debug_accum_mode;
    const bool _profile_info;


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
    std::vector<float> _var;
    // number of fpga samples, per frequency, in frame
    std::vector<int32_t> _n_valid_fpga_samples_in_vis;
    std::vector<float> _n_valid_sample_diff_sq_sum;
    std::vector<uint64_t> _n_rfi_samples_in_vis;
    std::vector<uint64_t> _n_pl_samples_in_vis;
    int64_t _vis_samples_in_out_frame;
    uint64_t _accum_fpga_start_tick;
    int64_t _accum_bin_idx;


    // The telescope
    const Telescope& _tel;
    
    const std::vector<vec3d_t> _feed_positions_m; ///< The position of each element in the telescope grid frame
    static constexpr std::complex<float> _sentinel_phase = std::complex<float>(2.0f, 2.0f); //  fringestop phases have |z| = 1.0

    // Reference to the prometheus metric that we will use for counting skipped
    // frames
    // TODO ...
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& skipped_frame_counter;
};

#endif
