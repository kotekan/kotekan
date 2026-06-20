#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, GetType
#include "N2Util.hpp"          // for frameID
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, get_chord_metadata, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO
#include "metadata.hpp"        // for metadataObject
#include "n2k_sk_globals.hpp"  // for n2k_globals

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>   // for assert
#include <cstdint>    // for uint64_t, int32_t, int64_t
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

/**
 * @class gpuSimulateRFISK
 *
 * @brief Simulate the external/n2k/src_lib/SkKernel.cu sk_kernel(), as called by
 * lib/cuda/cudaRFISKbar.cpp and lib/cuda/cudaRFISKtilde.cpp
 *
 * Given an input buffer containing S0, S1, and S2 values:
 *
 * S0 = \sum_{t \in t_rfi} PL[t]
 * S1 = \sum_{t \in t_rfi} PL[t] x |E[t]|^2
 * S2 = \sum_{t \in t_rfi} PL[t] x |E[t]|^4
 *
 * where `t_rfi` is a downsampled time variable, covering `downsampling_factor` fast time samples
 * `t`, PL is the packet loss mask, and E is the voltage/electric field.
 *
 * This stage computes an estimator for the bias-corrected spectral kurtosis (SK),
 * its bias (b), and the square root of its variance (sigma).
 *
 * It first computes each of these for the feeds independently, then averages over feeds, then
 * computes an RFI mask from the feed-averaged SK.
 *
 * SK for a single feed is:
 *
 * SK = ((S0 + 1) / (S0 - 1)) * (S0 * S2 / (S1 * S1) - 1) - bias
 *
 * The expected mean of SK for input signals with Gaussian noise is 1.0.  The `bias` and `sigma` are
 * interpolated from tables located in n2k_sk_globals.hpp, which has been copied from `n2k`.
 *
 * Where possible calculations here are done in double precision, when comparing output with n2k
 * the results should agree to at least 10^{-6}.
 *
 * For the single-feed results, computation only proceeds if the fraction of good counts
 * (S0 / rfi_total_downsampling_factor) is greater than `rfi_single_feed_min_good_frac` and the
 * average signal (S1 / S0 = <|E^2|>) is in `[rfi_mu_min, rfi_mu_max]`. Otherwise
 * SK, bias, sigma = (0, 0, -1) is written.
 *
 * For the feed-averaged results, single feeds results are only included if they passed the
 * single feed threshold, if the bad feed mask (bf_mask) is true, and the fraction of good samples
 * (Sum_e(S0[e]) / (rfi_total_downsampling_factor * num_elements)) is greater than
 * rfi_feed_averaged_min_good frac.  Feed-averaged values are weighed by the number of samples.
 *
 * The RFI mask is 1 (good) if the feed-averaged SK is within `rfi_sk_rfimask_sigmas`
 * feed-averaged sigmas of 1.0, otherwise the RFI mask is 0 (bad).
 *
 * This stage produces buffers compatible with either cudaRFISKtilde or cudaRFISKbar.
 *
 * @par Buffers
 * @buffer in_rfi_s012_buf      Buffer of input S012 values
 *         @buffer_format uint64
 *         @buffer_shape [samples_per_data_set/rfi_total_downsampling_factor, num_local_freq, 3,
 *              num_polarizations, num_dishes]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = rfi_total_downsampling_factor
 * @buffer in_bf_mask_buf       The bad feed mask
 *         @buffer_format uint8
 *         @buffer_shape [num_polarizations, num_dishes]
 *         @buffer_metadata chordMetadata
 * @buffer out_rfi_sk_buf       Buffer of single feed (SK, bias, sigma) values
 *         @buffer_format float32
 *         @buffer_shape [samples_per_data_set/rfi_total_downsampling_factor, num_local_freq, 3,
 *              num_polarizations, num_dishes]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = rfi_total_downsampling_factor
 * @buffer out_rfi_sktilde_buf  Buffer of feed-averaged (SK, bias, sigma) values
 *         @buffer_format float32
 *         @buffer_shape [samples_per_data_set/rfi_total_downsampling_factor, num_local_freq, 3]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = rfi_total_downsampling_factor
 * @buffer out_rfi_mask_buf     The RFI mask. A bit mask.
 *         @buffer_format uint1x8
 *         @buffer_shape    As a uint8: [samples_per_data_set/1024, num_local_freq, 128]
 *                          As a uint1: [samples_per_data_set/1024, num_local_freq, 1024]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = 1024
 *
 * @conf  num_polarizations     Int. Number of polarizations.
 * @conf  num_dishes            Int. Number of dishes.
 * @conf  num_local_freq        Int. Number of frequencies.
 * @conf  samples_per_data_set  Int. Number of fast time samples per buffer.
 * @conf  rfi_total_downsampling_factor   Int. Number of fast times `t` in a `t_rfi`. Must divide
 *                              `samples_per_data_set`. Output SK buffers will contain
 *                              `samples_per_data_set` / `downsampling_factor` time samples.
 * @conf  bar_mode              Bool. Whether to output an SKbar buffer or SK buffer.
 * @conf  rfi_sk_rfimask_sigmas             Double. SK sigma threshold for RFI mask.
 * @conf  rfi_single_feed_min_good_frac     Double. Threshold of good sample coverage for inclusion
 *                                          in single feed SK calculation.
 * @conf  rfi_feed_averaged_min_good_frac   Double. Threshold on good sample coverage for inclusion
 *                                          in feed-averaged SK calculation.
 * @conf  rfi_mu_min                        Double. Lower threshold on <|E^2|> for inclusion
 *                                          in single feed SK calculation.
 * @conf  rfi_mu_max                        Double. Upper threshold on <|E^2|> for inclusion
 *                                          in single feed SK calculation.
 *
 * @author Geoffrey Ryan
 */
class gpuSimulateRFISK : public kotekan::Stage {
public:
    gpuSimulateRFISK(Config& config, const std::string& unique_name,
                     bufferContainer& buffer_container);
    ~gpuSimulateRFISK();
    void main_thread() override;

    double get_bias(double x, double y);
    double get_sigma(double x, uint64_t s0);
    float cubic_interpolate(float x, float y0, float y1, float y2, float y3);
    double nice_cubic_interpolate(double x, double x0, double dx, double y0, double y1, double y2,
                                  double y3);
    bool test_cubic_interpolate();

private:
    const int64_t _num_polarizations;
    const int64_t _num_dishes;
    const int64_t _num_elements;
    const int64_t _num_local_freq;
    const int64_t _samples_per_data_set;
    const int64_t _rfi_downsampling_factor;
    const bool _bar_mode;
    const double _rfi_sk_rfimask_sigmas;
    const double _rfi_single_feed_min_good_frac;
    const double _rfi_feed_averaged_min_good_frac;
    const double _rfi_mu_min;
    const double _rfi_mu_max;
    /// Lower x limit in bias & sigma tables
    const float _xmin;
    /// Upper x limit in bias & sigma tables
    const float _xmax;
    /// Number of x samples in bias table.
    const uint64_t _bias_nx;
    /// Number of y samples in bias table.
    const uint64_t _bias_ny;
    /// Number of x samples in sigma table.
    const uint64_t _sigma_nx;

    /// bias table
    std::vector<float> _bias_coeffs;
    /// sigma table
    std::vector<float> _sigma_coeffs;

    Buffer* in_bf_mask_buf;
    Buffer* in_rfi_s012_buf;
    Buffer* out_rfi_sk_buf;
    Buffer* out_rfi_sktilde_buf;
    Buffer* out_rfi_mask_buf;
};

using kotekan::Stage;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(gpuSimulateRFISK);

gpuSimulateRFISK::gpuSimulateRFISK(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulateRFISK::main_thread, this)),
    _num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    _num_elements(_num_polarizations * _num_dishes),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_total_downsampling_factor")),
    _bar_mode(config.get<bool>(unique_name, "bar_mode")),
    _rfi_sk_rfimask_sigmas(config.get<double>(unique_name, "rfi_sk_rfimask_sigmas")),
    _rfi_single_feed_min_good_frac(
        config.get<double>(unique_name, "rfi_single_feed_min_good_frac")),
    _rfi_feed_averaged_min_good_frac(
        config.get<double>(unique_name, "rfi_feed_averaged_min_good_frac")),
    _rfi_mu_min(config.get<double>(unique_name, "rfi_mu_min")),
    _rfi_mu_max(config.get<double>(unique_name, "rfi_mu_max")), _xmin(n2k_globals::xmin),
    _xmax(n2k_globals::xmax), _bias_nx(n2k_globals::bias_nx), _bias_ny(n2k_globals::bias_ny),
    _sigma_nx(n2k_globals::sigma_nx), _bias_coeffs(std::vector<float>()),
    _sigma_coeffs(std::vector<float>()) {

    // Grab Buffers
    in_rfi_s012_buf = get_buffer("in_rfi_s012_buf");
    in_rfi_s012_buf->register_consumer(unique_name);

    in_bf_mask_buf = get_buffer("in_bf_mask_buf");
    in_bf_mask_buf->register_consumer(unique_name);

    out_rfi_sk_buf = get_buffer("out_rfi_sk_buf");
    out_rfi_sk_buf->register_producer(unique_name);

    out_rfi_sktilde_buf = get_buffer("out_rfi_sktilde_buf");
    out_rfi_sktilde_buf->register_producer(unique_name);

    out_rfi_mask_buf = get_buffer("out_rfi_mask_buf");
    out_rfi_mask_buf->register_producer(unique_name);

    int64_t nt = _samples_per_data_set / _rfi_downsampling_factor;
    size_t bf_mask_size = _num_elements;
    size_t rfi_s012_size = nt * _num_local_freq * 3 * _num_elements * sizeof(uint64_t);

    // Check input sizes and buffer compatibility
    if (_samples_per_data_set % _rfi_downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of rfi_downsampling_factor");
    }
    assert(_samples_per_data_set % _rfi_downsampling_factor == 0);

    if (in_rfi_s012_buf->frame_size != rfi_s012_size) {
        FATAL_ERROR("in_rfi_s012_buf ({:s}) has frame size: {:d}, expected: {:d}",
                    in_rfi_s012_buf->buffer_name, in_rfi_s012_buf->frame_size, rfi_s012_size);
    }
    assert(in_rfi_s012_buf->frame_size == rfi_s012_size);

    if (in_bf_mask_buf->frame_size != bf_mask_size) {
        FATAL_ERROR("in_bf_mask_buf ({:s}) has frame size: {:d}, expected: {:d}",
                    in_bf_mask_buf->buffer_name, in_bf_mask_buf->frame_size, bf_mask_size);
    }
    assert(in_bf_mask_buf->frame_size == bf_mask_size);

    // Make frame desc for produced buffers
    if (_bar_mode) {
        out_rfi_sk_buf->allocate_ndarray_frame_desc<float, 5>(
            "SKbar", {nt, _num_local_freq, 3, _num_polarizations, _num_dishes},
            {"Trfibar", "F", "SK", "P", "D"});
        out_rfi_sktilde_buf->allocate_ndarray_frame_desc<float, 3>(
            "SKbartilde", {nt, _num_local_freq, 3}, {"Trfibar", "F", "SK"});
        out_rfi_mask_buf->allocate_ndarray_frame_desc<kotekan::uint1x8_t, 3>(
            "RFImask", {_samples_per_data_set / 1024, _num_local_freq, 128},
            {"T8hi128", "F", "T8lo128"});
    } else {
        out_rfi_sk_buf->allocate_ndarray_frame_desc<float, 5>(
            "SK", {nt, _num_local_freq, 3, _num_polarizations, _num_dishes},
            {"Trfi", "F", "SK", "P", "D"});
        out_rfi_sktilde_buf->allocate_ndarray_frame_desc<float, 3>(
            "SKtilde", {nt, _num_local_freq, 3}, {"Trfi", "F", "SK"});
        out_rfi_mask_buf->allocate_ndarray_frame_desc<kotekan::uint1x8_t, 3>(
            "RFImask", {_samples_per_data_set / 1024, _num_local_freq, 128},
            {"T8hi128", "F", "T8lo128"});
    }

    // Check the size of the interpolation tables
    if (sizeof(n2k_globals::bsigma_coeffs) != (_bias_nx * _bias_ny + _sigma_nx) * sizeof(double)) {
        FATAL_ERROR("lib/testing/n2k_globals.hpp bsigma_coeffs is {:d} bytes."
                    "Expected ({:d} x {:d} + {:d}) x {:d} = {:d} bytes.",
                    sizeof(n2k_globals::bsigma_coeffs), _bias_nx, _bias_ny, _sigma_nx,
                    sizeof(double), (_bias_nx * _bias_ny + _sigma_nx) * sizeof(double));
    }
    assert(sizeof(n2k_globals::bsigma_coeffs)
           == (_bias_nx * _bias_ny + _sigma_nx) * sizeof(double));

    // Copy the n2k_globals::bsigma_coeffs into local arrays, converting to floats
    // as we go.
    uint64_t num_bias = _bias_nx * _bias_ny;
    for (uint64_t idx = 0; idx < num_bias; idx++)
        _bias_coeffs.push_back(static_cast<float>(n2k_globals::bsigma_coeffs[idx]));

    for (uint64_t idx = 0; idx < _sigma_nx; idx++)
        _sigma_coeffs.push_back(static_cast<float>(n2k_globals::bsigma_coeffs[num_bias + idx]));

    // Run a quick self test on the cubic interpolation routine.
    bool interpolate_self_test = test_cubic_interpolate();
    if (!interpolate_self_test)
        FATAL_ERROR("test_cubic_interpolate self test failed.");
    assert(interpolate_self_test);
}

gpuSimulateRFISK::~gpuSimulateRFISK() {}

void gpuSimulateRFISK::main_thread() {

    frameID in_bf_mask_frame_id(in_bf_mask_buf);
    frameID in_rfi_s012_frame_id(in_rfi_s012_buf);
    frameID out_rfi_sk_frame_id(out_rfi_sktilde_buf);
    frameID out_rfi_sktilde_frame_id(out_rfi_sktilde_buf);
    frameID out_rfi_mask_frame_id(out_rfi_mask_buf);

    while (!stop_thread) {
        uint8_t* bf_mask =
            (uint8_t*)in_bf_mask_buf->wait_for_full_frame(unique_name, in_bf_mask_frame_id);
        if (bf_mask == nullptr)
            break;
        uint64_t* rfi_s012 =
            (uint64_t*)in_rfi_s012_buf->wait_for_full_frame(unique_name, in_rfi_s012_frame_id);
        if (rfi_s012 == nullptr)
            break;
        float* rfi_sk =
            (float*)out_rfi_sk_buf->wait_for_empty_frame(unique_name, out_rfi_sk_frame_id);
        if (rfi_sk == nullptr)
            break;
        float* rfi_sktilde = (float*)out_rfi_sktilde_buf->wait_for_empty_frame(
            unique_name, out_rfi_sktilde_frame_id);
        if (rfi_sktilde == nullptr)
            break;
        uint8_t* rfi_mask =
            (uint8_t*)out_rfi_mask_buf->wait_for_empty_frame(unique_name, out_rfi_mask_frame_id);
        if (rfi_mask == nullptr)
            break;

        INFO("Simulating GPU RFI SK for {:s}[{:d}], {:s}[{:d}] and putting result in {:s}[{:d}], "
             "{:s}[{:d}], {:s}[{:d}]",
             in_bf_mask_buf->buffer_name, in_bf_mask_frame_id, in_rfi_s012_buf->buffer_name,
             in_rfi_s012_frame_id, out_rfi_sk_buf->buffer_name, out_rfi_sk_frame_id,
             out_rfi_sktilde_buf->buffer_name, out_rfi_sktilde_frame_id,
             out_rfi_mask_buf->buffer_name, out_rfi_mask_frame_id);

        // number of elements = number of dishes * polarizations
        uint64_t nt = _samples_per_data_set;
        uint64_t nf = _num_local_freq;
        uint64_t ne = _num_elements;

        uint64_t nt_rfi = nt / _rfi_downsampling_factor;
        // uint64_t nsub = _rfi_downsampling_factor;

        // array access strides in S012
        uint64_t sstride_s012 = ne;
        uint64_t fstride_s012 = 3 * ne;
        uint64_t tstride_s012 = nf * 3 * ne;

        // array access strides in SK
        uint64_t sstride_sk = ne;
        uint64_t fstride_sk = 3 * ne;
        uint64_t tstride_sk = nf * 3 * ne;

        // array access strides in SKtilde
        uint64_t fstride_sktilde = 3;
        uint64_t tstride_sktilde = nf * 3;

        // Set to 0 to start.
        for (uint64_t tfse = 0; tfse < tstride_sk * nt_rfi; tfse++)
            rfi_sk[tfse] = 0;
        for (uint64_t tfs = 0; tfs < tstride_sktilde * nt_rfi; tfs++)
            rfi_sktilde[tfs] = 0;
        for (uint64_t tf = 0; tf < nt * nf / 8; tf++)
            rfi_mask[tf] = 0;

        // Looping over entries in the rfi buffer.
        for (uint64_t t_rfi = 0; t_rfi < nt_rfi; t_rfi++) {

            for (uint64_t f = 0; f < nf; f++) {

                uint64_t s012_idx = f * fstride_s012 + t_rfi * tstride_s012;
                uint64_t sk_idx = f * fstride_sk + t_rfi * tstride_sk;
                uint64_t sktilde_idx = f * fstride_sktilde + t_rfi * tstride_sktilde;

                // First, loop through elements and calculate SK, bias, sigma for each.
                for (uint64_t e = 0; e < ne; e++) {

                    // Grab single-feed inputs
                    uint64_t s0 = rfi_s012[s012_idx + e];
                    uint64_t s1 = rfi_s012[s012_idx + sstride_s012 + e];
                    uint64_t s2 = rfi_s012[s012_idx + 2 * sstride_s012 + e];

                    // Check if within thresholds. If not, write dummy values and move on.
                    if (s0 < _rfi_downsampling_factor * _rfi_single_feed_min_good_frac
                        || s1 < _rfi_mu_min * s0 || s1 > _rfi_mu_max * s0) {
                        rfi_sk[sk_idx + 0 * sstride_sk + e] = 0.0f;
                        rfi_sk[sk_idx + 1 * sstride_sk + e] = 0.0f;
                        rfi_sk[sk_idx + 2 * sstride_sk + e] = -1.0f;
                        continue;
                    }

                    // SK = ((s0 + 1) / (s0 - 1)) * (s0 * s2 / s1^2 - 1)
                    // Compute numerator and denominator of uncorrected SK,
                    // do subtraction in ints for precision.
                    double sk_num = (s0 + 1) * static_cast<double>(s0 * s2 - s1 * s1);
                    double sk_den = (s0 - 1) * static_cast<double>(s1 * s1);

                    // compute x = log(<|E^s|>), y = 1/s0 for table interpolation.
                    double y = 1.0 / static_cast<double>(s0); // = 1/n
                    double mu = static_cast<double>(s1) / s0; // = s1f/n = <|E|^2>
                    double x = std::clamp(log(mu), (double)_xmin, (double)_xmax);

                    // Get the bias and sigma (via table interpolation)
                    double bias = get_bias(x, y);
                    double sigma = get_sigma(x, s0);

                    // Write the values
                    rfi_sk[sk_idx + 0 * sstride_sk + e] = sk_num / sk_den - bias;
                    rfi_sk[sk_idx + 1 * sstride_sk + e] = bias;
                    rfi_sk[sk_idx + 2 * sstride_sk + e] = sigma;
                } // e

                // Now we're calculating the feed-averaged values.

                // Accumulation variables. Feed-averaged values are weighted by good counts (S0)
                uint64_t n = 0;
                double w_sk_tilde = 0;
                double w_bias_tilde = 0;
                double w2_sigma2_tilde = 0.0; // Accumulate variance = sigma^2 instead of sigma.

                // Loop over elements, accumulate accordingly.
                for (uint64_t e = 0; e < ne; e++) {
                    // single feed values for this element.
                    float sk = rfi_sk[sk_idx + 0 * sstride_sk + e];
                    float bias = rfi_sk[sk_idx + 1 * sstride_sk + e];
                    float sigma = rfi_sk[sk_idx + 2 * sstride_sk + e];

                    // Only include if BF is good AND single feed passed the threshold
                    if (bf_mask[e] && sigma > 0.0) {
                        uint64_t s0 = rfi_s012[s012_idx + e];
                        n += s0;
                        w_sk_tilde += s0 * sk;
                        w_bias_tilde += s0 * bias;
                        // variance gets weighted by N^2 instead of N.
                        w2_sigma2_tilde += s0 * s0 * sigma * sigma;
                    }
                } // e


                // Now we can assign the feed-averaged SK values.  While we're at it we
                // do the check for if this sample should be RFI masked or not.

                // By default, sample is bad.
                bool rfi_good = false;

                // Make sure we pass the threshold, if not, write the dummy values.
                if (n < _rfi_feed_averaged_min_good_frac * _rfi_downsampling_factor * ne) {
                    rfi_sktilde[sktilde_idx + 0] = 0.0f;
                    rfi_sktilde[sktilde_idx + 1] = 0.0f;
                    rfi_sktilde[sktilde_idx + 2] = -1.0f;
                } else {

                    // We're good! Write the feed-averaged sk, bias, and sigma.

                    double sk = w_sk_tilde / n;
                    double sigma = sqrt(w2_sigma2_tilde) / n;

                    rfi_sktilde[sktilde_idx + 0] = sk;
                    rfi_sktilde[sktilde_idx + 1] = w_bias_tilde / n;
                    rfi_sktilde[sktilde_idx + 2] = sigma;

                    // While we have the values, check if sk is within the sigma threshold of 1.0.
                    if (sk >= 1.0 - _rfi_sk_rfimask_sigmas * sigma
                        && sk <= 1.0 + _rfi_sk_rfimask_sigmas * sigma)
                        rfi_good = true;
                }

                // Finally, we can write the RFI mask value.

                // RFI mask access strides.  The RFI mask has time split into a slow (t / 1024)
                // and fast (t % 1024) index.  This index addresses a BIT, but the strides are
                // in BYTES, so the fast stride is 128 bytes = 1024 bits.
                uint64_t fstride_rfimask = 128; // = 1024 / 8
                uint64_t thistride_rfimask = nf * 128;

                // Write the RFI mask.  The RFI mask has values for *every* time sample,
                // so this value is copied along rfi_downsampling_factor time samples.
                for (uint64_t t = t_rfi * _rfi_downsampling_factor;
                     t < (t_rfi + 1) * _rfi_downsampling_factor; t++) {

                    // Split the time into fast and slow, bytes, and bits.
                    uint64_t t_hi = t / 1024;
                    uint64_t t_lo = t % 1024;
                    uint64_t t_lo_byte = t_lo / 8;
                    uint8_t t_lo_bit = t_lo % 8;

                    // Calculate the byte index we're writing to.
                    uint64_t rfimask_idx =
                        t_hi * thistride_rfimask + f * fstride_rfimask + t_lo_byte;

                    // Write the bit!
                    if (rfi_good) {
                        rfi_mask[rfimask_idx] |= (1u << t_lo_bit);
                    } else {
                        rfi_mask[rfimask_idx] |= (0u << t_lo_bit);
                    }

                } // t

            } // f
        } // t_rfi

        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in =
            in_rfi_s012_buf->get_metadata(in_rfi_s012_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", in_rfi_s012_buf->buffer_name,
                        in_rfi_s012_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        in_rfi_s012_buf->buffer_name, in_rfi_s012_frame_id);
        }
        assert(metadata_is_chord(mc_in));

        const std::shared_ptr<const chordMetadata> meta_in = get_chord_metadata(mc_in);

        // Create output SK metadata
        out_rfi_sk_buf->allocate_new_metadata_object(out_rfi_sk_frame_id);
        const std::shared_ptr<metadataObject> mc_sk =
            out_rfi_sk_buf->get_metadata(out_rfi_sk_frame_id);
        if (!mc_sk) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata",
                        out_rfi_sk_buf->buffer_name, out_rfi_sk_frame_id);
        }
        assert(mc_sk);
        if (!metadata_is_chord(mc_sk)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        out_rfi_sk_buf->buffer_name, out_rfi_sk_frame_id);
        }
        assert(metadata_is_chord(mc_sk));
        const std::shared_ptr<chordMetadata> meta_sk = get_chord_metadata(mc_sk);
        assert(meta_sk);

        // Create output SKtilde metadata
        out_rfi_sktilde_buf->allocate_new_metadata_object(out_rfi_sktilde_frame_id);
        const std::shared_ptr<metadataObject> mc_sktilde =
            out_rfi_sktilde_buf->get_metadata(out_rfi_sktilde_frame_id);
        if (!mc_sktilde) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata",
                        out_rfi_sktilde_buf->buffer_name, out_rfi_sktilde_frame_id);
        }
        assert(mc_sktilde);
        if (!metadata_is_chord(mc_sktilde)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        out_rfi_sktilde_buf->buffer_name, out_rfi_sktilde_frame_id);
        }
        assert(metadata_is_chord(mc_sktilde));
        const std::shared_ptr<chordMetadata> meta_sktilde = get_chord_metadata(mc_sktilde);
        assert(meta_sktilde);

        // Create output rfi mask metadata
        out_rfi_mask_buf->allocate_new_metadata_object(out_rfi_mask_frame_id);
        const std::shared_ptr<metadataObject> mc_rfi_mask =
            out_rfi_mask_buf->get_metadata(out_rfi_mask_frame_id);
        if (!mc_rfi_mask) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata",
                        out_rfi_mask_buf->buffer_name, out_rfi_mask_frame_id);
        }
        assert(mc_rfi_mask);
        if (!metadata_is_chord(mc_rfi_mask)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        out_rfi_mask_buf->buffer_name, out_rfi_mask_frame_id);
        }
        assert(metadata_is_chord(mc_rfi_mask));
        const std::shared_ptr<chordMetadata> meta_rfi_mask = get_chord_metadata(mc_rfi_mask);
        assert(meta_rfi_mask);

        // Start with a copy
        meta_sk->deepCopy(meta_in);
        meta_sktilde->deepCopy(meta_in);
        meta_rfi_mask->deepCopy(meta_in);

        meta_sk->set_from_frame_desc(out_rfi_sk_buf->get_ndarray_frame_desc());
        meta_sktilde->set_from_frame_desc(out_rfi_sktilde_buf->get_ndarray_frame_desc());
        meta_rfi_mask->set_from_frame_desc(out_rfi_mask_buf->get_ndarray_frame_desc());

        // test that things are consistent
        meta_sk->check_frame_desc(out_rfi_sk_buf->get_ndarray_frame_desc());
        meta_sktilde->check_frame_desc(out_rfi_sktilde_buf->get_ndarray_frame_desc());
        meta_rfi_mask->check_frame_desc(out_rfi_mask_buf->get_ndarray_frame_desc());

        // Set non-NDArray things.
        meta_rfi_mask->set_time_downsampling_fpga(1024 * meta_in->get_time_downsampling_fpga()
                                                  / _rfi_downsampling_factor);

        INFO("Simulating GPU RFI SK done for {:s}[{:d}]+{:s}[{:d}] result is in "
             "{:s}[{:d}]+{:s}[{:d}]+{:s}[{:d}]",
             in_rfi_s012_buf->buffer_name, in_rfi_s012_frame_id, in_bf_mask_buf->buffer_name,
             in_bf_mask_frame_id, out_rfi_sk_buf->buffer_name, out_rfi_sk_frame_id,
             out_rfi_sktilde_buf->buffer_name, out_rfi_sktilde_frame_id,
             out_rfi_mask_buf->buffer_name, out_rfi_mask_frame_id);

        in_rfi_s012_buf->mark_frame_empty(unique_name, in_rfi_s012_frame_id++);
        out_rfi_sk_buf->mark_frame_full(unique_name, out_rfi_sk_frame_id++);
        out_rfi_sktilde_buf->mark_frame_full(unique_name, out_rfi_sktilde_frame_id++);
        out_rfi_mask_buf->mark_frame_full(unique_name, out_rfi_mask_frame_id++);
    } // while !stop_thread
}

double gpuSimulateRFISK::get_bias(double x, double y) {
    double dx = (_xmax - _xmin) / ((double)(_bias_nx - 1));

    // index of bin in bias table containing x, unless x is near the end
    // of the bin. Leaves at least 1 free bin to the left and at least two
    // free bins to the right.
    uint64_t bin_idx = std::clamp(static_cast<uint64_t>((x - _xmin) / dx), uint64_t{1}, _bias_nx - 3);
    // Index of first bin (or first table point) in the reconstruction.
    // Probably one to the left of where the sample point is.
    uint64_t i0 = bin_idx - 1;
    // x-value of point at i0.
    double x0 = _xmin + dx * i0;

    // We will use the table rows i0, i0+1, i0+2, i0+3.
    // Each row is a list of coefficients for a polynomial in y.

    double c[4] = {0.0};

    // Evaluate the polynomials in y for each row we need.
    for (uint64_t i = 0; i < 4; i++) {
        double yp = 1.0;
        for (uint64_t j = 0; j < _bias_ny; j++) {
            c[i] += _bias_coeffs[(i + i0) * _bias_ny + j] * yp;
            yp *= y;
        }
    }

    // Now we can interpolate
    return nice_cubic_interpolate(x, x0, dx, c[0], c[1], c[2], c[3]);
}

double gpuSimulateRFISK::get_sigma(double x, uint64_t s0) {
    double dx = (_xmax - _xmin) / ((double)(_sigma_nx - 1));

    // index of bin in sigma table containing x, unless x is near the end
    // of the bin. Leaves at least 1 free bin to the left and at least two
    // free bins to the right.
    uint64_t bin_idx = std::clamp(static_cast<uint64_t>((x - _xmin) / dx), uint64_t{1}, _sigma_nx - 3);
    // Index of first bin (or first table point) in the reconstruction.
    // Probably one to the left of where the sample point is.
    uint64_t i0 = bin_idx - 1;
    // x-value of point at i0.
    double x0 = _xmin + dx * i0;

    // table interpolation for sigma. Returns sigma * sqrt(s0)
    double sigma_tab = nice_cubic_interpolate(x, x0, dx, _sigma_coeffs[i0], _sigma_coeffs[i0 + 1],
                                              _sigma_coeffs[i0 + 2], _sigma_coeffs[i0 + 3]);
    double sigma = sigma_tab / sqrt(s0);

    return sigma;
}

/*
 * Copied straight from external/n2k/include/n2k/internals/interpolation.hpp
 */
float gpuSimulateRFISK::cubic_interpolate(float x, float y0, float y1, float y2, float y3) {

    constexpr float half = 1.0f / 2.0f;
    constexpr float third = 1.0f / 3.0f;

    float d01 = x * (y1 - y0);
    float d12 = (x - 1) * (y2 - y1);

    float c12 = x * (y2 - y1);
    float c23 = (x - 1) * (y3 - y2);

    float d012 = half * (x - 1) * (c12 - d01);
    float c123 = half * x * (c23 - d12);

    float c0123 = third * (x + 1) * (c123 - d012);

    return c0123 + d012 + c12 + y1;
}

double gpuSimulateRFISK::nice_cubic_interpolate(double x, double x0, double dx, double y0,
                                                double y1, double y2, double y3) {

    double x1 = x0 + dx;
    double x2 = x0 + 2 * dx;
    double x3 = x0 + 3 * dx;

    // Interpolating polynomials.
    // pN = 1 if x = xN and 0 if x = xM (M != N)
    double p0 = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3));
    double p1 = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3));
    double p2 = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3));
    double p3 = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2));

    return p0 * y0 + p1 * y1 + p2 * y2 + p3 * y3;
}

bool gpuSimulateRFISK::test_cubic_interpolate() {

    // y = x^3
    float y0 = -1.0f;
    float y1 = 0.0f;
    float y2 = 1.0f;
    float y3 = 8.0f;

    bool good = true;

    float atol = 5.0e-7;
    float rtol = 5.0e-7;

    std::vector<std::array<float, 2>> tests = {{-1.0f, y0},    {0.0f, y1},       {1.0f, y2},
                                               {2.0f, y3},     {-0.5f, -0.125f}, {0.5f, 0.125f},
                                               {1.5f, 3.375f}, {3.0f, 27.0f}};

    for (auto xy : tests) {
        float y = cubic_interpolate(xy[0], y0, y1, y2, y3);

        float max_y = std::max(fabs(y), fabs(xy[1]));

        if (fabs(y - xy[1]) > atol + max_y * rtol) {
            ERROR("cubic_interpolate({:f}, {:f}, {:f}, {:f}, {:f}) = {:f}, expected {:f}", xy[0],
                  y0, y1, y2, y3, y, xy[1]);
            good = false;
        }
    }

    return good;
}
