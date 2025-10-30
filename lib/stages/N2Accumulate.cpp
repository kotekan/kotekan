#include "N2Accumulate.hpp"

#include "CHORDTelescope.hpp"    // for CHORDTelescope
#include "Config.hpp"            // for Config
#include "N2FrameView.hpp"       // for N2FrameView
#include "N2Metadata.hpp"        // for alloc_N2_from_chord_metadata, N2Metadata
#include "N2Util.hpp"            // for frameID, modulo, ts_to_uint64, cmap, get_num_prod, cfloat
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"    // for DEBUG, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge

#include "fmt.hpp"      // for compile_string_to_view
#include "gsl-lite.hpp" // for span

#include <algorithm>  // for fill, copy
#include <complex>    // for norm, operator*, complex
#include <functional> // for bind, function, placeholders
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <sys/time.h> // for TIMEVAL_TO_TIMESPEC
#include <time.h>     // for size_t, timespec, timespec_get, TIME_UTC
#include <vector>     // for vector


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(N2Accumulate);


N2Accumulate::N2Accumulate(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2Accumulate::main_thread, this)),
    _tel(Telescope::instance().cast<CHORDTelescope>()),
    skipped_frame_counter(Metrics::instance().add_counter(
        "kotekan_N2accumulate_skipped_frame_total", unique_name, {"freq_id", "reason"})) {

    // Fetch configuration

    // number of frequencies in frame
    _num_freq_per_n2k_frame = config.get<int64_t>(unique_name, "num_freq_per_n2k_frame");
    assert(_num_freq_per_n2k_frame > 0);
    if (_num_freq_per_n2k_frame <= 0) {
        FATAL_ERROR("num_freq_per_n2k_frame is not positive: {:d}", _num_freq_per_n2k_frame);
        std::abort();
    }

    // accumulation setup
    _num_n2k_samples_to_accumulate =
        config.get<int64_t>(unique_name, "num_n2k_samples_to_accumulate");

    if (_num_n2k_samples_to_accumulate <= 0 || _num_n2k_samples_to_accumulate % 2 != 0) {
        FATAL_ERROR("N2Accumulate configured to use non-positive or odd "
                    "num_n2k_samples_to_accumulate: {:d}",
                    _num_n2k_samples_to_accumulate);
        std::abort();
    }

    _packet_loss_is_scalar = config.get<bool>(unique_name, "packet_loss_is_scalar");
    if (!_packet_loss_is_scalar)
        FATAL_ERROR("N2Accumulate configured to use packet loss matrix, which is not implemented.");
    assert(_packet_loss_is_scalar);

    // sampling information
    _n_fpga_samples_per_n2k_frame = config.get<int64_t>(unique_name, "samples_per_data_set");
    _n_fpga_samples_per_n2k_correlation = config.get<int64_t>(unique_name, "sub_integration_ntime");

    if (!(_n_fpga_samples_per_n2k_frame > 0)) {
        FATAL_ERROR("samples_per_data_set is not positve: {:d}", _n_fpga_samples_per_n2k_frame);
        std::abort();
    }
    if (!(_n_fpga_samples_per_n2k_correlation > 0)) {
        FATAL_ERROR("sub_integration_ntime is not positve: {:d}",
                    _n_fpga_samples_per_n2k_correlation);
        std::abort();
    }
    if (!(_n_fpga_samples_per_n2k_frame % _n_fpga_samples_per_n2k_correlation == 0)) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d})",
                    _n_fpga_samples_per_n2k_frame, _n_fpga_samples_per_n2k_correlation);
        std::abort();
    }

    _n_integrations_per_n2k_frame =
        _n_fpga_samples_per_n2k_frame / _n_fpga_samples_per_n2k_correlation;


    // Number of products sent by the GPU

    // Number of elements (polarization x dish) in the array.
    _num_elements = config.get<int64_t>(unique_name, "num_elements");
    if (!(_num_elements > 0)) {
        FATAL_ERROR("num_elements is not positive: {:d}", _num_elements);
        std::abort();
    }
    // Check num_elements is consistent with the count & correlation blocksizes.
    if (!(_num_elements % _n2k_correlation_blocksize == 0)) {
        FATAL_ERROR("num_elements ({:d}) is not a multiple of the correlation block size ({:d})",
                    _num_elements, _n2k_correlation_blocksize);
        std::abort();
    }
    if (!(_num_elements % (8 * _n2k_counts_blocksize) == 0)) {
        FATAL_ERROR("num_elements ({:d}) / 8 is not a multiple of the counts block size ({:d})",
                    _num_elements, _n2k_counts_blocksize);
        std::abort();
    }

    // sizes for blocked input correlation matrix
    _n2k_correlation_lin_blocks = _num_elements / _n2k_correlation_blocksize;
    _n2k_correlation_num_blocks =
        (_n2k_correlation_lin_blocks * (_n2k_correlation_lin_blocks + 1)) / 2;
    // Total number of correlation values per time & frequency,
    // because of blocking this will include some redundant values.
    _n2k_correlation_num_products =
        _n2k_correlation_num_blocks * _n2k_correlation_blocksize * _n2k_correlation_blocksize;

    // sizes for blocked input counts matrix
    _n2k_counts_lin_blocks = _num_elements / (8 * _n2k_counts_blocksize);
    _n2k_counts_num_blocks = (_n2k_counts_lin_blocks * (_n2k_counts_lin_blocks + 1)) / 2;
    // Total number of counts values per time & frequency,
    // because of blocking this will include some redundant values.
    _n2k_counts_num_products =
        _n2k_counts_num_blocks * _n2k_counts_blocksize * _n2k_counts_blocksize;

    // Total number of visibilities per time & frequency in output.
    _N2_num_products = (_num_elements * (_num_elements + 1)) / 2;

    _rfi_downsampling_factor = config.get<int64_t>(unique_name, "rfi_downsampling_factor");
    if (!(_rfi_downsampling_factor > 0)) {
        FATAL_ERROR("rfi_downsampling_factor is not positive: {:d}", _rfi_downsampling_factor);
        std::abort();
    }
    if (!(_rfi_downsampling_factor % 8 == 0)) {
        FATAL_ERROR("rfi_downsampling_factor is not a multiple of 8: {:d}",
                    _rfi_downsampling_factor);
        std::abort();
    }

    // Initializing these here using the computed _n2k_correlation_num_products *
    // _num_freq_per_n2k_frame (accumulate the full, blocked matrix x frequencies from the GPU)
    _vis = std::vector<int32_t>(2 * _num_freq_per_n2k_frame * _n2k_correlation_num_products,
                                0); // vis with complex as 2 ints
    _vis_even = std::vector<int32_t>(2 * _num_freq_per_n2k_frame * _n2k_correlation_num_products,
                                     0); // store even vis matrix for weights calculation

    _weights = std::vector<float>(_num_freq_per_n2k_frame * _n2k_correlation_num_products,
                                  0.0f); // real-valued weights
    // number of fpga samples, per frequency, in frame
    _n_valid_fpga_samples_in_vis = std::vector<int32_t>(_num_freq_per_n2k_frame, 0);
    _n_valid_fpga_samples_in_vis_even = std::vector<int32_t>(_num_freq_per_n2k_frame, 0);
    _n_valid_sample_diff_sq_sum = std::vector<float>(_num_freq_per_n2k_frame, 0);
    _n_rfi_samples_in_vis = std::vector<int32_t>(_num_freq_per_n2k_frame, 0);

    _vis_samples_in_out_frame = 0;
    _accum_fpga_start_tick = 0;

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    in_counts_buf = get_buffer("in_counts_buf");
    in_counts_buf->register_consumer(unique_name);

    in_rfimask_buf = get_buffer("in_rfimask_buf");
    in_rfimask_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Check buffer frame sizes
    size_t in_corr_frame_size = 2 * sizeof(int32_t) * _n2k_correlation_num_products
                                * _num_freq_per_n2k_frame * _n_integrations_per_n2k_frame;
    size_t in_counts_frame_size = sizeof(int32_t) * _n2k_counts_num_products
                                  * _num_freq_per_n2k_frame * _n_integrations_per_n2k_frame;
    size_t in_rfimask_frame_size = _num_freq_per_n2k_frame * _n_fpga_samples_per_n2k_frame / 8;
    size_t out_n2_frame_size = N2FrameView::calculate_frame_size(_num_elements, 0);

    if (in_buf->frame_size != in_corr_frame_size) {
        FATAL_ERROR("N2Accumulate in_buf ({:s}) has frame size {:d}. Expected {:d}.",
                    in_buf->buffer_name, in_buf->frame_size, in_corr_frame_size);
        std::abort();
    }
    if (in_counts_buf->frame_size != in_counts_frame_size) {
        FATAL_ERROR("N2Accumulate in_counts_buf ({:s}) has frame size {:d}. Expected {:d}.",
                    in_counts_buf->buffer_name, in_counts_buf->frame_size, in_counts_frame_size);
        std::abort();
    }
    if (in_rfimask_buf->frame_size != in_rfimask_frame_size) {
        FATAL_ERROR("N2Accumulate in_rfimask_buf ({:s}) has frame size {:d}. Expected {:d}.",
                    in_rfimask_buf->buffer_name, in_rfimask_buf->frame_size, in_rfimask_frame_size);
        std::abort();
    }
    if (out_buf->frame_size != out_n2_frame_size) {
        FATAL_ERROR("N2Accumulate out_buf ({:s}) has frame size {:d}. Expected {:d}.",
                    out_buf->buffer_name, out_buf->frame_size, out_n2_frame_size);
        std::abort();
    }

    // TODO... Should we ensure output buffer has enough frames (>= # frequencies) to take the
    // output without filling completely?
}

void N2Accumulate::main_thread() {

    auto& comp_time_seconds_metric =
        Metrics::instance().add_gauge("kotekan_N2_accum_time", unique_name);
    auto& samples_in_out_frame =
        Metrics::instance().add_gauge("kotekan_samples_in_accumulated_out_frame", unique_name);

    N2::frameID in_frame_id(in_buf);
    N2::frameID in_counts_frame_id(in_counts_buf);
    N2::frameID in_rfimask_frame_id(in_rfimask_buf);
    N2::frameID out_frame_id(out_buf);

    INFO("Accumulating GPU output for {:s}[{:d}] putting result in {:s}[{:d}]", in_buf->buffer_name,
         in_frame_id, out_buf->buffer_name, out_frame_id);

    // Start time of an output frame (initialize to now)
    timespec output_ts;
    timespec_get(&output_ts, TIME_UTC);
    // uint64_t t_output = N2::ts_to_uint64(output_ts);


    uint64_t corr_stride_t = 2 * _n2k_correlation_num_products * _num_freq_per_n2k_frame;
    uint64_t counts_stride_f = _n2k_counts_num_products;
    uint64_t counts_stride_t = _n2k_counts_num_products * _num_freq_per_n2k_frame;

    uint64_t rfi_stride_f = 128; // = rfimask_fast_time_len / bits_per_entry = 1024 / 8;
    uint64_t rfi_stride_thi = rfi_stride_f * _num_freq_per_n2k_frame;


    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        DEBUG("Waiting for new input frame {:s}[{:d}].", in_buf->buffer_name, in_frame_id);
        int32_t* corr = (int32_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (corr == nullptr)
            break;

        // Fetch a new counts frame and get its sequence id
        DEBUG("Waiting for new input counts frame {:s}[{:d}].", in_counts_buf->buffer_name,
              in_counts_frame_id);
        int32_t* counts =
            (int32_t*)in_counts_buf->wait_for_full_frame(unique_name, in_counts_frame_id);
        if (counts == nullptr)
            break;

        // Fetch a new rfimask frame and get its sequence id
        DEBUG("Waiting for new input counts frame {:s}[{:d}].", in_rfimask_buf->buffer_name,
              in_rfimask_frame_id);
        uint8_t* rfimask =
            (uint8_t*)in_rfimask_buf->wait_for_full_frame(unique_name, in_rfimask_frame_id);
        if (rfimask == nullptr)
            break;

        std::shared_ptr<chordMetadata> frame_metadata = get_chord_metadata(in_buf, in_frame_id);
        int64_t in_frame_num = frame_metadata->get_fpga_seq_num() / _n_fpga_samples_per_n2k_frame;

        // Start and end times of this frame
        bool gps_time_enabled = false;
        // Here we'll just use raw nanoseconds
        uint64_t t_frame_s;
        if (gps_time_enabled) {
            t_frame_s = N2::ts_to_uint64(frame_metadata->get_gps_time());
        } else {
            // If GPS time is not set, fall back to system time.
            /*
            timespec ts;
            timeval tv = frame_metadata->get_first_packet_recv_time();
            TIMEVAL_TO_TIMESPEC(&tv, &ts);
            t_frame_s = N2::ts_to_uint64(ts);
            */
            t_frame_s = 0; // TODO: move this logic to telescope
        }
        // uint64_t t_frame_e = t_frame_s + _in_frame_duration_nsec;
        comp_time_seconds_metric.set(t_frame_s / 1e9);

        // Accumulate each visibility sample in the in_frame
        // t_outer
        for (int64_t vis_samp_n = 0; vis_samp_n < _n_integrations_per_n2k_frame; ++vis_samp_n) {

            // Start and end times of the visibility matrix sample
            // uint64_t t_vis_s = t_frame_s + vis_samp_n * _in_frame_vis_duration_nsec;
            // uint64_t t_vis_e = t_vis_s + _in_frame_vis_duration_nsec;

            // "absolute" vis sample number
            int64_t vis_sample_num_abs = in_frame_num * _n_integrations_per_n2k_frame + vis_samp_n;

            DEBUG("Accumulating new visibility sample ({:d} of {:d} in frame).", vis_samp_n,
                  _n_integrations_per_n2k_frame);
            // DEBUG("   Times are [start, end, out, num] = [{:d}, {:d}, {:d}, {:d}]",
            //     t_vis_s, t_vis_e, t_output, vis_sample_num_abs );


            // Finalize accumulation if the visibility elements are past the output time...
            //  end on an odd frame too so we accumulate weights.
            // if (t_vis_s > t_output && vis_sample_num_abs % 2 == 1) {
            if (_vis_samples_in_out_frame >= _num_n2k_samples_to_accumulate) {

                INFO("Finishing N2Accumulate output frame. Accumulated {:d} visibility samples.",
                     _vis_samples_in_out_frame);
                samples_in_out_frame.set(_vis_samples_in_out_frame);
                output_and_reset(in_frame_id, out_frame_id);

                // t_output += 1000000000L; // TODO: Make this a config parameter. Is there a
                // library for LST?
                _vis_samples_in_out_frame = 0;
                _accum_fpga_start_tick = frame_metadata->get_fpga_seq_num()
                                         + vis_samp_n * _n_fpga_samples_per_n2k_correlation;
            }

            uint64_t corr_offset_t = vis_samp_n * corr_stride_t;

            // Double checking the accumulation arrays are the right shape
            assert(_vis.size() == corr_stride_t);
            assert(_vis_even.size() == corr_stride_t);
            assert(_weights.size() == corr_stride_t / 2);

            // The actual accumulation of visibility.
            for (uint64_t d = 0; d < corr_stride_t; ++d) {
                _vis[d] += corr[d + corr_offset_t];
            } // d

            // If we're working on an even sample, store it for differencing
            // with an odd sample. If odd, add to the _weights matrix.
            // Potential optimization: copying vis_even is only really
            // necessary if we've started accumulating a new frame
            if (vis_sample_num_abs % 2 == 0) {
                std::copy(corr + corr_offset_t, corr + corr_offset_t + corr_stride_t,
                          _vis_even.begin());
            } else {
                for (uint64_t d = 0;
                     d < (uint64_t)(_n2k_correlation_num_products * _num_freq_per_n2k_frame); ++d) {
                    float dr = corr[corr_offset_t + 2 * d + 0] - _vis_even[2 * d + 0];
                    float di = corr[corr_offset_t + 2 * d + 1] - _vis_even[2 * d + 1];
                    _weights[d] += dr * dr + di * di;
                } // d
            } // if even/odd

            // Track (frequency-dependent) lost samples
            for (int64_t f = 0; f < _num_freq_per_n2k_frame; ++f) {

                // Assume the packet loss is scalar, read the counts for the frame
                // from element 0.
                assert(_packet_loss_is_scalar);
                int64_t count_idx = f * counts_stride_f + vis_samp_n * counts_stride_t;

                int32_t valid_fpga_samples = counts[count_idx];

                _n_valid_fpga_samples_in_vis[f] += valid_fpga_samples;

                // Track the lost samples needed for the weights too.
                if (vis_sample_num_abs % 2 == 0) {
                    _n_valid_fpga_samples_in_vis_even[f] = valid_fpga_samples;
                } else {
                    float samples_diff = valid_fpga_samples - _n_valid_fpga_samples_in_vis_even[f];
                    _n_valid_sample_diff_sq_sum[f] += samples_diff * samples_diff;
                } // if even/odd
            } // f

            // Sum the RFI mask
            //
            // Each RFI mask frame has structure:
            //
            //      rfimask[Thi/1024, F, Tlo]
            //
            // In particular:
            //
            //      int1 rfimask[n_fpga_samples_per_n2k_frame / 1024, _num_freq_per_n2k_frame, 1024]
            //      int8 rfimask[n_fpga_samples_per_n2k_frame / 1024, _num_freq_per_n2k_frame, 128]
            //
            // Raw time sample index at start of vis sample (n2k integration):
            //
            //      t = vis_samp_n * n_fpga_samples_per_n2k_correlation
            //
            // For an int8 rfimask, for a raw index t:
            //
            //      thi = t / 1024
            //      tlo = (t % 1024) / 8
            //      tbit = t % 8
            //
            //      t = tbit + 8*tlo + 1024*thi
            //
            // Furthermore, the RFI mask is only computed every rfi_downsampling_factor
            // samples, so t can be incremented by rfi_downsampling_factor.
            // rfi_downsampling factor itself must be divisible by 32, so we know
            // tbit will always be 0.
            for (int64_t f = 0; f < _num_freq_per_n2k_frame; ++f) {
                for (int64_t t = vis_samp_n * _n_fpga_samples_per_n2k_correlation;
                     t < (vis_samp_n + 1) * _n_fpga_samples_per_n2k_correlation;
                     t += _rfi_downsampling_factor) {

                    // Casting to a uint64_t here is a micro-optimization,
                    // the assembly is slighty simpler if the compliler knows
                    // the numerator is non-negative.
                    int64_t thi = ((uint64_t)t) / 1024;
                    int64_t tlo = (((uint64_t)t) % 1024) / 8;

                    int64_t idx = thi * rfi_stride_thi + f * rfi_stride_f + tlo;

                    _n_rfi_samples_in_vis[f] +=
                        (1 - (rfimask[idx] & 0x1)) * _rfi_downsampling_factor;
                }

            } // f

            _vis_samples_in_out_frame++;

        } // t (vis samples in frame)

        // Advance to the next frame
        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        in_counts_buf->mark_frame_empty(unique_name, in_counts_frame_id++);
        in_rfimask_buf->mark_frame_empty(unique_name, in_rfimask_frame_id++);
    }
}

bool N2Accumulate::output_and_reset(N2::frameID& in_frame_id, N2::frameID& out_frame_id) {
    // Different frame for each frequency
    // But, same metadata
    std::shared_ptr<chordMetadata> chord_frame_metadata = get_chord_metadata(in_buf, in_frame_id);

    // strides into the N2K shaped accumulation array
    int64_t stride_ilo = _n2k_correlation_blocksize;
    int64_t stride_block = stride_ilo * _n2k_correlation_blocksize;
    int64_t stride_f = stride_block * _n2k_correlation_num_blocks;

    // Loop over frequency
    for (int64_t f = 0; f < _num_freq_per_n2k_frame; ++f) {

        if (out_buf->wait_for_empty_frame(unique_name, out_frame_id) == nullptr) {
            return false;
        }

        DEBUG("Allocating metadata.");
        out_buf->allocate_new_metadata_object(out_frame_id);
        std::shared_ptr<N2Metadata> meta = get_N2_metadata(out_buf, out_frame_id);

        meta->fpga_start_tick = _accum_fpga_start_tick;
        meta->frame_length_fpga_ticks =
            _vis_samples_in_out_frame * _n_fpga_samples_per_n2k_correlation;
        meta->num_elements = _num_elements;
        meta->num_prod = _N2_num_products;
        meta->num_ev = 0;
        meta->nfreq = _num_freq_per_n2k_frame;
        meta->freq_id = chord_frame_metadata->get_coarse_freq()[f];
        meta->n_valid_fpga_ticks = _n_valid_fpga_samples_in_vis[f];

        meta->frame_start_time_ns = _tel.to_time_ns(meta->fpga_start_tick);
        meta->freq_Hz = _tel.to_freq(meta->freq_id);
        meta->eop = _tel.get_EOP_at_time(
            _tel.to_time(meta->fpga_start_tick + meta->frame_length_fpga_ticks / 2));
        meta->n_rfi_fpga_ticks = _n_rfi_samples_in_vis[f];

        DEBUG("Creating N2FrameView for freq f[{:d}] = {:d}", f,
              chord_frame_metadata->get_coarse_freq()[f]);
        N2FrameView out_vis(out_buf, out_frame_id);

        // Sample numbers for normalizing weights
        DEBUG("Computing normalization.");
        int64_t ns = _n_valid_fpga_samples_in_vis[f]; // ns = "number of samples"
        float ins = (ns != 0) ? (1.0f / ((float)ns)) : 0.0f;

        // Copy data into buffer.
        // This requires changing from the GPU's blocked format to the triangular format N2Buffer
        // expects.

        // iterate over the N2K format (blocked lower triangular)
        int64_t block_idx = 0;
        for (int64_t ihi = 0; ihi < _n2k_correlation_lin_blocks; ihi++) {
            // Lower triangular blocks
            for (int64_t jhi = 0; jhi <= ihi; jhi++) {
                for (int64_t ilo = 0; ilo < _n2k_correlation_blocksize; ilo++) {
                    for (int64_t jlo = 0; jlo < _n2k_correlation_blocksize; jlo++) {
                        // 2D indices into the N2K matrix.
                        int64_t i = ilo + _n2k_correlation_blocksize * ihi;
                        int64_t j = jlo + _n2k_correlation_blocksize * jhi;

                        // Only proceed if we're in the *true* lower-triangular section of the
                        // matrix
                        if (j > i)
                            continue;

                        // index into the intermediate N2K-shaped array
                        int64_t idx =
                            jlo + ilo * stride_ilo + block_idx * stride_block + f * stride_f;

                        // Get the index into the n2 view.  N2 is an *upper* triangular unblocked
                        // form, so we use the global matrix indices to the lower-triangular N2K
                        // form, and compute the triangular index with their transpose.
                        //
                        //  N2K:                    N2:
                        //       j                        j
                        //       0  1  2  3               0  1  2  3
                        //      -----------              -----------
                        // i  0| 0                  i  0| 0  1  2  3
                        //    1| 1  2                  1|    4  5  6
                        //    2| 3  4  5               2|       7  8
                        //    3| 6  7  8  9            3|          9
                        //
                        // vis_N2(i, j) = vis_n2k(j, i)*

                        int64_t n2_idx = N2::cmap(j, i, _num_elements);

                        // Populate the visibility matrix, remember the upper-tri element
                        // is the conjugate of the lower-tri element.
                        N2::cfloat v{(float)_vis[2 * idx], (float)_vis[2 * idx + 1]};
                        out_vis.vis[n2_idx] = ins * std::conj(v);

                        if (ns > 0) {
                            _weights[idx] -=
                                std::norm(v) * _n_valid_sample_diff_sq_sum[f] * ins * ins;
                        }

                        out_vis.weight[n2_idx] = (ns > 0) ? ns * (ns / _weights[idx]) : 0;
                    } // jlo
                } // ilo

                block_idx++;
            } // jhi
        } // ihi

        out_vis.erms = -1;
        std::fill(out_vis.flags.begin(), out_vis.flags.end(), 0);
        std::fill(out_vis.gain.begin(), out_vis.gain.end(), N2::cfloat{-1.0f, 0.0f});

        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }

    DEBUG("Wrapping up accumulation buffer output copy.");

    std::fill(_vis.begin(), _vis.end(), 0);
    std::fill(_weights.begin(), _weights.end(), 0.0f);
    std::fill(_n_valid_fpga_samples_in_vis.begin(), _n_valid_fpga_samples_in_vis.end(), 0);
    std::fill(_n_valid_sample_diff_sq_sum.begin(), _n_valid_sample_diff_sq_sum.end(), 0);
    std::fill(_n_rfi_samples_in_vis.begin(), _n_rfi_samples_in_vis.end(), 0);

    return true;
}
