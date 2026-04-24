#include "N2Accumulate.hpp"

#include "Config.hpp"            // for Config
#include "N2FrameView.hpp"       // for N2FrameView
#include "N2Metadata.hpp"        // for N2Metadata, get_N2_metadata
#include "N2Util.hpp"            // for frameID, modulo, cfloat, cmap
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, get_chord_metadata
#include "jsonMetadata.hpp"      // for MAX_NUM_RFI_THRESHOLDS
#include "kotekanLogging.hpp"    // for FATAL_ERROR, DEBUG, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "timeUtil.hpp"          // for EOP

#include "fmt.hpp"      // for compile_string_to_view
#include "gsl-lite.hpp" // for span

#include <algorithm>  // for fill, copy
#include <assert.h>   // for assert
#include <complex>    // for conj, norm, operator*, complex
#include <cstdlib>    // for abort
#include <functional> // for bind, function, placeholders
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <omp.h>
#include <sched.h>
#include <time.h> // for size_t, timespec
#include <vector> // for vector


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(N2Accumulate);

enum class Mode {
    START,
    WAITING_FOR_ALIGNMENT,
    READY,
    ACCUMULATING,
};


N2Accumulate::N2Accumulate(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2Accumulate::main_thread, this)),
    _num_freq_per_n2k_frame(config.get<int64_t>(unique_name, "num_freq_per_n2k_frame")),
    _bin_in_ERA(config.get_default<bool>(unique_name, "bin_in_ERA", false)),
    _num_n2k_samples_to_accumulate(
        config.get_default<int64_t>(unique_name, "num_n2k_samples_to_accumulate", 0)),
    _num_bins_per_rotation(config.get_default<uint32_t>(unique_name, "num_bins_per_rotation", 0)),
    _packet_loss_is_scalar(config.get<bool>(unique_name, "packet_loss_is_scalar")),
    _n_fpga_samples_per_n2k_frame(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _n_fpga_samples_per_n2k_correlation(config.get<int64_t>(unique_name, "sub_integration_ntime")),
    _num_elements(config.get<int64_t>(unique_name, "num_elements")),
    _num_workers(config.get_default<int>(unique_name, "num_workers", 1)),
    _output_batch_size(config.get_default<int>(unique_name, "output_batch_size", 1)),
    _do_fringestop(config.get_default<bool>(unique_name, "do_fringestop", false)),
    _variance_mode(config.get<N2VarianceMode>(unique_name, "variance_mode")),
    _debug_accum_mode(config.get_default<int>(unique_name, "debug_accum_mode", 2)),
    _tel(Telescope::instance()),
    skipped_frame_counter(Metrics::instance().add_counter(
        "kotekan_N2accumulate_skipped_frame_total", unique_name, {"freq_id", "reason"})) {

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    INFO("Variance mode: {}", _variance_mode);

    // Ensure outgoing buffer is of type N2
    if (out_buf->buffer_type != "N2")
        FATAL_ERROR("N2Accumulate out_buf ({:s}) is not of type N2.", out_buf->buffer_name);

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    in_counts_buf = get_buffer("in_counts_buf");
    in_counts_buf->register_consumer(unique_name);

    in_rficounts_buf = get_buffer("in_rficounts_buf");
    in_rficounts_buf->register_consumer(unique_name);

    in_rfiframemask_buf = get_buffer("in_rfiframemask_buf");
    in_rfiframemask_buf->register_consumer(unique_name);

    // Sanity checks on initialization
    {
        // number of frequencies in incoming frames from n2k
        if (_num_freq_per_n2k_frame <= 0)
            FATAL_ERROR("num_freq_per_n2k_frame is not positive: {:d}", _num_freq_per_n2k_frame);

        // accumulation setup
        if (!_bin_in_ERA
            && (_num_n2k_samples_to_accumulate <= 0 || _num_n2k_samples_to_accumulate % 2 != 0))
            FATAL_ERROR("N2Accumulate configured to use fixed-sample accumulation "
                        "(bin_in_ERA is false) with non-positive or odd (should be "
                        "positive and even!) num_n2k_samples_to_accumulate: {:d}",
                        _num_n2k_samples_to_accumulate);

        if (!_packet_loss_is_scalar)
            FATAL_ERROR(
                "N2Accumulate configured to use packet loss matrix, which is not implemented.");

        // sampling information
        if (!(_n_fpga_samples_per_n2k_frame > 0))
            FATAL_ERROR("samples_per_data_set is not positve: {:d}", _n_fpga_samples_per_n2k_frame);
        if (!(_n_fpga_samples_per_n2k_correlation > 0))
            FATAL_ERROR("sub_integration_ntime is not positve: {:d}",
                        _n_fpga_samples_per_n2k_correlation);
        if (!(_n_fpga_samples_per_n2k_frame % _n_fpga_samples_per_n2k_correlation == 0))
            FATAL_ERROR(
                "samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d})",
                _n_fpga_samples_per_n2k_frame, _n_fpga_samples_per_n2k_correlation);

        // Number of elements (polarization x dish) in the array.
        if (_num_elements <= 0)
            FATAL_ERROR("num_elements is not positive: {:d}", _num_elements);

        // Check num_elements is consistent with the count & correlation blocksizes.
        if (!(_num_elements % _n2k_correlation_blocksize == 0))
            FATAL_ERROR(
                "num_elements ({:d}) is not a multiple of the correlation block size ({:d})",
                _num_elements, _n2k_correlation_blocksize);

        if (!(_num_elements % (8 * _n2k_counts_blocksize) == 0))
            FATAL_ERROR("num_elements ({:d}) / 8 is not a multiple of the counts block size ({:d})",
                        _num_elements, _n2k_counts_blocksize);

        // We output frames in batches. Make sure a whole batch will fit in the output buffer.
        if (_output_batch_size > out_buf->num_frames) {
            FATAL_ERROR("output_batch_size ({:d}) is greater than out_buf's num_frames (aka "
                        "buffer_depth) ({:d})",
                        _output_batch_size, out_buf->num_frames);
        }
    }

    // Compute derived quantities
    {
        // number of "integrations" (coarse time samples) per n2k (gpu) frame
        _n_integrations_per_n2k_frame =
            _n_fpga_samples_per_n2k_frame / _n_fpga_samples_per_n2k_correlation;
        assert(_n_fpga_samples_per_n2k_frame % _n_fpga_samples_per_n2k_correlation == 0
               && "_n_fpga_samples_per_n2k_frame must be a multiple of "
                  "_n_fpga_samples_per_n2k_correlation");

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
    }

    // Initialize these here using the computed _n2k_correlation_num_products *
    // _num_freq_per_n2k_frame (accumulate the full, blocked matrix x frequencies from the GPU)
    _vis = std::vector<int32_t>(2 * _num_freq_per_n2k_frame * _n2k_correlation_num_products,
                                0); // vis with complex as 2 ints
    _weights = std::vector<float>(_num_freq_per_n2k_frame * _n2k_correlation_num_products,
                                  0.0f); // real-valued weights

    // number of fpga samples, per frequency, in frame
    _n_valid_fpga_samples_in_vis = std::vector<int32_t>(_num_freq_per_n2k_frame, 0);
    _n_valid_sample_diff_sq_sum = std::vector<float>(_num_freq_per_n2k_frame, 0);
    _n_rfi_samples_in_vis = std::vector<int32_t>(_num_freq_per_n2k_frame, 0);

    _vis_samples_in_out_frame = 0;
    _accum_fpga_start_tick = -1;
    _accum_bin_idx = -1;

    // Ensure incoming buffer shapes and type are correct
    in_buf->allocate_ndarray_frame_desc(kotekan::int32, "n2k_correlation",
                                        {_n_integrations_per_n2k_frame, _num_freq_per_n2k_frame,
                                         _n2k_correlation_num_blocks, _n2k_correlation_blocksize,
                                         _n2k_correlation_blocksize, 2},
                                        {"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"});

    in_counts_buf->allocate_ndarray_frame_desc(kotekan::int32, "n2k_counts",
                                               {_n_integrations_per_n2k_frame,
                                                _num_freq_per_n2k_frame, _n2k_counts_num_blocks,
                                                _n2k_counts_blocksize, _n2k_counts_blocksize},
                                               {"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"});

    in_rficounts_buf->allocate_ndarray_frame_desc(
        kotekan::int32, "RFImask_count", {_n_integrations_per_n2k_frame, _num_freq_per_n2k_frame},
        {"Tc", "F"});

    in_rfiframemask_buf->allocate_ndarray_frame_desc(
        kotekan::uint8, "RFIFrameMask", {_n_integrations_per_n2k_frame, _num_freq_per_n2k_frame},
        {"Tc", "F"});


    // Validate that the output buffer's frame descriptor (set by bufferFactory) matches
    // what this stage will produce
    {
        auto out_frame_desc = out_buf->get_frame_description();
        if (!out_frame_desc) {
            FATAL_ERROR("N2Accumulate: Output buffer {:s} does not have a frame descriptor set",
                        out_buf->buffer_name);
        }
        auto n2_desc = std::dynamic_pointer_cast<const kotekan::N2FrameDesc>(out_frame_desc);
        if (!n2_desc) {
            FATAL_ERROR("N2Accumulate: Output buffer {:s} does not have an N2FrameDesc",
                        out_buf->buffer_name);
        }
        // Validate the descriptor matches what we expect to produce
        if (n2_desc->get_num_elements() != (uint32_t)_num_elements) {
            FATAL_ERROR(
                "N2Accumulate: Output buffer num_elements ({:d}) does not match expected ({:d})",
                n2_desc->get_num_elements(), _num_elements);
        }
        if (n2_desc->get_num_ev() != 0) {
            FATAL_ERROR("N2Accumulate: Output buffer num_ev ({:d}) must be 0",
                        n2_desc->get_num_ev());
        }
        if (n2_desc->get_num_products() != (uint32_t)_N2_num_products) {
            FATAL_ERROR(
                "N2Accumulate: Output buffer num_products ({:d}) does not match expected ({:d})",
                n2_desc->get_num_products(), _N2_num_products);
        }
        if (n2_desc->get_n2_layout() != N2Layout::FullUpperTri) {
            FATAL_ERROR("N2Accumulate: Output buffer n2_layout must be FullUpperTri");
        }
    }

    // TODO... Should we ensure output buffer has enough frames (>= # frequencies) to take the
    // output without filling completely?
}

void N2Accumulate::main_thread() {

    auto& comp_time_seconds_metric =
        Metrics::instance().add_gauge("kotekan_N2_accum_time", unique_name);
    auto& samples_in_out_frame =
        Metrics::instance().add_gauge("kotekan_samples_in_accumulated_out_frame", unique_name);

    frameID in_frame_id(in_buf);
    frameID in_counts_frame_id(in_counts_buf);
    frameID in_rficounts_frame_id(in_rficounts_buf);
    frameID in_rfiframemask_frame_id(in_rfiframemask_buf);
    frameID out_frame_id(out_buf);

    int previous_in_frame_id = -1;
    int previous_in_counts_frame_id = -1;
    int previous_in_rficounts_frame_id = -1;
    int previous_in_rfiframemask_frame_id = -1;

    INFO("Accumulating GPU output for {:s}[{:d}] putting result in {:s}[{:d}]", in_buf->buffer_name,
         in_frame_id, out_buf->buffer_name, out_frame_id);

    uint64_t corr_stride_t = 2 * _n2k_correlation_num_products * _num_freq_per_n2k_frame;
    uint64_t corr_stride_f = 2 * _n2k_correlation_num_products;
    int64_t counts_stride_t = _n2k_counts_num_products * _num_freq_per_n2k_frame;
    int64_t counts_stride_f = _n2k_counts_num_products;

    // A buffer to store the (possibly fringestopped) correlations of a single time and all
    // frequencies
    std::vector<int32_t> vis_even_vec(corr_stride_t, 0);
    const int32_t* corr_t0 = nullptr;
    const int32_t* corr_t1 = nullptr;
    const int32_t* counts_mat_t0 = nullptr;
    const int32_t* counts_mat_t1 = nullptr;
    const int32_t* rficounts_t0 = nullptr;
    const int32_t* rficounts_t1 = nullptr;
    const uint8_t* rfiframemask_t0 = nullptr;
    const uint8_t* rfiframemask_t1 = nullptr;

    std::vector<int32_t> n_valid_fpga_samples_t0(_num_freq_per_n2k_frame, 0);
    std::vector<int32_t> n_valid_fpga_samples_t1(_num_freq_per_n2k_frame, 0);

    // vis_even may be an odd frame if the first frame eencountered is odd,
    // since fpga_seq_num does not have to start from 0, in which case we skip
    // it.
    // int32_t const* vis_even = nullptr;

    // EOP at target fringestop time.
    EOP target_eop = eop_null;

    int num_dishes = _tel.cast<CHORDTelescope>().get_num_dishes();
    // storage for a single frequency's fringe phases, declared here so it is only
    // allocated once.
    std::vector<std::complex<float>> fringe_phase_t0(num_dishes, 1.0f);
    std::vector<std::complex<float>> fringe_phase_t1(num_dishes, 1.0f);

    if (_num_elements % num_dishes != 0)
        FATAL_ERROR("num_dishes {:d} (from telescope) is not a multiple of num_elements {:d}",
                    num_dishes, _num_elements);
    assert(_num_elements % num_dishes == 0);

    // We start with START.
    Mode mode = Mode::START;

    [[maybe_unused]] double prof_last_time = omp_get_wtime();

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        DEBUG("Waiting for new correlation frame {:s}[{:d}].", in_buf->buffer_name, in_frame_id);
        const int32_t* corr = (int32_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (corr == nullptr)
            break;

        // Fetch a new counts frame and get its sequence id
        DEBUG("Waiting for new input counts frame {:s}[{:d}].", in_counts_buf->buffer_name,
              in_counts_frame_id);
        const int32_t* counts_mat =
            (int32_t*)in_counts_buf->wait_for_full_frame(unique_name, in_counts_frame_id);
        if (counts_mat == nullptr)
            break;

        // Fetch a new rficounts frame and get its sequence id
        DEBUG("Waiting for new input RFIcounts frame {:s}[{:d}].", in_rficounts_buf->buffer_name,
              in_rficounts_frame_id);
        const int32_t* rficounts =
            (int32_t*)in_rficounts_buf->wait_for_full_frame(unique_name, in_rficounts_frame_id);
        if (rficounts == nullptr)
            break;

        // Fetch a new rfiframemask frame and get its sequence id
        DEBUG("Waiting for new input RFIFrameMask frame {:s}[{:d}].",
              in_rfiframemask_buf->buffer_name, in_rfiframemask_frame_id);
        const uint8_t* rfiframemask = (uint8_t*)in_rfiframemask_buf->wait_for_full_frame(
            unique_name, in_rfiframemask_frame_id);
        if (rfiframemask == nullptr)
            break;

        // Get metadata for all incoming frames.
        std::shared_ptr<chordMetadata> frame_metadata = get_chord_metadata(in_buf, in_frame_id);

        std::shared_ptr<chordMetadata> counts_metadata =
            get_chord_metadata(in_counts_buf, in_counts_frame_id);
        std::shared_ptr<chordMetadata> rficounts_metadata =
            get_chord_metadata(in_rficounts_buf, in_rficounts_frame_id);
        std::shared_ptr<chordMetadata> rfiframemask_metadata =
            get_chord_metadata(in_rfiframemask_buf, in_rfiframemask_frame_id);

        // Check synchronization
        if (frame_metadata->get_fpga_seq_num() != counts_metadata->get_fpga_seq_num()) {
            FATAL_ERROR("Correlation buffer {:s}[{:d}] seq={:d} has lost synchronization with "
                        "Counts buffer {:s}[{:d}] seq={:d}",
                        in_buf->buffer_name, in_frame_id, frame_metadata->get_fpga_seq_num(),
                        in_counts_buf->buffer_name, in_counts_frame_id,
                        counts_metadata->get_fpga_seq_num());
        }
        if (frame_metadata->get_fpga_seq_num() != rficounts_metadata->get_fpga_seq_num()) {
            FATAL_ERROR("Correlation buffer {:s}[{:d}] seq={:d} has lost synchronization with "
                        "RFICounts buffer {:s}[{:d}] seq={:d}",
                        in_buf->buffer_name, in_frame_id, frame_metadata->get_fpga_seq_num(),
                        in_rficounts_buf->buffer_name, in_rficounts_frame_id,
                        rficounts_metadata->get_fpga_seq_num());
        }
        if (frame_metadata->get_fpga_seq_num() != rfiframemask_metadata->get_fpga_seq_num()) {
            FATAL_ERROR("Correlation buffer {:s}[{:d}] seq={:d} has lost synchronization with "
                        "RFIFrameMask buffer {:s}[{:d}] seq={:d}",
                        in_buf->buffer_name, in_frame_id, frame_metadata->get_fpga_seq_num(),
                        in_rfiframemask_buf->buffer_name, in_rfiframemask_frame_id,
                        rfiframemask_metadata->get_fpga_seq_num());
        }

        // Record the current frame time being processed.
        comp_time_seconds_metric.set(_tel.to_time_ns(frame_metadata->get_fpga_seq_num()) / 1e9);


        // Sequence number for the start of this frame.
        int64_t seq0 = frame_metadata->get_fpga_seq_num();

        // Absolute frame number for this frame (since frame 0)
        int64_t in_frame_num = frame_metadata->get_fpga_seq_num() / _n_fpga_samples_per_n2k_frame;


        // Do some first-time initialization
        if (mode == Mode::START) {

            // During startup, _accum_bin_idx stores the bin index of the last
            // sample (e.g. the one *before* the stage started). This is usually
            // also the bin index of the current (first) frame.
            _accum_bin_idx = get_accum_abs_bin_idx(seq0);

            // Unless the first frame happens to start a new bin! Then
            // _accum_bin_idx should refer to the previous bin.
            if (is_seq_start_of_bin(seq0, _accum_bin_idx))
                _accum_bin_idx--;

            // Now the WAITING_FOR_ALIGNMENT check later will correctly detect
            // if the current frame starts a new bin.

            mode = Mode::WAITING_FOR_ALIGNMENT;
        }

        // Accumulate each visibility sample in the in_frame
        // t_outer
        for (int64_t t = 0; t < _n_integrations_per_n2k_frame; ++t) {

            [[maybe_unused]] double prof_start_time = omp_get_wtime();


            // "absolute" vis sample number
            int64_t t_abs = t + in_frame_num * _n_integrations_per_n2k_frame;

            // sequence number of this sample in the frame.
            int64_t seq = seq0 + t * _n_fpga_samples_per_n2k_correlation;

            DEBUG("Frame: {0:d}  Sample {1:d}: {2:d} seq: {3:d}", in_frame_num, t, t_abs, seq);

            int64_t bin_idx = get_accum_abs_bin_idx(seq);

            // Startup - wait to be at the beginning of an accumulation bin.
            if (mode == Mode::WAITING_FOR_ALIGNMENT) {


                if (bin_idx != _accum_bin_idx) {
                    // Away we go!
                    assert(t_abs % 2 == 0);
                    _vis_samples_in_out_frame = 0;
                    _accum_fpga_start_tick = seq;
                    _accum_bin_idx = bin_idx;
                    target_eop = get_accum_bin_EOP(bin_idx);

                    mode = Mode::ACCUMULATING;
                    DEBUG("MODE: Setting accum_fpga_start_tick: {0:d}", _accum_fpga_start_tick);
                }
            }

            // If we're not ready to accumulate, keep on spinning until we reach the correct frame.
            if (mode != Mode::ACCUMULATING) {
                DEBUG("Waiting for accumulation bin {:d} to start, currently at {:d}. Skipping"
                      " visibility sample {:d} of {:d} in frame with seq {:d}.",
                      _accum_bin_idx + 1, bin_idx, t, _n_integrations_per_n2k_frame, seq);
                continue;
            }

            DEBUG("Accumulating new visibility sample ({:d} of {:d} in frame).", t,
                  _n_integrations_per_n2k_frame);
            // DEBUG("   Times are [start, end, out, num] = [{:d}, {:d}, {:d}, {:d}]",
            //     t_vis_s, t_vis_e, t_output, t_abs );


            // Finalize accumulation if the visibility elements are past the output time...
            //  end on an odd frame too so we accumulate weights.
            // if (t_vis_s > t_output && t_abs % 2 == 1) { }
            if (_vis_samples_in_out_frame == _num_n2k_samples_to_accumulate) {

                INFO("Finishing N2Accumulate output frame. Accumulated {:d} visibility samples.",
                     _vis_samples_in_out_frame);
                samples_in_out_frame.set(_vis_samples_in_out_frame);
                output_and_reset(in_frame_id, in_rfiframemask_frame_id, out_frame_id);

                _vis_samples_in_out_frame = 0;
                _accum_fpga_start_tick =
                    frame_metadata->get_fpga_seq_num() + t * _n_fpga_samples_per_n2k_correlation;
            }

            uint64_t corr_offset_t = t * corr_stride_t;
            uint64_t counts_offset_t = t * counts_stride_t;
            uint64_t rfi_offset_t = t * _num_freq_per_n2k_frame;

            // Double checking the accumulation arrays are the right shape
            assert(_vis.size() == corr_stride_t);
            assert(_weights.size() == corr_stride_t / 2);


            // First: If we're an even frame, save pointers to the correlation and counts matrices
            // and get outta here.
            if (t_abs % 2 == 0) {
                corr_t0 = corr + corr_offset_t;
                counts_mat_t0 = counts_mat + counts_offset_t;
                rficounts_t0 = rficounts + rfi_offset_t;
                rfiframemask_t0 = rfiframemask + rfi_offset_t;
                continue;
            }

            corr_t1 = corr + corr_offset_t;
            counts_mat_t1 = counts_mat + counts_offset_t;
            rficounts_t1 = rficounts + rfi_offset_t;
            rfiframemask_t1 = rfiframemask + rfi_offset_t;

            EOP eop_t0 =
                _tel.get_EOP_at_time(_tel.to_time(seq - _n_fpga_samples_per_n2k_correlation / 2));
            EOP eop_t1 =
                _tel.get_EOP_at_time(_tel.to_time(seq + _n_fpga_samples_per_n2k_correlation / 2));

#pragma omp parallel for num_threads(_num_workers)
            for (int64_t f = 0; f < _num_freq_per_n2k_frame; ++f) {
                if (rfiframemask_t0[f] == 0 || rfiframemask_t1[f] == 0) {
                    _n_rfi_samples_in_vis[f] += 2 * _n_fpga_samples_per_n2k_correlation;
                    continue;
                }

                // Second: accum RFI.
                _n_rfi_samples_in_vis[f] += rficounts_t0[f] + rficounts_t1[f];

                // Third: normalization.  Accumulate the number of valid ticks in the last two
                // samples
                int64_t count_idx = f * counts_stride_f;

                int32_t count_t0 = counts_mat_t0[count_idx];
                int32_t count_t1 = counts_mat_t1[count_idx];

                _n_valid_fpga_samples_in_vis[f] += count_t0 + count_t1;

                float samples_diff = count_t1 - count_t0;
                _n_valid_sample_diff_sq_sum[f] += samples_diff * samples_diff;

                double freq_MHz =
                    _tel.to_freq_MHz(static_cast<freq_id_t>(frame_metadata->get_coarse_freq()[f]));
                // Fourth: Now for the hard part, visibilities and weights
                accum_corr_and_weight(
                    _vis.data() + f * corr_stride_f, _weights.data() + f * corr_stride_f / 2,
                    corr_t0 + f * corr_stride_f, corr_t1 + f * corr_stride_f, freq_MHz, target_eop,
                    eop_t0, eop_t1, count_t0, count_t1, fringe_phase_t0, fringe_phase_t1);
            }

            // We're adding frames in pairs, increment frame count by 2
            _vis_samples_in_out_frame += 2;

            // Finalize accumulation if the next sample is in a new bin.
            int64_t next_bin_idx = get_accum_abs_bin_idx(seq + _n_fpga_samples_per_n2k_correlation);
            if (next_bin_idx != _accum_bin_idx) {

                INFO("Finishing N2Accumulate output frame. Accumulated {:d} visibility samples.",
                     _vis_samples_in_out_frame);
                samples_in_out_frame.set(_vis_samples_in_out_frame);
                output_and_reset(in_frame_id, in_rfiframemask_frame_id, out_frame_id);

                _vis_samples_in_out_frame = 0;
                _accum_fpga_start_tick = seq + _n_fpga_samples_per_n2k_correlation;
                _accum_bin_idx = next_bin_idx;
                target_eop = get_accum_bin_EOP(next_bin_idx);
            }

            [[maybe_unused]] double prof_curr_time = omp_get_wtime();
            INFO("Adding input frame pair took {:f} ms + {:f} ms idle",
                 (prof_curr_time - prof_start_time) * 1000,
                 (prof_start_time - prof_last_time) * 1000);
            prof_last_time = prof_curr_time;

        } // t (vis samples in frame)

        // Advance to the next frame
        if (previous_in_frame_id != -1)
            in_buf->mark_frame_empty(unique_name, previous_in_frame_id);
        previous_in_frame_id = in_frame_id++;
        if (previous_in_counts_frame_id != -1)
            in_counts_buf->mark_frame_empty(unique_name, previous_in_counts_frame_id);
        previous_in_counts_frame_id = in_counts_frame_id++;
        if (previous_in_rficounts_frame_id != -1)
            in_rficounts_buf->mark_frame_empty(unique_name, previous_in_rficounts_frame_id);
        previous_in_rficounts_frame_id = in_rficounts_frame_id++;
        if (previous_in_rfiframemask_frame_id != -1)
            in_rfiframemask_buf->mark_frame_empty(unique_name, previous_in_rfiframemask_frame_id);
        previous_in_rfiframemask_frame_id = in_rfiframemask_frame_id++;
    }
}

int64_t N2Accumulate::calculate_ERA_bin_idx_from_time(const timespec& t_inst) {

    EOP eop = _tel.get_EOP_at_time(t_inst);
    int64_t t_ut1 = get_UT1_from_time(t_inst, eop.delta_UT1_inst);
    int64_t nrot;
    double ERA_deg = get_ERA_from_UT1(t_ut1, &nrot); // ERA is always in [0.0, 360.0)

    // Calculate which bin this index is in
    int64_t ERA_idx = static_cast<int64_t>(floor((ERA_deg / 360.0) * _num_bins_per_rotation));

    return _num_bins_per_rotation * nrot + ERA_idx;
}

int64_t N2Accumulate::get_accum_abs_bin_idx(uint64_t seq) {

    if (_bin_in_ERA) {

        // number of ticks in an even/odd pair of samples
        uint64_t ticks_per_sample_pair = 2 * _n_fpga_samples_per_n2k_correlation;

        // the tick number for the start of the current even/odd pair
        uint64_t seq_start = (seq / ticks_per_sample_pair) * ticks_per_sample_pair;

        // sequence number for the center of the pair (the beginning of the odd sample)
        uint64_t seq_cen = seq_start + _n_fpga_samples_per_n2k_correlation;

        // Get the instrument time at the center of the pair
        timespec t_inst = _tel.to_time(seq_cen);

        // Return the bin idx;
        return calculate_ERA_bin_idx_from_time(t_inst);

    } else {
        int64_t fpga_ticks_per_accum =
            _num_n2k_samples_to_accumulate * _n_fpga_samples_per_n2k_correlation;
        int64_t idx = seq / fpga_ticks_per_accum;

        return idx;
    }
}

EOP N2Accumulate::get_accum_bin_EOP(int64_t accum_bin_idx) {
    if (_bin_in_ERA) {
        // extract the rotation number and ERA bin from the index
        int64_t nrot = accum_bin_idx / _num_bins_per_rotation;
        int64_t ERA_idx = accum_bin_idx % _num_bins_per_rotation;

        // ERA of bin center
        double ERA_cen = (360.0 * ERA_idx) / _num_bins_per_rotation;

        // UT1 time at bin center
        int64_t t_ut1 = get_UT1_from_ERA(nrot, ERA_cen);

        // return the EOP
        return _tel.get_EOP_at_UT1(t_ut1);
    } else {

        // extract the sequence number of the start of the bin
        int64_t fpga_ticks_per_accum =
            _num_n2k_samples_to_accumulate * _n_fpga_samples_per_n2k_correlation;
        uint64_t seq_start = static_cast<uint64_t>(accum_bin_idx) * fpga_ticks_per_accum;

        // sequence number at center of bin, we know fpga_ticks_per_accum is even
        uint64_t seq_cen = seq_start + fpga_ticks_per_accum / 2;

        // instrument time at bin center
        timespec ts_cen = _tel.to_time(seq_cen);

        // Return the EOP
        return _tel.get_EOP_at_time(ts_cen);
    }
}

bool N2Accumulate::is_seq_start_of_bin(uint64_t seq, int64_t bin_idx) {
    if (_bin_in_ERA) {

        // Only way to check this is to check the bin idx of the last frame.
        // Since seq >= 0, we have to be a little tricky since we might be looking
        // before frame 0.

        // number of ticks in an even/odd pair of samples
        uint64_t ticks_per_sample_pair = 2 * _n_fpga_samples_per_n2k_correlation;
        // the tick number for the start of the current even/odd pair

        if (seq % ticks_per_sample_pair != 0)
            return false;

        // Get the instrument time of the center of the last sample pair.
        int64_t t_start_ns = _tel.to_time_ns(seq);
        int64_t t_cen_ns = _tel.to_time_ns(seq + ticks_per_sample_pair / 2);
        int64_t t_last_cen_ns = t_start_ns - (t_cen_ns - t_start_ns);
        timespec t_last_cen = nanosec_i64_to_timespec(t_last_cen_ns);

        int64_t last_bin_idx = calculate_ERA_bin_idx_from_time(t_last_cen);

        return last_bin_idx != bin_idx;

    } else {
        int64_t fpga_ticks_per_accum =
            _num_n2k_samples_to_accumulate * _n_fpga_samples_per_n2k_correlation;

        return (seq % fpga_ticks_per_accum == 0);
    }
}

void N2Accumulate::accum_corr_and_weight(int32_t* vis_f, float* weight_f, const int32_t* corr_t0_f,
                                         const int32_t* corr_t1_f, double freq_MHz, EOP& target_eop,
                                         EOP& eop_t0, EOP& eop_t1, int32_t count_t0,
                                         int32_t count_t1,
                                         std::vector<std::complex<float>>& fringe_phase_t0,
                                         std::vector<std::complex<float>>& fringe_phase_t1) {

    if (_debug_accum_mode == 0) {
        for (int64_t d = 0; d < 2 * _n2k_correlation_num_products; d++) {
            vis_f[d] += corr_t0_f[d] + corr_t1_f[d];
        }
        for (int64_t d = 0; d < _n2k_correlation_num_products; d++) {
            float dr = corr_t1_f[2 * d + 0] - corr_t0_f[2 * d + 0];
            float di = corr_t1_f[2 * d + 1] - corr_t0_f[2 * d + 1];
            weight_f[d] += dr * dr + di * di;
        }
    } // debug_accum_mode 0
    else if (_debug_accum_mode == 3) {

        uint64_t corr_stride_b = 2 * _n2k_correlation_blocksize * _n2k_correlation_blocksize;
        int num_dishes = _tel.cast<CHORDTelescope>().get_num_dishes();

        // 1/N for even (idx ~ 0) sample
        float inv_n_t0 = (count_t0 <= 0) ? 0.0f : 1.0f / count_t0;
        // 1/N for odd  (idx ~ 1) sample
        float inv_n_t1 = (count_t1 <= 0) ? 0.0f : 1.0f / count_t1;
        // var(corr_e/Ne / corr_o/No) ~ 1/Ne + 1/No
        // 1/var ~ 1/(1/Ne + 1/No) = Ne No / (Ne + No)
        float inv_dvis_var = 0.0f;
        if (count_t0 > 0 && count_t1 > 0) {
            inv_dvis_var = static_cast<float>(count_t0 * count_t1) / (count_t0 + count_t1);
        }

        if (_do_fringestop) {
            // Physical frequency for this f
            // Compute the fringestopping phases for this frequency
            _tel.cast<CHORDTelescope>().fringestop_phases_1d(freq_MHz, eop_t1, target_eop,
                                                             fringe_phase_t1);
            _tel.cast<CHORDTelescope>().fringestop_phases_1d(freq_MHz, eop_t0, target_eop,
                                                             fringe_phase_t0);
        }

        uint64_t block_idx = 0;
        for (int64_t ihi = 0; ihi < _n2k_correlation_lin_blocks; ihi++) {
            for (int64_t jhi = 0; jhi <= ihi; jhi++) {
                // For this stage to run, _num_elements must be a multiple of 64.
                // Since correlation blocksize is 16, there will always be a
                // multiple of 4 correlation_linear_blocks.  So for num_polarization
                // = 2, a block will not cross a polarization boundary, and we're
                // guaranteed all elements in a block will share a polarization.
                uint64_t di0 = _n2k_correlation_blocksize * ihi % num_dishes;
                uint64_t dj0 = _n2k_correlation_blocksize * jhi % num_dishes;
                uint64_t offset_b = block_idx * corr_stride_b;
                uint64_t weight_offset_b = block_idx * corr_stride_b / 2;

                const int32_t* corr_t0_fb = corr_t0_f + offset_b;
                const int32_t* corr_t1_fb = corr_t1_f + offset_b;
                int32_t* vis_fb = vis_f + offset_b;
                float* weight_fb = weight_f + weight_offset_b;

                const std::complex<float>* phase_i = fringe_phase_t1.data() + di0;
                const std::complex<float>* phase_j = fringe_phase_t1.data() + dj0;
                const std::complex<float>* phase_even_i = fringe_phase_t0.data() + di0;
                const std::complex<float>* phase_even_j = fringe_phase_t0.data() + dj0;

                if (_variance_mode == N2VarianceMode::CHIMEv1) {
                    for (int64_t ilo = 0; ilo < _n2k_correlation_blocksize; ilo++) {
                        for (int64_t jlo = 0; jlo < _n2k_correlation_blocksize; jlo++) {

                            uint64_t idx = 2 * (ilo * _n2k_correlation_blocksize + jlo);
                            uint64_t w_idx = ilo * _n2k_correlation_blocksize + jlo;

                            std::complex<float> vis_even{static_cast<float>(corr_t0_fb[idx + 0]),
                                                         static_cast<float>(corr_t0_fb[idx + 1])};
                            std::complex<float> vis_odd{static_cast<float>(corr_t1_fb[idx + 0]),
                                                        static_cast<float>(corr_t1_fb[idx + 1])};

                            if (_do_fringestop) {
                                // To apply phases:
                                //  Fringestop(V_ij) = V_ij * exp{i*(phi_i - phi_j)}
                                //                   = V_ij * Phase_i *
                                //                   conj(Phase_j)
                                std::complex<float> phase_odd =
                                    phase_i[ilo] * std::conj(phase_j[jlo]);
                                std::complex<float> phase_even =
                                    phase_even_i[ilo] * std::conj(phase_even_j[jlo]);

                                vis_odd *= phase_odd;
                                vis_even *= phase_even;
                            }

                            vis_fb[idx + 0] += vis_even.real() + vis_odd.real();
                            vis_fb[idx + 1] += vis_even.imag() + vis_odd.imag();

                            std::complex<float> dvis = vis_odd - vis_even;

                            weight_fb[w_idx] +=
                                dvis.real() * dvis.real() + dvis.imag() * dvis.imag();
                        } // jlo
                    } // ilo
                } else if (_variance_mode == N2VarianceMode::EvenOddPosDef) {
                    for (int64_t ilo = 0; ilo < _n2k_correlation_blocksize; ilo++) {
                        for (int64_t jlo = 0; jlo < _n2k_correlation_blocksize; jlo++) {

                            uint64_t idx = 2 * (ilo * _n2k_correlation_blocksize + jlo);
                            uint64_t w_idx = ilo * _n2k_correlation_blocksize + jlo;

                            std::complex<float> vis_even{static_cast<float>(corr_t0_fb[idx + 0]),
                                                         static_cast<float>(corr_t0_fb[idx + 1])};
                            std::complex<float> vis_odd{static_cast<float>(corr_t1_fb[idx + 0]),
                                                        static_cast<float>(corr_t1_fb[idx + 1])};

                            if (_do_fringestop) {
                                // To apply phases:
                                //  Fringestop(V_ij) = V_ij * exp{i*(phi_i - phi_j)}
                                //                   = V_ij * Phase_i *
                                //                   conj(Phase_j)
                                std::complex<float> phase_odd =
                                    phase_i[ilo] * std::conj(phase_j[jlo]);
                                std::complex<float> phase_even =
                                    phase_even_i[ilo] * std::conj(phase_even_j[jlo]);

                                vis_odd *= phase_odd;
                                vis_even *= phase_even;
                            }

                            vis_fb[idx + 0] += vis_even.real() + vis_odd.real();
                            vis_fb[idx + 1] += vis_even.imag() + vis_odd.imag();

                            std::complex<float> dvis = inv_n_t1 * vis_odd - inv_n_t0 * vis_even;
                            weight_fb[w_idx] +=
                                inv_dvis_var
                                * (dvis.real() * dvis.real() + dvis.imag() * dvis.imag());
                        } // jlo
                    } // ilo
                } // variance_mode
                block_idx++;
            } // jhi
        } // ihi
    } // debug_accum_mode 3
}

bool N2Accumulate::output_and_reset(frameID& in_frame_id, frameID& in_rfiframemask_frame_id, frameID& out_frame_id) {
    [[maybe_unused]] double prof_out_start_time = omp_get_wtime();
    // Different frame for each frequency
    // But, mostly same metadata
    std::shared_ptr<chordMetadata> chord_frame_metadata = get_chord_metadata(in_buf, in_frame_id);
    std::shared_ptr<chordMetadata> rfiframemask_metadata = get_chord_metadata(in_rfiframemask_buf, in_rfiframemask_frame_id);

    int64_t ticks_in_accum = _vis_samples_in_out_frame * _n_fpga_samples_per_n2k_correlation;

    EOP eop_time_center = _tel.get_EOP_at_time(
                _tel.to_time(_accum_fpga_start_tick + ticks_in_accum / 2));
    EOP eop_target = get_accum_bin_EOP(_accum_bin_idx);

    double ERA_deg_start = -1.0;
    double ERA_deg_end = -1.0;
    double ERAL_deg_start = -1.0;
    double ERAL_deg_end = -1.0;

    if (_bin_in_ERA) {
        double era = eop_target.ERA_deg;
        int32_t era_idx = static_cast<int32_t>(floor((era / 360.0) * _num_bins_per_rotation));
        ERA_deg_start = (era_idx * 360.0) / _num_bins_per_rotation;
        ERA_deg_end = ((era_idx+1) * 360.0) / _num_bins_per_rotation;
        ERAL_deg_start = -1; //TODO: update
        ERAL_deg_end = -1;   //TODO: update
    } else {
        ERA_deg_start = _tel.get_EOP_at_time(_tel.to_time(_accum_fpga_start_tick)).ERA_deg;
        ERA_deg_end = _tel.get_EOP_at_time(_tel.to_time(_accum_fpga_start_tick + ticks_in_accum)).ERA_deg;
        ERAL_deg_start = -1; //TODO: update
        ERAL_deg_end = -1;   //TODO: update
    }
           
    int64_t accum_start_time_ns = _tel.to_time_ns(_accum_fpga_start_tick);

    // Grab RFI excision metadata to paste into N2 metadata.
    if (!rfiframemask_metadata->has_rfi_frame_excision_enabled()) 
        FATAL_ERROR("RFIFrameMask buffer {:s}[{:d}] does not have rfi_frame_excision_enabled metadata.", in_rfiframemask_buf->buffer_name, in_rfiframemask_frame_id);
    
    bool rfi_frame_excision_enabled = rfiframemask_metadata->get_rfi_frame_excision_enabled();
    
    if (!rfiframemask_metadata->has_rfi_frame_excision_thresholds()) 
        FATAL_ERROR("RFIFrameMask buffer {:s}[{:d}] does not have rfi_frame_excision_thresholds metadata.", in_rfiframemask_buf->buffer_name, in_rfiframemask_frame_id);

    std::vector<std::array<float, 2>> thresholds_pack = rfiframemask_metadata->get_rfi_frame_excision_thresholds();
    if (thresholds_pack.size() > jsonMetadata::MAX_NUM_RFI_THRESHOLDS) {
        FATAL_ERROR("RFIFrameMask buffer {:s}[{:d}] has too large rfi thresholds buffer {:d}, expected <= {:d}", in_rfiframemask_buf->buffer_name, in_rfiframemask_frame_id, thresholds_pack.size(), jsonMetadata::MAX_NUM_RFI_THRESHOLDS);
    }

    size_t num_thresholds = thresholds_pack.size();
    std::array<float, jsonMetadata::MAX_NUM_RFI_THRESHOLDS> rfi_threshold = {0.0f};
    std::array<float, jsonMetadata::MAX_NUM_RFI_THRESHOLDS> rfi_fraction = {0.0f};
    for (size_t k = 0; k < num_thresholds; k++) {
        rfi_threshold[k] = thresholds_pack[k][0];
        rfi_fraction[k] = thresholds_pack[k][1];
    }

    // strides into the N2K shaped accumulation array
    int64_t stride_ilo = _n2k_correlation_blocksize;
    int64_t stride_block = stride_ilo * _n2k_correlation_blocksize;
    int64_t stride_f = stride_block * _n2k_correlation_num_blocks;

    const int64_t freq_block_size = _output_batch_size;
    const int64_t num_output_workers = std::min(_num_workers, _output_batch_size);
    const int64_t num_freq_blocks =
        (_num_freq_per_n2k_frame + freq_block_size - 1) / freq_block_size;

    std::vector<std::shared_ptr<N2Metadata>> metas(freq_block_size);

    assert(_vis_samples_in_out_frame == _num_n2k_samples_to_accumulate);

    [[maybe_unused]] double prof_out_setup_time = 0;
    [[maybe_unused]] double prof_out_work_time = 0;
    [[maybe_unused]] double prof_out_free_time = 0;

    // Loop over frequency blocks
    for (int64_t fb = 0; fb < num_freq_blocks; fb++) {

        [[maybe_unused]] double prof_out_t0 = omp_get_wtime();
        // Wait for a block of frames to be available.  Grab them and get them metadata.
        for (int64_t f_idx = 0; f_idx < freq_block_size; f_idx++) {
            int64_t f = f_idx + fb * freq_block_size;
            if (f < _num_freq_per_n2k_frame) {
                if (out_buf->wait_for_empty_frame(unique_name, out_frame_id + f_idx) == nullptr) {
                    return false;
                }
                out_buf->allocate_new_metadata_object(out_frame_id + f_idx);
                metas[f_idx] = get_N2_metadata(out_buf, out_frame_id + f_idx);
                DEBUG("Creating N2FrameView for freq f[{:d}] = {:d}", f,
                      chord_frame_metadata->get_coarse_freq()[f]);
                // out_fv[f_idx] = out_vis;
                // out_fv[f_idx]{out_buf, out_frame_id + f_idx};
            } else {
                metas[f_idx] = nullptr;
            }
        } // f_idx
        [[maybe_unused]] double prof_out_t1 = omp_get_wtime();

        // Write the accumulated data to the output frames, and set their metadata.
        // We can do this in parallel!
#pragma omp parallel for num_threads(num_output_workers)
        for (int64_t f_idx = 0; f_idx < freq_block_size; f_idx++) {
            int64_t f = f_idx + fb * freq_block_size;
            if (f >= _num_freq_per_n2k_frame)
                continue;

            std::shared_ptr<N2Metadata> meta = metas[f_idx];

            meta->freq_id = chord_frame_metadata->get_coarse_freq()[f];
            meta->freq_MHz = _tel.to_freq_MHz(meta->freq_id);
            
            meta->abs_time_idx = _accum_bin_idx;

            meta->time_center_eop = eop_time_center;
            meta->bin_eop = eop_target;
            
            meta->bin_start_ERA_deg = ERA_deg_start;
            meta->bin_end_ERA_deg = ERA_deg_end;
            meta->bin_start_ERAL = ERAL_deg_start; // TODO: update
            meta->bin_end_ERAL = ERAL_deg_end;   // TODO: update

            meta->fpga_start_tick = _accum_fpga_start_tick;
            meta->frame_start_time_ns = accum_start_time_ns;
            meta->frame_length_fpga_ticks = ticks_in_accum;
            meta->n_valid_fpga_ticks = _n_valid_fpga_samples_in_vis[f];
            meta->n_rfi_fpga_ticks = _n_rfi_samples_in_vis[f];
            
            meta->rfi_frame_excision_enabled = rfi_frame_excision_enabled;
            meta->rfi_frame_excision_num = num_thresholds;
            meta->rfi_frame_excision_threshold = rfi_threshold;
            meta->rfi_frame_excision_fraction = rfi_fraction;

            std::vector<std::array<float, 2>> thresholds = rfiframemask_metadata->get_rfi_frame_excision_thresholds();
            
            if (chord_frame_metadata->has_dataset_id()) {
                meta->dataset_id = chord_frame_metadata->get_dataset_id();
            }
            N2FrameView out_vis(out_buf, out_frame_id + f_idx);

            // Sample numbers for normalizing weights
            int64_t ns = _n_valid_fpga_samples_in_vis[f]; // ns = "number of samples"
            float ins = (ns != 0) ? (1.0f / ((float)ns)) : 0.0f;

            // Copy data into buffer.
            // This requires changing from the GPU's blocked format to the triangular format
            // N2Buffer expects.

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

                            // Get the index into the n2 view.  N2 is an *upper* triangular
                            // unblocked form, so we use the global matrix indices to the
                            // lower-triangular N2K form, and compute the triangular index with
                            // their transpose.
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

                            if (_variance_mode == N2VarianceMode::CHIMEv1) {
                                float bias =
                                    std::norm(v) * _n_valid_sample_diff_sq_sum[f] * ins * ins;


                                // DEBUG("{} {} w1: {:.12f} v2: {:.12f} |v|^2: {:.12f}, dN^2:
                                // {:.12f} |v|^2 dN^2: {:.12f} diff: {:.12e} ==: {}", i, j,
                                // _weights[idx], std::norm(v), std::norm(v)*ins*ins,
                                // _n_valid_sample_diff_sq_sum[f], bias, _weights[idx] - bias,
                                // _weights[idx] == bias);

                                if (ns > 0) {
                                    _weights[idx] -= bias;
                                }

                                // DEBUG("{} {} w2: {}", i, j, _weights[idx]);

                                out_vis.weight[n2_idx] = (ns > 0 && _weights[idx] != 0.0f)
                                                             ? ns * (ns / _weights[idx])
                                                             : 0.0f;
                                // DEBUG("{} {} w3: {}", i, j, out_vis.weight[n2_idx]);
                            } else if (_variance_mode == N2VarianceMode::EvenOddPosDef) {
                                if (_vis_samples_in_out_frame % 2 != 0) {
                                    FATAL_ERROR(
                                        "EvenOdd variance estimator requires an even number of "
                                        "frames, got: {} for accumulation {} at seq {}",
                                        _vis_samples_in_out_frame, _accum_bin_idx,
                                        _accum_fpga_start_tick);
                                }
                                assert(_vis_samples_in_out_frame % 2 == 0);

                                int64_t num_var_samp = _vis_samples_in_out_frame / 2;
                                int64_t norm = ns * num_var_samp;
                                out_vis.weight[n2_idx] =
                                    (norm > 0 && _weights[idx] > 0) ? norm / _weights[idx] : 0.0f;
                            } else {
                                FATAL_ERROR("Cannot output weights for variance_mode: {}",
                                            _variance_mode);
                            }
                        } // jlo
                    } // ilo

                    block_idx++;
                } // jhi
            } // ihi

            out_vis.erms = -1;
            out_vis.radiometer_chi2 = 1.0f;
            std::fill(out_vis.flags.begin(), out_vis.flags.end(), 0);
            std::fill(out_vis.gain.begin(), out_vis.gain.end(), N2::cfloat{-1.0f, 0.0f});
            std::fill(out_vis.mask.begin(), out_vis.mask.end(), static_cast<uint8_t>(1u));
        } // f_idx

        [[maybe_unused]] double prof_out_t2 = omp_get_wtime();

        // All the frames in the block are full. Release them and increment out_frame_id;
        for (int64_t f_idx = 0; f_idx < freq_block_size; f_idx++) {
            int64_t f = f_idx + fb * freq_block_size;
            if (f < _num_freq_per_n2k_frame)
                out_buf->mark_frame_full(unique_name, out_frame_id++);
        } // f_idx

        [[maybe_unused]] double prof_out_t3 = omp_get_wtime();

        prof_out_setup_time += prof_out_t1 - prof_out_t0;
        prof_out_work_time += prof_out_t2 - prof_out_t1;
        prof_out_free_time += prof_out_t3 - prof_out_t2;

    } // fb

    DEBUG("Wrapping up accumulation buffer output copy.");

    [[maybe_unused]] double prof_out_fill_time = omp_get_wtime();

    // _vis and _weights are large, 0 them in parallel.
#pragma omp parallel for simd num_threads(_num_workers)
    for (uint64_t i = 0; i < _vis.size(); i++)
        _vis[i] = 0;

#pragma omp parallel for simd num_threads(_num_workers)
    for (uint64_t i = 0; i < _weights.size(); i++)
        _weights[i] = 0.0f;

    // These arrays are smaller, single threaded is fine.
    std::fill(_n_valid_fpga_samples_in_vis.begin(), _n_valid_fpga_samples_in_vis.end(), 0);
    std::fill(_n_valid_sample_diff_sq_sum.begin(), _n_valid_sample_diff_sq_sum.end(), 0);
    std::fill(_n_rfi_samples_in_vis.begin(), _n_rfi_samples_in_vis.end(), 0);

    [[maybe_unused]] double prof_out_end_time = omp_get_wtime();

    INFO("Outputting {:d} frames took {:f} ms\n    setup: {:f} ms\n    work:  {:f} ms\n    free:  "
         "{:f} ms\n    fill:  {:f} ms",
         _num_freq_per_n2k_frame, 1000 * (prof_out_end_time - prof_out_start_time),
         1000 * prof_out_setup_time, 1000 * prof_out_work_time, 1000 * prof_out_free_time,
         1000 * (prof_out_end_time - prof_out_fill_time));

    return true;
}

std::string N2VarianceMode_to_string(const N2VarianceMode& m) {
    switch (m) {
        case N2VarianceMode::CHIMEv1:
            return "CHIMEv1";
        case N2VarianceMode::EvenOddPosDef:
            return "EvenOddPosDef";
        default:
            FATAL_ERROR_NON_OO("Unknown N2VarianceMode: {:d}", static_cast<uint32_t>(m));
    }
}

N2VarianceMode N2VarianceMode_from_string(const std::string& s) {
    if (s == "CHIMEv1") {
        return N2VarianceMode::CHIMEv1;
    } else if (s == "EvenOddPosDef") {
        return N2VarianceMode::EvenOddPosDef;
    } else {
        FATAL_ERROR_NON_OO(R"(Could not parse "{:s}" into N2VarianceMode)", s);
    }
}

std::ostream& operator<<(std::ostream& os, const N2VarianceMode& m) {
    os << N2VarianceMode_to_string(m);
    return os;
}

std::string format_as(const N2VarianceMode& m) {
    return N2VarianceMode_to_string(m);
}

void to_json(nlohmann::json& j, const N2VarianceMode& m) {
    j = N2VarianceMode_to_string(m);
}

void from_json(const nlohmann::json& j, N2VarianceMode& m) {
    m = N2VarianceMode_from_string(j);
}
