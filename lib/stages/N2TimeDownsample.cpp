#include "N2TimeDownsample.hpp"

#include <stdint.h>               // for uint32_t, int64_t, uint64_t, int32_t
#include <time.h>                 // for timespec
#include <algorithm>              // for equal
#include <complex>                // for complex, operator*, conj
#include <functional>             // for bind, function
#include <stdexcept>              // for runtime_error
#include <vector>                 // for vector
#include <memory>                 // for shared_ptr, __shared_ptr_access

#include "Config.hpp"             // for Config
#include "N2FrameView.hpp"        // for N2FrameView
#include "N2Util.hpp"             // for frameID, modulo
#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"          // for Telescope, ElementOrder
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "geoUtil.hpp"            // for vec3d_t
#include "kotekanLogging.hpp"     // for DEBUG, FATAL_ERROR, ERROR
#include "prometheusMetrics.hpp"  // for Counter, MetricFamily, Metrics
#include "timeUtil.hpp"           // for EOP, get_ERA_from_UT1, get_UT1_from_ERA, eop_null
#include "fmt.hpp"                // for compile_string_to_view
#include "gsl-lite.hpp"           // for span
#include "FrameDesc.hpp"          // for FrameDesc

#define GIGA 1'000'000'000L

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(N2TimeDownsample);

N2TimeDownsample::N2TimeDownsample(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2TimeDownsample::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Validate that input and output buffers have compatible N2 frame descriptors
    auto in_desc = in_buf->get_frame_desc();
    auto out_desc = out_buf->get_frame_desc();
    if (!in_desc || !out_desc) {
        FATAL_ERROR("N2TimeDownsample: Input and output buffers must have frame descriptors set");
    }
    if (*in_desc != *out_desc) {
        FATAL_ERROR("N2TimeDownsample: Input and output buffer frame descriptors must match");
    }

    // Get the number of bins per earth rotation (sidereal day)
    num_bins_per_rotation = config.get<uint32_t>(unique_name, "num_bins_per_rotation");

    max_age = config.get_default<float>(unique_name, "max_age", 120.0);

    do_fringestop = config.get_default<bool>(unique_name, "do_fringestop", true);

    num_elements = config.get<size_t>(unique_name, "num_elements");
    nprod = 0;

    input_order = config.get<ElementOrder>(unique_name, "input_order");
    feed_positions_m = Telescope::instance().get_feed_positions_m(num_elements, input_order);
}

void N2TimeDownsample::main_thread() {

    frameID frame_id(in_buf);
    frameID output_frame_id(out_buf);
    unsigned int nframes = 0; // the number of frames accumulated so far

    uint32_t era_bin_idx = 0;  // index of the current ERA bin,
                               // in [0, num_bins_per_rotation-1]
    double era_deg_lo = 0.0;   // lower bound of current ERA bin, in degrees
    double era_deg_hi = 360.0; // upper bound of current ERA bin, in degrees
    double era_bin_width = 360.0 / num_bins_per_rotation;

    struct EOP eop_target = eop_null; // EOP at center of current bin.

    int64_t num_rotations = -1;
    int32_t freq_id = -1; // needs to be set by first frame
    double freq_MHz = -1.0;

    // Whether we're waiting for an accumulation bin to start.
    bool wait_for_alignment = true;

    auto& skipped_frame_counter = Metrics::instance().add_counter(
        "kotekan_timedownsample_skipped_frame_total", unique_name, {"freq_id", "reason"});

    const Telescope& tel = Telescope::instance();


    // Calculate the startup (FPGA seq = 0) EOP and ERA index, for absolute time indexing
    timespec ts_startup = tel.to_time(0);
    struct EOP eop_startup = tel.get_EOP_at_time(ts_startup);

    uint64_t era_bin_idx_startup = (uint64_t)(eop_startup.ERA_deg / era_bin_width);
    int64_t num_rotations_startup;
    get_ERA_from_UT1(eop_startup.t_ut1_ns, &num_rotations_startup);

    uint64_t prev_abs_time_idx = 0;

    // Make an array to hold the per-element phases.
    std::vector<std::complex<float>> fringe_phase(num_elements, 1.0);

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((in_buf->wait_for_full_frame(unique_name, frame_id)) == nullptr) {
            break;
        }

        N2FrameView frame(in_buf, frame_id);

        DEBUG("Input frame - num_elements: {:d} - t: {:d}", frame.num_elements,
              frame.frame_start_time_ns);

        // The first frame
        if (freq_id == -1) {

            // Get parameters from first frame
            freq_id = frame.freq_id;
            freq_MHz = frame.freq_MHz;
            nprod = frame.num_prod;
            num_eigenvectors = frame.num_ev;

            era_bin_idx = (uint32_t)(frame.bin_eop.ERA_deg / era_bin_width);
            era_deg_lo = era_bin_idx * era_bin_width;
            era_deg_hi = (era_bin_idx + 1) * era_bin_width;

            double era_deg_target = 0.5 * (era_deg_lo + era_deg_hi);

            // Initialize num_rotations from first frame.
            get_ERA_from_UT1(frame.bin_eop.t_ut1_ns, &num_rotations);

            // Get UT1 time at target ERA
            int64_t t_ut1_ns = get_UT1_from_ERA(num_rotations, era_deg_target);

            // Set EOP at target ERA
            eop_target = tel.get_EOP_at_UT1(t_ut1_ns);

            // Check ERA at the beginning of the frame. If it's earlier than
            // the bin edge, we'll fill the bin and can start integrating
            // immediately.
            /*
            timespec frame_start = {.tv_sec = (time_t)(frame.frame_start_time_ns / 1'000'000'000),
                                    .tv_nsec = (long)(frame.frame_start_time_ns % 1'000'000'000)};
            struct EOP eop_start = tel.get_EOP_at_time(frame_start);

            if (eop_start.ERA_deg < era_deg_lo || eop_start.ERA_deg >= era_deg_hi)
                wait_for_alignment = false;
            */
        }

        // Check that this is the frequency we care about,
        // throw runtime error if not.
        if (frame.freq_id != (unsigned)freq_id) {
            throw std::runtime_error("Cannot downsample stream with more than one frequency.");
        }


        // Check if bin index has changed. If so, we're at the start of a
        // new bin.
        uint32_t new_era_bin_idx = (uint32_t)(frame.bin_eop.ERA_deg / era_bin_width);
        if (new_era_bin_idx != era_bin_idx)
            wait_for_alignment = false;


        DEBUG("T:   {:d}s + {:d}ns", frame.bin_eop.t_inst_ns / GIGA,
              frame.bin_eop.t_inst_ns % GIGA);
        DEBUG("UT1: {:d}s + {:d}ns", frame.bin_eop.t_ut1_ns / GIGA, frame.bin_eop.t_ut1_ns % GIGA);
        DEBUG("ERA: {:f}; ERA_target: {:f}; ERA_bin_lo: {:f}; ERA_bin_hi: {:f}",
              frame.bin_eop.ERA_deg, eop_target.ERA_deg, era_deg_lo, era_deg_hi);

        // Don't start accumulating unless at the start of window
        if (nframes == 0 && wait_for_alignment) {
            // Skip this frame
            skipped_frame_counter.labels({std::to_string(freq_id), "alignment"}).inc();
            in_buf->mark_frame_empty(unique_name, frame_id++);
            continue;
        } else if (nframes == 0) { // Start a new accumulation

            // We know we're in a fresh bin now.
            wait_for_alignment = false;

            // Update the window.
            era_bin_idx = (uint32_t)(frame.bin_eop.ERA_deg / era_bin_width);
            era_deg_lo = era_bin_idx * era_bin_width;
            era_deg_hi = (era_bin_idx + 1) * (360.0 / num_bins_per_rotation);

            double era_deg_target = 0.5 * (era_deg_lo + era_deg_hi);

            // Get the current num_rotations from frame.
            get_ERA_from_UT1(frame.bin_eop.t_ut1_ns, &num_rotations);

            // Get UT1 time at target ERA
            int64_t t_ut1_ns = get_UT1_from_ERA(num_rotations, era_deg_target);

            // Set EOP at target ERA / UT1
            eop_target = tel.get_EOP_at_UT1(t_ut1_ns);

            // Wait for an empty frame
            if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
                break;
            }

            // Copy frame into output buffer
            auto output_frame = N2FrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);

            // Set the output to target EOP.
            output_frame.time_center_eop = eop_null; // TODO: set properly later
            output_frame.bin_eop = eop_target;
            output_frame.bin_start_ERA_deg = era_deg_lo;
            output_frame.bin_end_ERA_deg = era_deg_hi;
            output_frame.bin_start_ERAL_deg = -1; // TODO: update
            output_frame.bin_end_ERAL_deg = -1;   // TODO: update

            // Set the output absolute time index, from the difference between the current ERA bin
            // and the ERA bin at startup
            output_frame.abs_time_idx =
                (era_bin_idx + num_bins_per_rotation * (num_rotations - num_rotations_startup))
                - era_bin_idx_startup;

            // Check that the input isn't coming too fast.
            if (prev_abs_time_idx > 0 && output_frame.abs_time_idx != prev_abs_time_idx + 1) {
                ERROR("New downsampled frame will have abs_time_idx = {:d}, expected previous "
                      "abs_time_idx {:d} + 1.",
                      output_frame.abs_time_idx, prev_abs_time_idx);
            }
            prev_abs_time_idx = output_frame.abs_time_idx;

            // Initialize the weights, and weigh vis/weight by number of samples.
            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] =
                    (output_frame.weight[i] != 0.0f) ? 1. / output_frame.weight[i] : 0.0f;
                output_frame.vis[i] *= output_frame.n_valid_fpga_ticks;
                output_frame.weight[i] *= output_frame.n_valid_fpga_ticks;
            }

            if (do_fringestop) {
                // Get the per-dish fringestopping phases.
                tel.fill_fringestop_phases_1d(freq_MHz, frame.bin_eop, eop_target, feed_positions_m, fringe_phase);

                size_t idx = 0;
                for (size_t i = 0; i < num_elements; i++) {
                    for (size_t j = i; j < num_elements; j++) {

                        // To apply phases:
                        //  Fringestop(V_ij) = V_ij * exp{i*(phi_i - phi_j)}
                        //                   = V_ij * Phase_i * conj(Phase_j)

                        output_frame.vis[idx] *= fringe_phase[i] * std::conj(fringe_phase[j]);

                        idx++;
                    }
                }
            }

            // evec and eval averages are weighted by number of valid samples
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] *= output_frame.n_valid_fpga_ticks;
                for (uint32_t j = 0; j < num_elements; j++) {
                    // Eigenvectors get phases too, but are computed to always
                    // have 0 phase in first element, so ensure the first
                    // element in each is unchanged.
                    int k = i * num_elements + j;
                    output_frame.evec[k] *= fringe_phase[j] * std::conj(fringe_phase[0]);
                    output_frame.evec[k] *= output_frame.n_valid_fpga_ticks;
                }
            }
            output_frame.erms *= output_frame.n_valid_fpga_ticks;

            // Go to next frame
            nframes += 1;
            in_buf->mark_frame_empty(unique_name, frame_id++);
            continue;
        }

        // If we're here, an accumulation has started.

        auto output_frame = N2FrameView(out_buf, output_frame_id);

        // Check we are still in accumulation window
        if (frame.bin_eop.ERA_deg >= era_deg_lo && frame.bin_eop.ERA_deg < era_deg_hi) {

            // Recalculate fringestop phases
            if (do_fringestop)
                tel.fill_fringestop_phases_1d(freq_MHz, frame.bin_eop, eop_target, feed_positions_m, fringe_phase);
            // Accumulate contents of buffer
            size_t idx = 0;
            for (size_t i = 0; i < num_elements; i++) {
                for (size_t j = i; j < num_elements; j++) {

                    // Weight by number of valid samples and include a fringe-stopping phase
                    // (will be 1.0 if do_fringestop is false)
                    std::complex<float> w = (fringe_phase[i] * std::conj(fringe_phase[j]))
                                            * ((float)frame.n_valid_fpga_ticks);

                    // Accumulate
                    output_frame.vis[idx] += w * frame.vis[idx];
                    idx++;
                }
            }

            // average inverse weights, i.e. variance
            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] +=
                    (frame.weight[i] != 0.0f) ? frame.n_valid_fpga_ticks / frame.weight[i] : 0.0f;
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] += frame.eval[i] * frame.n_valid_fpga_ticks;
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    // Ensure 1st element of each evec doesn't get a phase.
                    std::complex<float> phase = fringe_phase[j] * std::conj(fringe_phase[0]);
                    output_frame.evec[k] +=
                        frame.evec[k] * phase * ((float)frame.n_valid_fpga_ticks);
                }
            }
            output_frame.erms += frame.erms * frame.n_valid_fpga_ticks;

            // Accumulate integration totals
            output_frame.n_valid_fpga_ticks += frame.n_valid_fpga_ticks;
            output_frame.n_rfi_fpga_ticks += frame.n_rfi_fpga_ticks;
            output_frame.n_rfi_only_fpga_ticks += frame.n_rfi_only_fpga_ticks;
            output_frame.n_pl_fpga_ticks += frame.n_pl_fpga_ticks;
            output_frame.frame_length_fpga_ticks += frame.frame_length_fpga_ticks;

            // Move to next frame
            nframes += 1;
            in_buf->mark_frame_empty(unique_name, frame_id++);

        } else {

            double output_age =
                1.0e-9 * (frame.frame_start_time_ns - output_frame.frame_start_time_ns);
            if (output_age > max_age) {
                DEBUG("Skipping - age {:g} > max_age {:g}", output_age, max_age);
                skipped_frame_counter.labels({std::to_string(freq_id), "age"}).inc();
                nframes = 0;
                continue;
            }

            // Otherwise, stop accumulating
            if (output_frame.n_valid_fpga_ticks > 0) {
                for (size_t i = 0; i < nprod; i++) {

                    output_frame.vis[i] /= output_frame.n_valid_fpga_ticks;
                    // extra factor of nsamp for sample variance
                    output_frame.weight[i] =
                        output_frame.n_valid_fpga_ticks * nframes / output_frame.weight[i];
                }
                for (uint32_t i = 0; i < num_eigenvectors; i++) {
                    output_frame.eval[i] /= output_frame.n_valid_fpga_ticks;
                    for (uint32_t j = 0; j < num_elements; j++) {
                        int k = i * num_elements + j;
                        output_frame.evec[k] /= output_frame.n_valid_fpga_ticks;
                    }
                }
                output_frame.erms /= output_frame.n_valid_fpga_ticks;
            } else {
                // Likely redundant, if we're here, only 0's could have
                //  been added to these anyways.
                for (size_t i = 0; i < nprod; i++) {
                    output_frame.vis[i] = 0.0f;
                    output_frame.weight[i] = 0.0f;
                }
                for (uint32_t i = 0; i < num_eigenvectors; i++) {
                    output_frame.eval[i] = 0.0f;
                    for (uint32_t j = 0; j < num_elements; j++) {
                        int k = i * num_elements + j;
                        output_frame.evec[k] = 0.0f;
                    }
                }
                output_frame.erms = 0.0f;
            }

            DEBUG("Output frame - num_elements: {:d}", output_frame.num_elements);
            DEBUG("Output T:   {:d}s + {:d}ns", output_frame.bin_eop.t_inst_ns / GIGA,
                  output_frame.bin_eop.t_inst_ns % GIGA);
            DEBUG("Output UT1: {:d}s + {:d}ns", output_frame.bin_eop.t_ut1_ns / GIGA,
                  output_frame.bin_eop.t_ut1_ns % GIGA);
            DEBUG("Output ERA: {:f}", output_frame.bin_eop.ERA_deg);

            char* addr0 = (char*)&(output_frame.vis[0]);
            DEBUG("Output frame Struct - vis: {}, w: {}, flags: {}, eval: {}, evec: {}, emeth: {}, "
                  "erms: {}, gain: {}",
                  (char*)&(output_frame.vis[0]) - addr0, (char*)&(output_frame.weight[0]) - addr0,
                  (char*)&(output_frame.flags[0]) - addr0, (char*)&(output_frame.eval[0]) - addr0,
                  (char*)&(output_frame.evec[0]) - addr0, (char*)&(output_frame.emethod) - addr0,
                  (char*)&(output_frame.erms) - addr0, (char*)&(output_frame.gain[0]) - addr0);
            // mark as full
            out_buf->mark_frame_full(unique_name, output_frame_id++);
            // reset accumulation and move on, starting with this frame
            nframes = 0;
        }
    }
}
