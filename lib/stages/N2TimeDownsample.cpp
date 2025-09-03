#include "N2TimeDownsample.hpp"

#include "CHORDTelescope.hpp"    // for CHORDTelescope
#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for mark_frame_empty, allocate_new_metadata_object, mark_fr...
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Metrics
// #include "visBuffer.hpp"         // for VisFrameView
#include "N2FrameView.hpp"
#include "timeUtil.hpp" // for get_ERA_from_UT1, get_UT1_from_ERA
#include "visUtil.hpp"  // for frameID, modulo, cfloat, operator-, ts_to_double

#include "gsl-lite.hpp" // for span

#include <atomic>     // for atomic_bool
#include <complex>    // for complex
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, uint64_t, int32_t
#include <time.h>     // for timespec
#include <tuple>      // for get
#include <vector>     // for vector

#define GIGA 1'000'000'000L

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(N2TimeDownsample);

N2TimeDownsample::N2TimeDownsample(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2TimeDownsample::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Get the number of bins per earth rotation (sidereal day)
    num_bins_per_rotation = config.get<uint32_t>(unique_name, "num_bins_per_rotation");

    max_age = config.get_default<float>(unique_name, "max_age", 120.0);

    do_fringestop = config.get_default<bool>(unique_name, "do_fringestop", true);

    num_elements = 0;
    nprod = 0;
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
    double freq_Hz = -1.0;

    // Whether we're waiting for an accumulation bin to start.
    bool wait_for_alignment = true;

    auto& skipped_frame_counter = Metrics::instance().add_counter(
        "kotekan_timedownsample_skipped_frame_total", unique_name, {"freq_id", "reason"});

    const CHORDTelescope& tel = Telescope::instance().cast<CHORDTelescope>();

    // Make an array to hold the per-dish phases.
    int num_dishes = tel.get_num_dishes();
    std::vector<std::complex<double>> fringe_phase(num_dishes, 1.0);

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
            freq_Hz = frame.freq_Hz;
            nprod = frame.num_prod;
            num_elements = frame.num_elements;
            num_eigenvectors = frame.num_ev;

            era_bin_idx = (uint32_t)(frame.eop.ERA_deg / era_bin_width);
            era_deg_lo = era_bin_idx * era_bin_width;
            era_deg_hi = (era_bin_idx + 1) * era_bin_width;

            double era_deg_target = 0.5 * (era_deg_lo + era_deg_hi);

            // Initialize num_rotations from first frame.
            get_ERA_from_UT1(frame.eop.t_ut1, &num_rotations);

            // Get UT1 time at target ERA
            int64_t t_ut1 = get_UT1_from_ERA(num_rotations, era_deg_target);

            // Set EOP at target ERA
            eop_target = tel.get_EOP_at_UT1(t_ut1);

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
        uint32_t new_era_bin_idx = (uint32_t)(frame.eop.ERA_deg / era_bin_width);
        if (new_era_bin_idx != era_bin_idx)
            wait_for_alignment = false;


        DEBUG("T:   {:d}s + {:d}ns", frame.eop.t_inst / GIGA, frame.eop.t_inst % GIGA);
        DEBUG("UT1: {:d}s + {:d}ns", frame.eop.t_ut1 / GIGA, frame.eop.t_ut1 % GIGA);
        DEBUG("ERA: {:f}; ERA_target: {:f}; ERA_bin_lo: {:f}; ERA_bin_hi: {:f}", frame.eop.ERA_deg,
              eop_target.ERA_deg, era_deg_lo, era_deg_hi);

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
            era_bin_idx = (uint32_t)(frame.eop.ERA_deg / era_bin_width);
            era_deg_lo = era_bin_idx * era_bin_width;
            era_deg_hi = (era_bin_idx + 1) * (360.0 / num_bins_per_rotation);

            double era_deg_target = 0.5 * (era_deg_lo + era_deg_hi);

            // Get the current num_rotations from frame.
            get_ERA_from_UT1(frame.eop.t_ut1, &num_rotations);

            // Get UT1 time at target ERA
            int64_t t_ut1 = get_UT1_from_ERA(num_rotations, era_deg_target);

            // Set EOP at target ERA / UT1
            eop_target = tel.get_EOP_at_UT1(t_ut1);

            // Wait for an empty frame
            if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
                break;
            }

            // Copy frame into output buffer
            auto output_frame = N2FrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id);

            // Set the output to target EOP.
            output_frame.eop = eop_target;

            //Initialize the weights, and weigh vis/weight by number of samples.
            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] = 1. / output_frame.weight[i];
                output_frame.vis[i] *= output_frame.n_valid_fpga_ticks_in_frame;
                output_frame.weight[i] *= output_frame.n_valid_fpga_ticks_in_frame;
            }

            if (do_fringestop) {
                // Get the per-dish fringestopping phases.
                tel.fringestop_phases_1d(freq_Hz, frame.eop, eop_target, fringe_phase);

                // This indexing requires the el_id = (n_dish)*pol_id + dish_id
                size_t idx = 0;
                for (size_t i = 0; i < num_elements; i++) {
                    for (size_t j = i; j < num_elements; j++) {

                        // To apply phases:
                        //  Fringestop(V_ij) = V_ij * exp{i*(phi_i - phi_j)}
                        //                   = V_ij * Phase_i * conj(Phase_j)

                        size_t d_i = i % num_dishes;
                        size_t d_j = j % num_dishes;
                        output_frame.vis[idx] *= fringe_phase[d_i] * std::conj(fringe_phase[d_j]);

                        idx++;
                    }
                }
            }

            // evec and eval averages are weighted by number of valid samples
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] *= output_frame.n_valid_fpga_ticks_in_frame;
                for (uint32_t j = 0; j < num_elements; j++) {
                    // Eigenvectors get phases too.
                    int k = i * num_elements + j;
                    size_t d_j = j % num_dishes;
                    output_frame.evec[k] *= fringe_phase[d_j];
                    output_frame.evec[k] *= output_frame.n_valid_fpga_ticks_in_frame;
                }
            }
            output_frame.erms *= output_frame.n_valid_fpga_ticks_in_frame;

            // Go to next frame
            nframes += 1;
            in_buf->mark_frame_empty(unique_name, frame_id++);
            continue;
        }

        // If we're here, an accumulation has started.

        auto output_frame = N2FrameView(out_buf, output_frame_id);

        // Check we are still in accumulation window
        if (frame.eop.ERA_deg >= era_deg_lo && frame.eop.ERA_deg < era_deg_hi) {

            // Recalculate fringestop phases
            if (do_fringestop)
                tel.fringestop_phases_1d(freq_Hz, frame.eop, eop_target, fringe_phase);

            // Accumulate contents of buffer
            size_t idx = 0;
            for (size_t i = 0; i < num_elements; i++) {
                for (size_t j = i; j < num_elements; j++) {

                    size_t d_i = i % num_dishes;
                    size_t d_j = j % num_dishes;

                    // Computing the total phase in double precision
                    // in case one of the dish phases is small.
                    // Adding the weighting by valid samples here as well.
                    std::complex<double> w_doub = (fringe_phase[d_i] * std::conj(fringe_phase[d_j])) * ((double) frame.n_valid_fpga_ticks_in_frame);

                    // Now truncate the phase to a float to match vis[]
                    // Have to be explicit about this, compiler complains
                    // otherwise.
                    N2::cfloat w{(float)w_doub.real(), (float)w_doub.imag()};

                    // Accumulate
                    output_frame.vis[idx] += w * frame.vis[idx];
                    idx++;
                }
            }

            // average inverse weights, i.e. variance
            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] += frame.n_valid_fpga_ticks_in_frame / frame.weight[i];
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] += frame.eval[i] * frame.n_valid_fpga_ticks_in_frame;
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    size_t d_j = j % num_dishes;
                    N2::cfloat phase{(float)fringe_phase[d_j].real(),
                                     (float)fringe_phase[d_j].imag()};
                    output_frame.evec[k] += frame.evec[k] * phase * ((float) frame.n_valid_fpga_ticks_in_frame);
                }
            }
            output_frame.erms += frame.erms * frame.n_valid_fpga_ticks_in_frame;

            // Accumulate integration totals
            output_frame.n_valid_fpga_ticks_in_frame += frame.n_valid_fpga_ticks_in_frame;
            output_frame.n_rfi_fpga_ticks += frame.n_rfi_fpga_ticks;
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
            for (size_t i = 0; i < nprod; i++) {
                output_frame.vis[i] /= output_frame.n_valid_fpga_ticks_in_frame;
                // extra factor of nsamp for sample variance
                output_frame.weight[i] = output_frame.n_valid_fpga_ticks_in_frame * nframes / output_frame.weight[i];
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] /= output_frame.n_valid_fpga_ticks_in_frame;
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] /= output_frame.n_valid_fpga_ticks_in_frame;
                }
            }
            output_frame.erms /= output_frame.n_valid_fpga_ticks_in_frame;

            DEBUG("Output frame - num_elements: {:d}", output_frame.num_elements);
            DEBUG("Output T:   {:d}s + {:d}ns", output_frame.eop.t_inst / GIGA,
                  output_frame.eop.t_inst % GIGA);
            DEBUG("Output UT1: {:d}s + {:d}ns", output_frame.eop.t_ut1 / GIGA,
                  output_frame.eop.t_ut1 % GIGA);
            DEBUG("Output ERA: {:f}", output_frame.eop.ERA_deg);

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
