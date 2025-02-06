#include "N2TimeDownsample.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for mark_frame_empty, allocate_new_metadata_object, mark_fr...
#include "bufferContainer.hpp"   // for bufferContainer
#include "CHORDTelescope.hpp" // for CHORDTelescope
#include "kotekanLogging.hpp"    // for DEBUG
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Metrics
//#include "visBuffer.hpp"         // for VisFrameView
#include "N2FrameView.hpp"
#include "visUtil.hpp"           // for frameID, modulo, cfloat, operator-, ts_to_double

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


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(N2TimeDownsample);

N2TimeDownsample::N2TimeDownsample(Config& config,
                                   const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&N2TimeDownsample::main_thread, this)) {

    // Fetch the buffers, register
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Get the number of bins per earth rotation (sidereal day)
    num_bins_per_rotation = config.get_default<uint32_t>(unique_name,
                        "num_bins_per_rotation", 86400);
    
    max_age = config.get_default<float>(unique_name, "max_age", 120.0);

    do_fringestop = config.get_default<int>(unique_name, "do_fringestop",
                                         true);

    num_elements = 0;
    nprod = num_elements * (num_elements + 1) / 2;
}

void N2TimeDownsample::main_thread() {

    frameID frame_id(in_buf);
    frameID output_frame_id(out_buf);
    unsigned int nframes = 0; // the number of frames accumulated so far
    
    unsigned int era_bin_idx = 0; // index of the current ERA bin,
                                  // in [0, num_bins_per_rotation-1]
    double era_deg_lo = 0.0;      // lower bound of current ERA bin, in degrees
    double era_deg_hi = 360.0;    // upper bound of current ERA bin, in degrees
    double era_target = -1.0;     // Center of ERA bin, time to which we're
                                  // fringestopping.
    double xp_target = 0.0;
    double yp_target = 0.0;

    int32_t freq_id = -1; // needs to be set by first frame
    double freq_Hz = -1.0;


    auto& skipped_frame_counter = Metrics::instance().add_counter(
        "kotekan_timedownsample_skipped_frame_total", unique_name, {"freq_id", "reason"});
    
    const CHORDTelescope& tel = Telescope::instance().cast<CHORDTelescope>();

    int num_dishes = tel.get_num_dishes();
    std::vector<std::complex<double>> fringe_phase(num_dishes, 1.0);

    //uint64_t seq_len_ns = tel.seq_length_nsec();

    //double seq_to_era = 1.0e-9 * seq_len_ns * 360.0 * 1.00273781191135448 / 86400.0;

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if ((in_buf->wait_for_full_frame(unique_name, frame_id)) == nullptr) {
            break;
        }

        N2FrameView frame(in_buf, frame_id);
        uint64_t fpga_seq_start = frame.fpga_start_tick;
        
        DEBUG("Input frame - num_elements: {:d}", frame.num_elements);

        // The first frame
        if (freq_id == -1) {

            // Get parameters from first frame
            freq_id = frame.freq_id;
            freq_Hz = frame.freq_Hz;
            nprod = frame.num_prod;
            num_elements = frame.num_elements;
            num_eigenvectors = frame.num_ev;
            
            era_bin_idx = (unsigned int) (num_bins_per_rotation
                                      * frame.era_deg / 360.0);
            era_deg_lo = era_bin_idx * (360.0 / num_bins_per_rotation);
            era_deg_hi = (era_bin_idx + 1) * (360.0 / num_bins_per_rotation);
            era_target = 0.5*(era_deg_lo + era_deg_hi);
        }

        // Check that this is the frequency we care about,
        // throw runtime error if not.
        if (frame.freq_id != (unsigned)freq_id) {
            throw std::runtime_error("Cannot downsample stream with more than one frequency.");
        }

        // Get position within accumulation window
        /*
        wdw_pos = (fpga_seq_start % wdw_len) / frame.frame_length_fpga_ticks;
        DEBUG("wdw_pos: {:d}; wdw_end: {:d}", wdw_pos, wdw_end);
        DEBUG("ERA: {:f}; ERA_target: {:f} ERA_wdw_len: {:f}", frame.era_deg,
                era_target, wdw_len_era);
        */
        DEBUG("ERA: {:f}; ERA_target: {:f}; ERA_bin_lo: {:f}; ERA_bin_hi: {:f}",
                frame.era_deg, era_target, era_deg_lo, era_deg_hi);

        // Don't start accumulating unless at the start of window
        // TODO: Re-implement for ERA bins.
        /*
        if (nframes == 0 and wdw_pos != 0) {
            // Skip this frame
            skipped_frame_counter.labels({std::to_string(freq_id), "alignment"}).inc();
            in_buf->mark_frame_empty(unique_name, frame_id++);
            continue;
        }
        */

        // Start a new accumulation
        if (nframes == 0) { 
            // Update window
            /*
            wdw_end = fpga_seq_start + wdw_len;
            era_target = frame.era_deg
                         - 0.5 * frame.frame_length_fpga_ticks*seq_to_era
                         + 0.5 * wdw_len_era;
            if(era_target > 360)
                era_target -= 360;
            */
            
            era_bin_idx = (unsigned int) (num_bins_per_rotation
                                      * frame.era_deg / 360.0);
            era_deg_lo = era_bin_idx * (360.0 / num_bins_per_rotation);
            era_deg_hi = (era_bin_idx + 1) * (360.0 / num_bins_per_rotation);
            era_target = 0.5*(era_deg_lo + era_deg_hi);

            // Wait for an empty frame
            if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
                break;
            }

            // Copy frame into output buffer
            auto output_frame =
                N2FrameView::copy_frame(in_buf, frame_id,
                                        out_buf, output_frame_id);

            // Increase the total frame length
            //output_frame.frame_length_fpga_ticks *= nsamp;

            // Set the target ERA.
            output_frame.era_deg = era_target;
            output_frame.xp_as = xp_target;
            output_frame.yp_as = yp_target;

            if(do_fringestop) {
                tel.fringestop_phases_1d(freq_Hz, frame.era_deg, frame.xp_as,
                                         frame.yp_as, era_target, xp_target,
                                         yp_target, fringe_phase);

                size_t idx = 0;
                for(size_t i = 0; i < num_elements; i++) {
                    for(size_t j = i; j < num_elements; j++) {

                        size_t d_i = i % num_dishes;
                        size_t d_j = j % num_dishes;
                        output_frame.vis[idx] *= 
                            fringe_phase[d_i] * std::conj(fringe_phase[d_j]);

                        idx++;
                    }
                }
            }

            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] = 1. / output_frame.weight[i];
            }

            // Go to next frame
            nframes += 1;
            in_buf->mark_frame_empty(unique_name, frame_id++);
            continue;
        }

        auto output_frame = N2FrameView(out_buf, output_frame_id);

        // Check we are still in accumulation window
        //if (fpga_seq_start < wdw_end) {
        if (frame.era_deg >= era_deg_lo && frame.era_deg < era_deg_hi) {
            
            //Recalculate fringestop phases
            if(do_fringestop)
                tel.fringestop_phases_1d(freq_Hz, frame.era_deg, frame.xp_as,
                                         frame.yp_as, era_target, xp_target,
                                         yp_target, fringe_phase);

            // Accumulate contents of buffer
            size_t idx = 0;
            for(size_t i = 0; i < num_elements; i++) {
                for(size_t j = i; j < num_elements; j++) {

                    size_t d_i = i % num_dishes;
                    size_t d_j = j % num_dishes;

                    std::complex<double> w_doub = fringe_phase[d_i]
                                        * std::conj(fringe_phase[d_j]);
                    N2::cfloat w_fs{(float) w_doub.real(),
                                    (float) w_doub.imag()};

                    //DEBUG("fringestop phase: {}-{}: {}+{}i", i, j,
                    //        w_fs.real(), w_fs.imag());

                    output_frame.vis[idx] += w_fs * frame.vis[idx];
                    idx++;
                }
            }
            
            // average inverse weights, i.e. variance
            for (size_t i = 0; i < nprod; i++) {
                output_frame.weight[i] += 1. / frame.weight[i];
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] += frame.eval[i];
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] += frame.evec[k];
                }
            }
            output_frame.erms += frame.erms;

            // Accumulate integration totals
            output_frame.n_valid_fpga_ticks_in_frame += frame.n_valid_fpga_ticks_in_frame;
            output_frame.n_rfi_fpga_ticks += frame.n_rfi_fpga_ticks;
            output_frame.frame_length_fpga_ticks += frame.frame_length_fpga_ticks;

            // Move to next frame
            nframes += 1;
            in_buf->mark_frame_empty(unique_name, frame_id++);

        } else {

            double output_age = 1.0e-9 * (frame.frame_start_time_ns
                                          - output_frame.frame_start_time_ns);
            if (output_age > max_age) {
                skipped_frame_counter.labels({std::to_string(freq_id), "age"}).inc();
                nframes = 0;
                continue;
            }

            // Otherwise, stop accumulating
            for (size_t i = 0; i < nprod; i++) {
                output_frame.vis[i] /= nframes;
                // extra factor of nsamp for sample variance
                output_frame.weight[i] = nframes * nframes / output_frame.weight[i];
            }
            for (uint32_t i = 0; i < num_eigenvectors; i++) {
                output_frame.eval[i] /= nframes;
                for (uint32_t j = 0; j < num_elements; j++) {
                    int k = i * num_elements + j;
                    output_frame.evec[k] /= nframes;
                }
            }
            output_frame.erms /= nframes;

            DEBUG("Output frame - num_elements: {:d}",
                    output_frame.num_elements);

            char *addr0 = (char *)&(output_frame.vis[0]);
            DEBUG("Output frame Struct - vis: {}, w: {}, flags: {}, eval: {}, evec: {}, emeth: {}, erms: {}, gain: {}",
                    (char *)&(output_frame.vis[0])  - addr0,
                    (char *)&(output_frame.weight[0]) - addr0,
                    (char *)&(output_frame.flags[0]) - addr0,
                    (char *)&(output_frame.eval[0]) - addr0,
                    (char *)&(output_frame.evec[0]) - addr0,
                    (char *)&(output_frame.emethod) - addr0,
                    (char *)&(output_frame.erms) - addr0,
                    (char *)&(output_frame.gain[0]) - addr0);
            // mark as full
            out_buf->mark_frame_full(unique_name, output_frame_id++);
            // reset accumulation and move on, starting with this frame
            nframes = 0;
        }
    }
}
