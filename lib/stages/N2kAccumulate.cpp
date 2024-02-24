#include "N2kAccumulate.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for register_producer, Buffer, allocate_new_metadata_object
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for chordMetadata, get_fpga_seq_num
#include "configUpdater.hpp"     // for configUpdater
#include "factory.hpp"           // for FACTORY
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO, logLevel, DEBUG
#include "metadata.h"            // for metadataContainer
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Metrics
#include "version.h"             // for get_git_commit_hash
#include "visUtil.hpp"           // for prod_ctype, frameID, modulo, input_ctype, operator+

#include <algorithm>  // for copy, max, fill, copy_backward, equal, transform
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cmath>      // for pow
#include <complex>    // for operator*, complex
#include <sys/time.h> // for TIMEVAL_TO_TIMESPEC
#include <time.h>     // for size_t, timespec
#include <vector>     // for vector, vector<>::iterator, __alloc_traits<>::value_type


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(N2kAccumulate);


N2kAccumulate::N2kAccumulate(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2kAccumulate::main_thread, this)),
    skipped_frame_counter(Metrics::instance().add_counter(
        "kotekan_N2kaccumulate_skipped_frame_total", unique_name, {"freq_id", "reason"})) {

    auto& tel = Telescope::instance();

    // Fetch configuration

    // number of frequencies in frame
    _num_freq_in_frame = config.get<int32_t>(unique_name, "num_local_freq");

    // sampling information
    _n_fpga_samples_per_N2k_frame = config.get<int32_t>(unique_name, "samples_per_data_set"); // same as in output frame, just coarsened
    _n_fpga_samples_N2k_integrates_for = config.get<int32_t>(unique_name, "sub_integration_ntime");
    _n_vis_samples_per_N2k_output_frame = _n_fpga_samples_per_N2k_frame / _n_fpga_samples_N2k_integrates_for;
    
    _n_vis_samples_per_in_frame = _n_vis_samples_per_N2k_output_frame;
    _n_fpga_samples_per_vis_sample = _n_fpga_samples_N2k_integrates_for;
    _in_frame_duration_nsec = (uint64_t) _n_fpga_samples_per_N2k_frame * (uint64_t) tel.seq_length_nsec();
    _in_frame_vis_duration_nsec = _in_frame_duration_nsec / _n_vis_samples_per_in_frame;

    // Number of products sent by the GPU
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_vis_products = _num_freq_in_frame*_num_elements*_num_elements; // right now, N2k sends over a full matrix
    // TODO: Only loop through half of blocked matrix
    // size_t vis_block_size = config.get<size_t>(unique_name, "block_size");
    // size_t n_vis_blocks = num_elements / vis_block_size;
    // _num_out_vis_products = _num_freq_in_frame * n_vis_blocks * (n_vis_blocks + 1) * block_size * block_size / 2;

    // Initializing these here using the computed _num_vis_products
    _vis = std::vector<int32_t>(2 * _num_vis_products, 0); // vis with complex as 2 ints
    _vis_even = std::vector<int32_t>(2 * _num_vis_products, 0); // store even vis matrix for differencing
    _weights = std::vector<int32_t>(_num_vis_products, 0); // real-valued weights
    // number of fpga samples, per frequency, in frame
    _n_valid_fpga_samples_in_vis = std::vector<int32_t>(_num_freq_in_frame, 0);
    _n_valid_fpga_samples_in_vis_even = std::vector<int32_t>(_num_freq_in_frame, 0);
    _n_valid_sample_diff_sq_sum = std::vector<int32_t>(_num_freq_in_frame, 0);

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
}

void N2kAccumulate::main_thread() {

    DEBUG("Starting N2kAccumulate with ...");

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    timespec output_ts;
    timespec_get(&output_ts, TIME_UTC);
    uint64_t t_output = ts_to_uint64(output_ts);

    size_t vis_samples_in_out_frame = 0;

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;
        int32_t* input = (int32_t*)in_frame;
        size_t in_frame_num = get_fpga_seq_num(in_buf, in_frame_id) / _n_fpga_samples_per_N2k_frame;
        chordMetadata* frame_metadata = (chordMetadata*) in_buf->get_metadata(in_frame_id);

        // Start and end times of this frame
        bool gps_time_enabled = false;
        // Here we'll just use raw nanoseconds
        uint64_t t_frame_s;
        if (gps_time_enabled) {
            t_frame_s = ts_to_uint64(frame_metadata->chime.gps_time);
        } else {
            // If GPS time is not set, fall back to system time.
            timespec ts;
            TIMEVAL_TO_TIMESPEC( &frame_metadata->chime.first_packet_recv_time,
                &ts );
            t_frame_s = ts_to_uint64(ts);
        }
        // uint64_t t_frame_e = t_frame_s + _in_frame_duration_nsec;

        // Accumulation
        for (size_t vis_samp_n = 0; vis_samp_n < _n_vis_samples_per_in_frame; ++vis_samp_n) {

            // Start and end times of the visibility matrix sample
            uint64_t t_vis_s = t_frame_s + vis_samp_n*_in_frame_vis_duration_nsec;
            // uint64_t t_vis_e = t_vis_s + _in_frame_vis_duration_nsec;

            // "absolute" vis sample number
            size_t vis_sample_num_abs = in_frame_num*_n_vis_samples_per_in_frame + vis_samp_n;
            
            DEBUG("Acumulating new visibility sample ({:d} of {:d} in frame).",
                vis_samp_n, _n_vis_samples_per_in_frame );
            // DEBUG("   Times are [start, end, out, num] = [{:d}, {:d}, {:d}, {:d}]",
            //     t_vis_s, t_vis_e, t_output, vis_sample_num_abs );


            // Finalize accumulation if the visibility elements are past the output time...
            //  end on an odd frame too so we accumulate weights.
            if(t_vis_s > t_output
                and vis_sample_num_abs % 2 == 1) {

                DEBUG("Finishing N2kAccumulate output frame. Accumulated {:d} visibility samples.",
                    vis_samples_in_out_frame);
                output_and_reset( out_frame_id );

                t_output += 1000000000L; // TODO: Make this a config parameter
                vis_samples_in_out_frame = 0;
            }

            // Actual accumulation
            // TODO: optimize looping?
            for (size_t f = 0; f < _num_freq_in_frame; ++f) {

                for (size_t d = 0; d < 2*_num_vis_products; ++d) {
                    _vis[d] += input[d];
                } // d

                int32_t lost_in_vis_sample = frame_metadata->lost_fpga_samples[f][vis_samp_n];
                int32_t valid_fpga_samples = _n_fpga_samples_per_vis_sample - lost_in_vis_sample;
                _n_valid_fpga_samples_in_vis[f] += valid_fpga_samples;

                // If we're working on an even sample, store it for differencing with an
                // odd sample. Potential optimization: copying vis_even is only really
                // necessary if we've started accumulating a new frame
                if (vis_sample_num_abs % 2 == 0) {
                    // copy input buffer vis matrix to vis_even
                    std::copy(input, input + 2*_num_vis_products, _vis_even.begin());
                    _n_valid_fpga_samples_in_vis_even[f] = valid_fpga_samples;
                } else {
                    for (size_t d = 0; d < _num_vis_products; ++d) {
                        int32_t dr = _vis[2*d + 0] - _vis_even[2*d + 0];
                        int32_t di = _vis[2*d + 1] - _vis_even[2*d + 1];
                        _weights[d] += (dr * dr + di * di);
                    } // d

                    float samples_diff = valid_fpga_samples - _n_valid_fpga_samples_in_vis_even[f];
                    _n_valid_sample_diff_sq_sum[f] += samples_diff*samples_diff;
                } // if even

            } // f
            vis_samples_in_out_frame++;

        } // t (vis samples in frame)

        // Advance to the next frame
        in_buf->mark_frame_empty(unique_name, in_frame_id++);
    }
}


bool N2kAccumulate::output_and_reset( frameID &out_frame_id )
{
    // Different frame for each frequency

    // Loop over frequency
    for (size_t f = 0; f < _num_freq_in_frame; ++f) {

        if (out_buf->wait_for_empty_frame(unique_name, out_frame_id) == nullptr) {
            return false;
        }

        auto out_frame = VisFrameView::create_frame_view(out_buf, out_frame_id,
            _num_elements, _num_vis_products, 0, true);
        // Any need to adjust metadata?

        float ns = _n_valid_fpga_samples_in_vis[f]; // ns = "number of samples"
        float ins = (ns != 0.0) ? (1.0 / ns) : 0.0;

        // Copy data into buffer
        // TODO: For now just copy over. Eventually this should change to (possibly) restructure
        // the visibility matrix.
        for (size_t d = 0; d < _num_vis_products; ++d) {
            // Populate the visibility matrix
            cfloat v = {(float)_vis[2*d+1], (float)_vis[2*d+0]}; // TODO: conjugate or no? What does downstream expect?
            out_frame.vis[d] = ins * v;
            // de-bias and populate the weights matrix (with the inverse variance)
            _weights[d] -= std::norm(v) * _n_valid_sample_diff_sq_sum[f] / ns / ns;
            out_frame.weight[d] = ns*ns / _weights[d];
        }

        out_buf->mark_frame_full(unique_name, out_frame_id++);

        std::fill(_vis.begin(), _vis.end(), 0);
        std::fill(_weights.begin(), _weights.end(), 0);
        std::fill(_n_valid_fpga_samples_in_vis.begin(), _n_valid_fpga_samples_in_vis.end(), 0);
        std::fill(_n_valid_sample_diff_sq_sum.begin(), _n_valid_sample_diff_sq_sum.end(), 0);

    }

    return true;
}


// Checks to do...
// Make sure output buffer has enough frames (>= # frequencies) and are sized correctly?

